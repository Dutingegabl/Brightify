"""Reproducible, inspectable runs around Brightify's MCPL/KDSource workflow."""

import json
import math
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from brightify import data_augmentation, data_comparison


DEFAULT_COMPARISON_COLUMNS = ("ekin", "x", "y", "z", "ux", "uy", "uz")
METADATA_VERSION = 1


def _json_safe(value, active=None):
    """Convert common Python and NumPy values to JSON-compatible values."""
    if active is None:
        active = set()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)

    cls = type(value)
    descriptor = {"__type__": "{}.{}".format(cls.__module__, cls.__qualname__)}
    object_id = id(value)
    if object_id in active:
        descriptor["repr"] = repr(value)
        return descriptor

    active.add(object_id)
    try:
        if isinstance(value, dict):
            return {
                str(key): _json_safe(item, active) for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            if isinstance(value, set):
                items.sort(key=repr)
            return [_json_safe(item, active) for item in items]
        if hasattr(value, "tolist"):
            return _json_safe(value.tolist(), active)
        if hasattr(value, "item"):
            return _json_safe(value.item(), active)
        try:
            attributes = vars(value)
        except TypeError:
            attributes = None
        if attributes:
            descriptor["attributes"] = _json_safe(attributes, active)
        else:
            descriptor["repr"] = repr(value)
        return descriptor
    finally:
        active.remove(object_id)


def _file_info(path):
    path = Path(path).expanduser().resolve()
    info = {"path": str(path)}
    if path.is_file():
        stat = path.stat()
        info.update({"size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns})
    return info


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


class RunResult:
    """Artifacts and stage records produced by a :class:`Run`."""

    def __init__(
        self,
        model_path=None,
        bws_path=None,
        generated_mcpl=None,
        stages=None,
        timings=None,
        comparisons=None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.bws_path = Path(bws_path) if bws_path else None
        self.generated_mcpl = Path(generated_mcpl) if generated_mcpl else None
        self.stages = dict(stages or {})
        self.timings = dict(timings or {})
        self.comparisons = list(comparisons or [])

    @property
    def status(self):
        statuses = [stage.get("status") for stage in self.stages.values()]
        if "failed" in statuses:
            return "failed"
        if "running" in statuses:
            return "running"
        if "completed" in statuses:
            return "completed"
        return "pending"

    def to_dict(self):
        return {
            "status": self.status,
            "model_path": str(self.model_path) if self.model_path else None,
            "bws_path": str(self.bws_path) if self.bws_path else None,
            "generated_mcpl": (
                str(self.generated_mcpl) if self.generated_mcpl else None
            ),
            "stages": _json_safe(self.stages),
            "timings": _json_safe(self.timings),
            "comparisons": _json_safe(self.comparisons),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            model_path=data.get("model_path"),
            bws_path=data.get("bws_path"),
            generated_mcpl=data.get("generated_mcpl"),
            stages=data.get("stages"),
            timings=data.get("timings"),
            comparisons=data.get("comparisons"),
        )


class ComparisonResult:
    """Metrics and provenance for comparing two MCPL datasets."""

    def __init__(
        self,
        run_id,
        reference_id,
        generated_path,
        reference_path,
        columns,
        scaling_factor,
        metrics,
        created_at=None,
    ):
        self.run_id = run_id
        self.reference_id = reference_id
        self.generated_path = Path(generated_path)
        self.reference_path = Path(reference_path)
        self.columns = tuple(columns)
        self.scaling_factor = scaling_factor
        self.metrics = metrics
        self.created_at = created_at or _timestamp()

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "reference_id": self.reference_id,
            "generated_path": str(self.generated_path),
            "reference_path": str(self.reference_path),
            "columns": list(self.columns),
            "scaling_factor": _json_safe(self.scaling_factor),
            "metrics": _json_safe(self.metrics),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            run_id=data["run_id"],
            reference_id=data.get("reference_id"),
            generated_path=data["generated_path"],
            reference_path=data["reference_path"],
            columns=data["columns"],
            scaling_factor=data["scaling_factor"],
            metrics=data["metrics"],
            created_at=data.get("created_at"),
        )

    def plot(self, metric, columns=None, ax=None):
        """Plot one metric's per-column values and return ``(figure, axes)``."""
        import matplotlib.pyplot as plt

        values = None
        for group in ("univariate", "multivariate"):
            group_metrics = self.metrics.get(group, {})
            if metric in group_metrics:
                values = group_metrics[metric]
                break
        if values is None:
            raise KeyError("Metric {!r} is not present in this comparison".format(metric))

        if isinstance(values, dict):
            selected = columns if columns is not None else sorted(values)
            labels = []
            scores = []
            for column in selected:
                value = values.get(column)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    labels.append(str(column))
                    scores.append(float(value))
            if not scores:
                raise ValueError(
                    "Metric {!r} has no numeric values for the selected columns".format(
                        metric
                    )
                )
        elif isinstance(values, (int, float)) and math.isfinite(float(values)):
            labels = [metric]
            scores = [float(values)]
        else:
            raise ValueError("Metric {!r} does not contain plottable values".format(metric))

        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure
        ax.bar(labels, scores)
        ax.set_ylabel(metric)
        ax.set_title("{}: {} vs {}".format(metric, self.run_id, self.reference_id))
        figure.tight_layout()
        return figure, ax


class Run:
    """One reproducible fit/resample run and its persisted results.

    ``run_dir`` is dedicated to this run. Its ``run.json`` manifest is updated
    as stages finish; MCPL and XML files remain separate, normal artifacts.
    """

    def __init__(
        self,
        input_mcpl,
        run_dir,
        fit_parameters=None,
        name=None,
        run_id=None,
    ):
        self.id = run_id or uuid.uuid4().hex
        self.name = name or self.id
        self.input_mcpl = Path(input_mcpl).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.model_path = self.run_dir / "model.xml"
        self.bws_path = self.run_dir / "model.bws"
        self.generated_mcpl_path = self.run_dir / "generated.mcpl.gz"
        self.fit_parameters = dict(fit_parameters or {})
        self.resample_parameters = {}
        self.created_at = _timestamp()
        self.result = RunResult()
        self.metadata_path = self.run_dir / "run.json"

        if self.metadata_path.exists():
            raise FileExistsError(
                "Run manifest already exists: {}".format(self.metadata_path)
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.save()

    @property
    def status(self):
        return self.result.status

    def to_dict(self):
        return {
            "metadata_version": METADATA_VERSION,
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "input_mcpl": _file_info(self.input_mcpl),
            "run_dir": str(self.run_dir),
            "fit_parameters": _json_safe(self.fit_parameters),
            "resample_parameters": _json_safe(self.resample_parameters),
            "result": self.result.to_dict(),
        }

    def save(self):
        """Atomically write the run manifest and return its path."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".run-", suffix=".json.tmp", dir=str(self.run_dir)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(self.to_dict(), file, indent=2, sort_keys=True)
                file.write("\n")
            os.replace(temporary_path, str(self.metadata_path))
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise
        return self.metadata_path

    @classmethod
    def load(cls, manifest_path):
        """Load a manifest for inspection and comparisons.

        Arbitrary runtime objects in parameters, such as KDSource geometry
        instances, are preserved as descriptive JSON data, not reconstructed.
        """
        manifest_path = Path(manifest_path).expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        version = data.get("metadata_version")
        if version != METADATA_VERSION:
            raise ValueError(
                "Unsupported run metadata version: {!r}".format(version)
            )

        run = cls.__new__(cls)
        run.id = data["id"]
        run.name = data.get("name", run.id)
        run.created_at = data["created_at"]
        run.input_mcpl = Path(data["input_mcpl"]["path"])
        run.run_dir = Path(data.get("run_dir", manifest_path.parent))
        run.fit_parameters = data.get("fit_parameters", {})
        run.resample_parameters = data.get("resample_parameters", {})
        result_data = data.get("result", {})
        run.result = RunResult.from_dict(result_data)
        run.model_path = run.result.model_path or run.run_dir / "model.xml"
        run.bws_path = run.result.bws_path or run.run_dir / "model.bws"
        run.generated_mcpl_path = (
            run.result.generated_mcpl or run.run_dir / "generated.mcpl.gz"
        )
        run.metadata_path = manifest_path
        return run

    def _begin_stage(self, name):
        stage = {"status": "running", "started_at": _timestamp()}
        self.result.stages[name] = stage
        self.save()
        return time.perf_counter()

    def _finish_stage(self, name, started, error=None):
        stage = self.result.stages[name]
        stage["finished_at"] = _timestamp()
        stage["duration_seconds"] = time.perf_counter() - started
        stage["status"] = "failed" if error is not None else "completed"
        if error is not None:
            stage["error"] = "{}: {}".format(type(error).__name__, error)
        self.result.timings[name] = stage["duration_seconds"]
        self.save()

    def fit(self, **overrides):
        """Fit a KDSource model using ``data_augmentation.fit``."""
        if not self.input_mcpl.is_file():
            raise FileNotFoundError("Input MCPL file not found: {}".format(self.input_mcpl))
        if self.model_path.exists():
            raise FileExistsError("Model file already exists: {}".format(self.model_path))
        if self.bws_path.exists():
            raise FileExistsError("Bandwidths file already exists: {}".format(self.bws_path))


        self.fit_parameters.update(overrides)
        started = self._begin_stage("fit")
        try:
            model_path, bws_path = data_augmentation.fit(
                mcpl_file_path=str(self.input_mcpl),
                xml_file_path=str(self.model_path),
                bw_file_path=str(self.bws_path),
                **self.fit_parameters
            )
        except Exception as error:
            self._finish_stage("fit", started, error)
            raise
        self.result.model_path = Path(model_path)
        self.result.bws_path = Path(bws_path) if bws_path is not None else None
        self._finish_stage("fit", started)
        return self.result.model_path, self.result.bws_path

    def resample(self, amount, rng=None, overwrite_if_exists=False):
        """Generate an MCPL file from this run's fitted model."""
        if self.result.model_path is None or not self.result.model_path.is_file():
            raise RuntimeError("Fit a model successfully before resampling")
        if self.generated_mcpl_path.exists() and not overwrite_if_exists:
            raise FileExistsError(
                "Generated MCPL file already exists: {}".format(self.generated_mcpl_path)
            )

        self.resample_parameters = {"amount": amount, "rng": _json_safe(rng)}
        started = self._begin_stage("resample")
        try:
            data_augmentation.resample(
                xml_file_path=str(self.result.model_path),
                mcpl_file_path=str(self.generated_mcpl_path),
                amount=amount,
                rng=rng,
                overwrite_if_exists=overwrite_if_exists,
            )
        except Exception as error:
            self._finish_stage("resample", started, error)
            raise
        self.result.generated_mcpl = self.generated_mcpl_path
        self._finish_stage("resample", started)
        return self.result.generated_mcpl

    def compare_to(
        self,
        reference,
        columns=None,
        scaling_factor=1.0,
        verbose=False,
    ):
        """Compare generated data to another run or an MCPL file.

        For a reference ``Run``, its generated MCPL is preferred, falling back
        to its input MCPL when it has not generated an output.
        """
        generated_path = self.result.generated_mcpl
        if generated_path is None or not generated_path.is_file():
            raise RuntimeError("Resample data successfully before comparing this run")

        if isinstance(reference, Run):
            reference_path = reference.result.generated_mcpl or reference.input_mcpl
            reference_id = reference.id
        else:
            reference_path = Path(reference).expanduser().resolve()
            reference_id = None
        if not Path(reference_path).is_file():
            raise FileNotFoundError("Reference MCPL file not found: {}".format(reference_path))

        selected_columns = tuple(
            DEFAULT_COMPARISON_COLUMNS if columns is None else columns
        )
        stage_name = "compare_{}".format(len(self.result.comparisons) + 1)
        started = self._begin_stage(stage_name)
        try:
            columns_to_read = set(selected_columns)
            generated_data = data_augmentation.read_df_from_mcpl(
                generated_path, columns=columns_to_read
            )
            reference_data = data_augmentation.read_df_from_mcpl(
                reference_path, columns=columns_to_read
            )
            metrics = data_comparison.evaluate_all_metrics(
                df_p=generated_data,
                df_q=reference_data,
                columns=set(selected_columns),
                scaling_factor=scaling_factor,
                verbose=verbose,
            )
            comparison = ComparisonResult(
                run_id=self.id,
                reference_id=reference_id or str(reference_path),
                generated_path=generated_path,
                reference_path=reference_path,
                columns=selected_columns,
                scaling_factor=scaling_factor,
                metrics=metrics,
            )
            self.result.comparisons.append(comparison.to_dict())
        except Exception as error:
            self._finish_stage(stage_name, started, error)
            raise
        self._finish_stage(stage_name, started)
        return comparison

    def plot_metrics(self, metric, comparison_index=-1, columns=None, ax=None):
        """Plot a metric from a saved comparison and return ``(figure, axes)``."""
        try:
            data = self.result.comparisons[comparison_index]
        except IndexError:
            raise IndexError("This run has no comparison at that index")
        return ComparisonResult.from_dict(data).plot(metric, columns=columns, ax=ax)


def compare_runs(run, reference, columns=None, scaling_factor=1.0, verbose=False):
    """Convenience function equivalent to ``run.compare_to(reference, ...)``."""
    return run.compare_to(
        reference=reference,
        columns=columns,
        scaling_factor=scaling_factor,
        verbose=verbose,
    )


__all__ = [
    "ComparisonResult",
    "DEFAULT_COMPARISON_COLUMNS",
    "Run",
    "RunResult",
    "compare_runs",
]
