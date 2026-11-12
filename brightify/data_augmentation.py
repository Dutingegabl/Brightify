# This module should allow the brightify user to:
# I. Check the statistics of his original input data, and if it's sufficient for the desired resolution of the brightify calculations.
# II. Find out the geometry of his data automatically, or specify it manually.
# III. Generate (resample) more "synthetic" data that is similar to the original input data, using kdsource, in order to increase the statistics.
# IV. Check how close the generated data is to other data (in particular to the original input data, i.e. if it converges)

# required methods
# 1. read mcpl into dataframe, only the needed columns
# 2. write dataframe to mcpl
# 3. detect geometry of data
# 4. perform kds fit based on a geometry and other parameters
# 5. resample to mcpl based on original mcpl and fit output
# 6. comparing of two dataframes, using of different metrics:
# - Wasserstein Distance (univariate)
# - Kullback-Leibler-Divergence or Jensen-Shannon Distance (both univariate)
# - Classifier-Two-Sample-Test (C2ST) (multivariate)

# ----- IMPORTS -----
import sys
import os
import multiprocessing
from pathlib import Path
import tempfile
import mcpl
import pandas as pd
import numpy as np
import kdsource.api as kds
import scipy.spatial.transform as st
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ----- CONSTANTS -----

DEFAULT_COLUMN_NAMES = ["ekin",
                   "pdgcode",
                   "x",
                   "y",
                   "z",
                   "ux",
                   "uy",
                   "uz",
                   "weight",
                   "time",
                   "polx",
                   "poly",
                   "polz",
                   "userflags"]

DEFAULT_VALUES = {
    "ekin": 0.0,
    "pdgcode": 2112, # neutron
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "ux": 0.0, # note that if all ux, uy and uz are zero, mcpl file cannot be written using kds
    "uy": 0.0,
    "uz": 0.0,
    "weight": 1.0,
    "time": 0.0,
    "polx": None, # for polx, poly, polz and userflags, the mcpl writing function allows None value
    "poly": None,
    "polz": None,
    "userflags": None,
}

COLUMN_ALIASES_LOWERCASE = {
    "ekin": ["ekin", "e", "energy", "ekin_mev", "e_mev", "energy_mev", "mev"],
    "pdgcode": ["pdgcode", "pdg_code", "pdg", "pid", "p_id", "particle_id", "id", "particle", "kf"],
    "x": ["x", "pos_x", "posx", "x_pos", "xpos"],
    "y": ["y", "pos_y", "posy", "y_pos", "ypos"],
    "z": ["z", "pos_z", "posz", "z_pos", "zpos"],
    "ux": ["ux", "u_x", "x_u", "xu", "vx", "v_x", "x_v", "xv", "dir_x", "dirx", "x_dir", "xdir", "u", "velx", "vel_x", "xvel", "x_vel"],
    "uy": ["uy", "u_y", "y_u", "yu", "vy", "v_y", "y_v", "yv", "dir_y", "diry", "y_dir", "ydir", "v", "vely", "vel_y", "yvel", "y_vel"],
    "uz": ["uz", "u_z", "z_u", "zu", "vz", "v_z", "z_v", "zv", "dir_z", "dirz", "z_dir", "zdir", "w", "velz", "vel_z", "zvel", "z_vel"],
    "weight": ["weight", "w", "wt", "wgt", "weights", "weigt", "weigts"],
    "time": ["time", "t", "t_sec", "sec", "time_ns", "ns"],
    "polx": ["polx", "pol_x", "x_pol", "xpol"],
    "poly": ["poly", "pol_y", "y_pol", "ypol"],
    "polz": ["polz", "pol_z", "z_pol", "zpol"],
    "userflags": ["userflags", "user", "flags", "flag", "usr", "f"]
}

# ----- GENERAL -----

def process_func(func, **kwargs):
    """
    Run func with kwargs as parameters in a separate process,
    to guard against python kernel crashes cause e.g. by calling C functions.
    If func crashes, an Exception is raised that reveals the console output of func.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as log_file:
        log_file_name = log_file.name
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker_func, args=(queue, log_file_name, func), kwargs=kwargs)
    p.start()
    p.join() # Wait for the process to finish

    with open(log_file_name, 'r') as f:
       log = f.read().strip()
    # Check if the process crashed (exitcode != 0)
    if p.exitcode != 0:
        msg = f"The processing of 'func' crashed fatally (exit code {p.exitcode})."
        if log:
            msg += f"\n'func' log:\n{log}"
        raise RuntimeError(msg)
    if not queue.empty():
        status, result = queue.get()
        if status == "SUCCESS":
            return result
        else:
            msg = f"Error: {result}"
            if log:
                msg += f"\n'func' log:\n{log}"
            raise Exception(msg)
    raise RuntimeError("Unknown failure.")

def _worker_func(queue, log_file_name, func, **kwargs):
    """Helper for process_func().
    Runs func with kwargs in a separate process. If the func crashes, only this process dies.
    - queue: multiprocessing Queue
    - log_file_name: name of (temporary) file where to put the terminal output of func
    - func: method to process with kwargs as parameters
    - kwargs: parameters for func

    Returns
    --
    - Output of func (if no crash). If crash, then an exception is raised carrying the stderr and stdout (if exist).
    """
    with open(log_file_name, 'w') as log_file:
        sys.stderr.flush()
        sys.stdout.flush()        
        # Save original file descriptors so we can restore them later
        orig_stderr = os.dup(2)
        orig_stdout = os.dup(1)        
        # Redirect OS-level stderr (2) and stdout (1) straight into our file
        os.dup2(log_file.fileno(), 2)
        os.dup2(log_file.fileno(), 1)
    try:
        result = func(**kwargs)
        queue.put(("SUCCESS", result))
    except Exception as e:
        queue.put(("ERROR", str(e)))
    finally:
        sys.stderr.flush()
        sys.stdout.flush()
        os.dup2(orig_stderr, 2)
        os.dup2(orig_stdout, 1)
        os.close(orig_stderr)
        os.close(orig_stdout)

# ----- 1. -----
def read_mcpl_to_df(mcpl_file_path, n: int =-1, columns: set = {"ekin","pdgcode","x","y","z","ux","uy","uz","weight","time","polx","poly","polz","userflags"}):
    """
    Read up to n particles from the mcpl file into a pandas dataframe. n==-1 means read all.
    Only the specified columns are read.
    """
    columns = sorted(columns, key=lambda col: DEFAULT_COLUMN_NAMES.index(col))
    blocklength = 100000
    gen_file = mcpl.MCPLFile(str(Path(mcpl_file_path).absolute()), blocklength=blocklength)
    datas = []
    for i, p in enumerate(gen_file.particle_blocks):
        mapping = {
            "ekin": p.ekin,  # Kinetic energy
            "pdgcode": p.pdgcode,  # Particle PDG code
            "x": p.x,  # x-position
            "y": p.y,  # y-position
            "z": p.z,  # z-position
            "ux": p.ux,  # Velocity in x-direction
            "uy": p.uy,  # Velocity in y-direction
            "uz": p.uz,  # Velocity in z-direction
            "weight": p.weight,  # Weight of the particle
            "time": p.time,
            "polx": p.polx,
            "poly": p.poly,
            "polz": p.polz,
            "userflags": p.userflags,
            }
        data = pd.DataFrame({col:mapping[col] for col in columns if col in mapping.keys()})
        datas.append(data)
        if n != -1 and (i + 1) * blocklength >= n:
            break
    data = pd.concat(datas, ignore_index=True)
    if n != -1 and len(data) > n:
        data = data[:n]
    return data

# ----- 2. -----
def write_df_to_mcpl(data: pd.DataFrame, mcpl_file_path, double_prec=False, force=False):
    """
    Write dataframe to a mcpl file using kdsource.
    Missing columns will be set to default values found in DEFAULT_VALUES,
    except polx, poly, polz and userflags, which may be set to None in the mcpl file format.
    Note that writing will fail if e.g. the direction vectors are not normalized or entirely zero, or energy is negative etc.
    """
    write_data = dict()
    N = len(data)
    for col, default in DEFAULT_VALUES.items():
        if col in data.columns:
            write_data[col] = data[col]
        elif default is not None:
            write_data[col] = np.full(N, default)
        else:
            write_data[col] = None
    # call the write_mcpl method using the process_func wrapper to avoid kernel crash and see reasons for failure (e.g. bad columns, negative energy etc.)
    process_func(kds.write_mcpl, filename=str(Path(mcpl_file_path).absolute()), **write_data, double_prec=double_prec, force=force)

# ----- 3. -----

def automatic_geometry(df, trasl=None, rot=None):
    """
    Analyzes an MCPL pandas DataFrame using user-supplied trasl and rot frames to:
      1. Transform input coordinates into the local coordinate system.
      2. Compute the 8-variable importance vector (ekin, x, y, z, ux, uy, uz, time) 
         in that aligned frame.
      3. Automatically select and instantiate the best matching Metrics from geom.py,
         returning a fully constructed Geometry object.
    """
    vars_list = ["ekin", "x", "y", "z", "ux", "uy", "uz", "time"]
    weights = df["weight"].values if "weight" in df.columns else np.ones(len(df))
    
    # ---------------------------------------------------------
    # 1. Apply User Transformation (trasl & rot) to Data First
    # ---------------------------------------------------------
    pos = df[["x", "y", "z"]].values.astype(float)
    dir_vecs = df[["ux", "uy", "uz"]].values.astype(float)
    
    # Parse rotation if given as array-like
    if rot is not None and not isinstance(rot, st.Rotation):
        rot = np.array(rot)
        if rot.shape == (4,):
            rot = st.Rotation.from_quat(rot)
        elif rot.shape == (3, 3):
            rot = st.Rotation.from_matrix(rot)
        elif rot.shape == (3,):
            rot = st.Rotation.from_rotvec(rot)
        else:
            raise ValueError("Invalid rot format supplied.")

    # Match Geometry transform behavior: apply trasl, then rot (using inverse=True to get local frame)
    if trasl is not None:
        trasl_arr = np.array(trasl).reshape(1, 3)
        pos -= trasl_arr
    
    if rot is not None:
        pos = rot.apply(pos, inverse=True)
        dir_vecs = rot.apply(dir_vecs, inverse=True)

    # Assemble aligned coordinates for analysis
    transformed_data = np.column_stack([
        df["ekin"].values,
        pos,
        dir_vecs,
        df["time"].values if "time" in df.columns else np.zeros(len(df))
    ])

    # ---------------------------------------------------------
    # 2. Compute 8-Variable Importance Vector in Aligned Frame
    # ---------------------------------------------------------
    importance_vector = []
    for i, var in enumerate(vars_list):
        col_data = transformed_data[:, i]
        w_mean = np.average(col_data, weights=weights)
        w_var = np.average((col_data - w_mean) ** 2, weights=weights)
        
        # If variance is effectively zero, importance is 0
        if w_var < 1e-14 or np.isnan(w_var):
            importance_vector.append(0.0)
        else:
            importance_vector.append(float(np.sqrt(w_var)))
            
    importance_vector = np.array(importance_vector)

    # Isolated aligned coordinate variables for metric configuration
    x_val = pos[:, 0]
    y_val = pos[:, 1]
    z_val = pos[:, 2]

    # ---------------------------------------------------------
    # 3. Detect and Select Optimal Metrics
    # ---------------------------------------------------------
    metrics_selected = []

    # --- A. Energy Dimension (ekin -> index 0) ---
    if importance_vector[0] == 0:
        metrics_selected.append(kds.geom.Energy())
    else:
        ekin_vals = df["ekin"].values
        if np.all(ekin_vals > 0) and (np.max(ekin_vals) / np.min(ekin_vals) > 10):
            E0 = float(np.max(ekin_vals))
            metrics_selected.append(kds.geom.Lethargy(E0=E0))
        else:
            metrics_selected.append(kds.geom.Energy())

    # --- B. Position & Direction Dimensions ---
    z_is_constant = importance_vector[3] == 0
    xy_active = np.any(importance_vector[1:3] > 0)
    dir_active = np.any(importance_vector[4:7] > 0)

    # Check for Guide structure (position and direction coupled)
    if xy_active and dir_active and not z_is_constant:
        xwidth = float(np.max(np.abs(x_val)) * 2)
        yheight = float(np.max(np.abs(y_val)) * 2)
        zmax = float(np.max(z_val))
        if xwidth > 0 and yheight > 0:
            metrics_selected.append(kds.geom.Guide(xwidth=xwidth, yheight=yheight, zmax=zmax))
    else:
        # Position Metric Selection
        if not xy_active and z_is_constant:
            metrics_selected.appendkds.geom.(SurfXY(xmin=0, xmax=0, ymin=0, ymax=0, z=float(np.mean(z_val))))
        elif z_is_constant:
            z_fixed = float(np.mean(z_val))
            xmin, xmax = float(np.min(x_val)), float(np.max(x_val))
            ymin, ymax = float(np.min(y_val)), float(np.max(y_val))
            
            r_vals = np.sqrt(x_val**2 + y_val**2)
            is_polar = np.std(r_vals) < (np.mean(r_vals) * 0.5) and np.min(r_vals) == 0
            
            if is_polar:
                rho_max = float(np.max(r_vals))
                metrics_selected.append(kds.geom.SurfR(rho_min=0, rho_max=rho_max, z=z_fixed))
            else:
                metrics_selected.append(kds.geom.SurfXY(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, z=z_fixed))
        else:
            xmin, xmax = float(np.min(x_val)), float(np.max(x_val))
            ymin, ymax = float(np.min(y_val)), float(np.max(y_val))
            zmin, zmax = float(np.min(z_val)), float(np.max(z_val))
            metrics_selected.append(kds.geom.Vol(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax))

        # Direction Metric Selection (if Guide wasn't added)
        if not isinstance(metrics_selected[-1], kds.geom.Guide):
            if not dir_active:
                metrics_selected.append(kds.geom.Isotrop())
            else:
                metrics_selected.append(kds.geom.PolarMu())

    # --- C. Time Dimension (time -> index 7) ---
    if importance_vector[7] > 0:
        time_vals = df["time"].values
        if np.all(time_vals > 0) and (np.max(time_vals) / np.min(time_vals) > 10):
            metrics_selected.append(kds.geom.Decade())
        else:
            metrics_selected.append(kds.geom.Time())

    # ---------------------------------------------------------
    # 4. Construct and Return Optimal Geometry
    # ---------------------------------------------------------
    optimal_geometry = kds.geom.Geometry(metrics=metrics_selected, trasl=trasl, rot=rot)
    
    return {
        "geometry": optimal_geometry,
        "importance_vector": importance_vector,
        "trasl": trasl,
        "rot": rot,
        "detected_metrics": [m.__class__.__name__ for m in metrics_selected]
    }

def basic_geometry():
    return kds.geom.Geometry(metrics=[kds.geom.Energy(), kds.geom.Time(), kds.geom.Vol(), kds.geom.Isotrop()], trasl=None, rot=None)

# ----- 4. -----
def fit(mcpl_file_path, xml_file_path,
        plist_params: dict = {"trasl":None, "rot":None, "switch_x2z":False, "set_params":True},
        geometry: kds.geom.Geometry = basic_geometry(),
        source_params: dict = {"bw":"auto", "J":1, "kernel":"gaussian"},
        fit_params: dict = {"N":-1, "skip":0, "scaling":None, "importance":None, "bw_method":None},
        fit_bw_kwargs: dict = {},
        save_params: dict = {"bwfile":None, "adjust_N":True}):
    """
    This makes a fit and saves a model xml file
    """
    plist = kds.PList(str(Path(mcpl_file_path).absolute()), readformat="mcpl", pt="n", **plist_params)
    source = kds.KDSource(plist, geometry, **source_params)
    source.fit(**fit_params, **fit_bw_kwargs)
    save_path = str(Path(xml_file_path).absolute())
    source.save(xmlfilename=save_path, **save_params)
    return save_path

# ----- 5. -----
def resample(xml_file_path, mcpl_file_path, amount,
             rng=None,
             overwrite_if_exists=False):
    params = {"kds_source_file": str(Path(xml_file_path).absolute()),
              "destination_mcpl": str(Path(mcpl_file_path).absolute()),
              "nout": amount,
              "rng": rng,
              "force": overwrite_if_exists}
    process_func(kds.resample_to_mcpl, **params)

# ----- 6. -----

def wasserstein(df_p, df_q, columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"}):
    results = {}
    for col in columns:
        if col not in df_p.columns or col not in df_q.columns:
            continue
        # Works directly on raw data (no binning needed)
        results[col] = wasserstein_distance(df_p[col], df_q[col])
    return results

def kl_div_or_js_dist(df_p, df_q, columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    bins=10,
    epsilon=1e-10,
    metric="kl",
):
  """Calculates column-wise divergence or distance between two pandas DataFrames P and Q.

  Parameters
  -----
  - df_p: Reference DataFrame (P)
  - df_q: Comparison DataFrame (Q)
  - columns: which columns to evaluate
  - bins: Number of bins for continuous numerical columns
  - epsilon: Smoothing factor to prevent zero-probability errors
  - metric: "kl" for asymmetric Kullback-Leibler Divergence (P || Q), "js" for symmetric Jensen-Shannon Distance

  Returns
  -----
  - dict of divergence/distance values for demanded columns.
  """
  results = {}

  for col in columns:
    if col not in df_p.columns or col not in df_q.columns:
      continue

    p_data = df_p[col]
    q_data = df_q[col]

    min_val = min(p_data.min(), q_data.min())
    max_val = max(p_data.max(), q_data.max())

    # If all values are identical, distance/divergence is 0
    if min_val == max_val:
      results[col] = 0.0
      continue

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    p_counts, _ = np.histogram(p_data, bins=bin_edges)
    q_counts, _ = np.histogram(q_data, bins=bin_edges)

    # Convert counts to probability distributions and apply smoothing
    p_prob = p_counts.astype(float) + epsilon
    q_prob = q_counts.astype(float) + epsilon

    p_prob /= p_prob.sum()
    q_prob /= q_prob.sum()

    # Calculate based on selected metric
    if metric == "kl":
      # Compute KL Divergence: D_KL(P || Q) (Asymmetric, unbounded)
      results[col] = entropy(p_prob, q_prob)
    elif metric == "js":
      # Compute Jensen-Shannon Distance (Symmetric, bounded between 0 and 1)
      results[col] = jensenshannon(p_prob, q_prob)
    else:
      raise ValueError("Unsupported metric. Choose 'kl' or 'js'.")

  return results

def c2st(df_p, df_q, columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"}):
  # Filter common columns
  cols = [c for c in columns if c in df_p.columns and c in df_q.columns]
  # Prepare data and labels
  p_subset = df_p[cols]
  q_subset = df_q[cols]
  X = pd.concat([p_subset, q_subset], axis=0)
  y = np.array([0] * len(p_subset) + [1] * len(q_subset))
  # Train a classifier with cross-validation
  clf = RandomForestClassifier(n_estimators=50, random_state=42)
  scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
  # Return mean accuracy (0.5 = no difference, 1.0 = completely different)
  return scores.mean()

# -------------- UNUSED: --------------

# def check_statistics(self, resolution):
#     """
#     Analyze the data and return a value between 0 and 1 that indicates how good the statistics is for the desired resolution.
#     """
#     return

# def resample_df(
#     source,
#     dest,
#     nout,
#     rng=None,
#     force=False,
#     no_df=False,
#     del_dest=False,
#     del_source=False,
# ):
#     """
#     Generates data at `dest` and by default returns it as a pandas dataframe.
#     - source: source model .xml file (which contains the original data path that is required for sampling)
#     - dest: a file path with .mcpl.gz ending
#     - nout: amount of particles generated
#     - rng: rng parameter
#     - force: whether `dest` file may be overwritten if it already exists
#     - no_df: whether you do NOT want to receive a dataframe for some reason
#     - del_dest: whether to delete file at `dest` in the end
#     - del_source: whether or not to delete the source .xml file in the end

#     Returns
#     - pandas dataframe, or `dest`, or None
#     """
#     # print("Amount of particles to generate:", str(nout))
#     kds.resample_to_mcpl(
#         kds_sourcefile=source,
#         destination_mcpl=dest,
#         nout=nout,
#         rng=rng,
#         force=force,
#     )
#     # print("Data generated at", dest)
#     if not no_df:
#         data = read_data(dest)
#     if del_dest:
#         Path(dest).unlink()
#     if del_source:
#         Path(source).unlink()
#     if not no_df:
#         return data
#     if not del_dest:
#         return dest
#     return None

# def resample_df_gen(source, dest, nout, rng=None, del_dest=False, del_source=False):
#     """
#     A generator for dataframes via the resample method using an intermediate mcpl.gz file at `dest`. ATTENTION: Overwrites `dest` in each iteration!
#     - source: source model .xml file (which contains the original data path that is required for sampling)
#     - dest: a file path with .mcpl.gz ending
#     - nout: amount of particles per dataframe
#     - rng: If you specify it, it will be used for all the generated data. If None, uses new random rng for each iteration.
#     - del_dest: whether or not to delete the most recent intermediate mcpl.gz file on the disk
#     - del_source: whether or not to delete the source .xml file in the end (after the generator has stopped)
#     """
#     try:
#         while True:
#             yield resample_df(
#                 source=source,
#                 dest=dest,
#                 nout=nout,
#                 rng=rng,
#                 force=True,
#                 no_df=False,
#                 del_dest=del_dest,
#                 del_source=False,
#             )
#     finally:
#         if del_source:
#             Path(source).unlink()

# def generate_data( 
#         inputFile,
#         output_path=None,
#         overwrite_output=False,
#         n_gen=1_000,
#         pt="n",
#         n_fit=-1,
#         kernel="gaussian",
#         bw="silv",
#         pCurrent=1.0,
#         model_path=None,
#         overwrite_model=False,
#         generator=False,
#     ):
#         """
#         Generates data similar to inputFile using kdsource, which uses KDE (Kernel Density Estimation) modeling.

#         Parameters
#         ---
#         - inputFile: file path to input data; requires ending `.mcpl` or `.mcpl.gz`
#         - output_path: file path for generated data (optional, see "Returns" section); file ending will be `.mcpl.gz`
#         - overwrite_output: whether already existing file at output_path may be overwritten
#         - n_gen: amount of particles to generate
#         - pt: particle type ("n", "p", or "e") of generated particles and of input particles used for fitting the KDE model
#         - n_fit: amount of input particles (with type `pt`) used for fitting of the source model (-1 means all)
#         - kernel: kernel shape ("gaussian", "box", or "epa" (Epanechnikov))
#         - bw: method for finding kernel bandwidth ("silv", "knn" or "mlcv")
#         - pCurrent: total source particle current
#         - model_path: where to save the model xml file (None makes it temporary)
#         - overwrite_model: whether already existing file at model_path may be overwritten
#         - generator: whether you want a generator for dataframes instead of one dataframe. `n_gen` will be the amount per yield.

#         Returns
#         ---
#         - If output_path is specified: None. (The data will be saved to the file.)
#         - If output_path is None: Pandas dataframe. Or, if `generator` is True: Generator of pandas dataframes.
#         """

#         inputFile = str(Path(inputFile).absolute())

#         return_a_df = False
#         output_is_temp = False
#         model_is_temp = False

#         if output_path is not None:
#             output_path = Path(output_path).absolute().with_suffix(".mcpl.gz")
#             if output_path.exists() and not overwrite_output:
#                 raise ValueError(
#                     "Output file already exists and overwriting is not allowed"
#                 )
#             output_path = str(output_path)
#         else:
#             return_a_df = True
#             output_path = str(Path(tempfile.gettempdir()) / "brightify_kdsource.mcpl.gz")
#             output_is_temp = True

#         if model_path is not None:
#             model_path = Path(model_path).absolute().with_suffix(".xml")
#             if model_path.exists() and not overwrite_model:
#                 raise ValueError("Model file already exists and overwriting is not allowed")
#             model_path = str(model_path)
#         else:
#             model_path = str(Path(tempfile.gettempdir()) / "brightify_kdsource.xml")
#             model_is_temp = True

#         my_input_plist = kds.PList(
#             filename=inputFile,
#             readformat="mcpl",
#             pt=pt,
#             trasl=None,
#             rot=None,
#             switch_x2z=False,
#             set_params=True,
#         )
#         # print(vars(my_input_plist))

#         my_energy = kds.geom.Energy()
#         my_time = kds.geom.Time()
#         my_position = kds.geom.Vol()
#         my_directions = kds.geom.Isotrop()
#         my_geometry = kds.Geometry([my_energy, my_time, my_position, my_directions])
#         # print(vars(my_geometry))

#         my_source = kds.KDSource(
#             plist=my_input_plist, geom=my_geometry, bw=bw, J=pCurrent, kernel=kernel
#         )
#         # print(vars(my_source))

#         if n_fit == -1 or n_fit > my_input_plist.N:
#             n_fit = my_input_plist.N
#         # print("Amount of particles for fitting:", n_fit)

#         my_source.fit(N=n_fit)

#         model_path = str(Path(my_source.save(xmlfilename=model_path)).absolute())
#         # if not model_is_temp:
#         #     print("Source model saved to", model_path)
#         # else:
#         #     print("Temporary source model saved to", model_path)

#         if not return_a_df:
#             kds.resample_to_mcpl(
#                 kds_sourcefile=model_path,
#                 destination_mcpl=output_path,
#                 nout=n_gen,
#                 rng=None,
#                 force=True,
#             )
#             if model_is_temp:
#                 Path(model_path).unlink()
#             return output_path
#         else:
#             if not generator:
#                 return resample_df(
#                     source=model_path,
#                     dest=output_path,
#                     nout=n_gen,
#                     rng=None,
#                     force=True,
#                     no_df=False,
#                     del_dest=output_is_temp,
#                     del_source=model_is_temp,
#                 )
#             else:
#                 return resample_df_gen(
#                     source=model_path,
#                     dest=output_path,
#                     nout=n_gen,
#                     rng=None,
#                     del_dest=output_is_temp,
#                     del_source=model_is_temp,
#                 )

# def organize_columns(data: pd.DataFrame, col_map=None, norm_dir=True, inplace=True):
#     """This renames columns to standard names using the COLUMN_ALIASES_LOWERCASE dictionary,
#     and drops columns whose names cannot be interpreted.
#     It makes the returned data contain all columns that mcpl data can contain.
#     It also makes all columns sparse arrays, and uses the DEFAULT_VALUES as fill values if a column was missing in the input.

#     Parameters
#     ----
#     - data: pandas DataFrame
#     - col_map (optional): A dictionary explaining any uncommon column names in the input data.
#     Most are automatically recognized, but you can specify here e.g. col_map = {"my_coord_x": "x", "my_coord_y": "y"}.
#     The dictionary values must be among the following standard names: "ekin", "pdgcode", "x", "y", "z", "ux", "uy", "uz", "weight", "time", "polx", "poly", "polz", "userflags"
#     Note that this allows you to swap columns, e.g. col_map = {"my_x":"y", "my_y":"x"}.
#     - norm_dir: Whether to normalize the direction vectors (ux, uy, uz) to manitude 1. If all of the three columns are missing or zero, an error is produced.

#     Returns
#     ----
#     DataFrame (if inplace is True, it is a reference to the manipulated original, else it is a new dataframe)
#     """
#     # check columns validity
#     if len(set(data.columns)) < len(data.columns):
#         raise ValueError("Input DataFrame must not have the same column name more than once")
#     col_map = col_map if col_map is not None else {}
#     for col,std_name in col_map.items():
#         if col not in data.columns:
#             raise ValueError("The key columns in col_map must exist in the data")
#         if std_name not in COLUMN_ALIASES_LOWERCASE.keys():
#             raise ValueError("The value columns in col_map must be among the standard names")
#     if len(set(col_map.values())) < len(col_map.values()):
#         raise ValueError("No standard column name may be used more than once in the values of col_map")

#     # build inv_col_map
#     inv_col_map = {std_name:col for col,std_name in col_map.items()}
#     for std_name, aliases in COLUMN_ALIASES_LOWERCASE.items():
#         if std_name in inv_col_map.keys():
#             continue
#         else:
#             columns_matching = []
#             for col in data.columns:
#                 if col.lower() in aliases:
#                     columns_matching.append(col)
#             if len(columns_matching) == 0:
#                 inv_col_map[std_name] = None
#             elif len(columns_matching) == 1:
#                 inv_col_map[std_name] = columns_matching[0]
#             else:
#                 raise ValueError("More than one column is matching the standard column name '"+std_name+"', namely: "+str(columns_matching))

#     # initialize dataframe to be returned
#     matched_orig_cols = {orig for orig in inv_col_map.values() if orig is not None}
#     if inplace:
#         # 1. Drop unwanted columns immediately to free memory
#         unwanted_cols = [c for c in data.columns if c not in matched_orig_cols]
#         if unwanted_cols:
#             data.drop(columns=unwanted_cols, inplace=True, errors='ignore')
#         # 2. Rename existing matched columns to their standard names right away
#         rename_map = {orig: std for std, orig in inv_col_map.items() if orig is not None and orig != std}
#         if rename_map:
#             data.rename(columns=rename_map, inplace=True)            
#         target_df = data
#     else:
#         # If not inplace, start with an empty DataFrame with the correct index
#         target_df = pd.DataFrame(index=data.index)

#     # Process column-by-column: sparsify and assign
#     for std_name, default_val in DEFAULT_VALUES.items():
#         # 1. Fetch source series
#         if inplace:
#             series = target_df.get(std_name) # returns None if column not existing
#         else:
#             orig_col = inv_col_map[std_name]
#             series = data.get(orig_col) if orig_col is not None else None
#         # 2. Apply rules based on column presence and contents
#         if series is None:
#             # Rule 1: Missing column -> Populate with DEFAULT_VALUES[std_name]
#             missing_series = pd.Series(default_val, index=target_df.index)
#             target_df[std_name] = pd.arrays.SparseArray(missing_series, fill_value=default_val)
#         elif series.dropna().empty:
#             # Rule 2: Present, but all NaNs/nulls -> Make sparse using NaN as the fill value
#             target_df[std_name] = pd.arrays.SparseArray(series, fill_value=float('nan'))
#         else:
#             # Rule 3: Present with valid data -> Optimal fill value (constant or mode)
#             if series.nunique() == 1:
#                 optimal_fill = series.iloc[0]
#             else:
#                 mode_vals = series.mode(dropna=False)
#                 optimal_fill = mode_vals.iloc[0] if not mode_vals.empty else default_val
#             target_df[std_name] = pd.arrays.SparseArray(series, fill_value=optimal_fill)

#     # Normalize direction vectors if requested
#     if norm_dir:            
#         # Convert to dense numpy arrays for calculation
#         ux = target_df['ux'].to_numpy(dtype=float)
#         uy = target_df['uy'].to_numpy(dtype=float)
#         uz = target_df['uz'].to_numpy(dtype=float)
#         # Calculate magnitude (norm)
#         norm = np.sqrt(ux**2 + uy**2 + uz**2)
#         # Check if all entries/columns are zero or missing (norm is 0 everywhere)
#         if np.all(norm == 0):
#             raise ValueError("Direction vectors (ux, uy, uz) are all missing or zero.")
#         # Avoid division by zero for any individual zero-vectors if necessary,
#         # or normalize where norm > 0
#         valid_mask = norm > 0
        
#         # Initialize normalized arrays
#         norm_ux = np.zeros_like(ux)
#         norm_uy = np.zeros_like(uy)
#         norm_uz = np.zeros_like(uz)
        
#         norm_ux[valid_mask] = ux[valid_mask] / norm[valid_mask]
#         norm_uy[valid_mask] = uy[valid_mask] / norm[valid_mask]
#         norm_uz[valid_mask] = uz[valid_mask] / norm[valid_mask]
        
#         # Re-assign back to target_df as sparse arrays
#         target_df['ux'] = pd.arrays.SparseArray(norm_ux, fill_value=0.0)
#         target_df['uy'] = pd.arrays.SparseArray(norm_uy, fill_value=0.0)
#         target_df['uz'] = pd.arrays.SparseArray(norm_uz, fill_value=0.0)


#     return target_df

# def remove_na(data: pd.DataFrame, inplace=True, ignore_index=True):
#     """
#     Returns data with rows (not columns) removed that have one (or more) "na" (not assigned/empty/NaN) values, using pandas dropna() method.
#     - inplace: Set to True to manipulate the input dataframe object instead of obtaining a copy.
#     - ignore_index: Set True if you want the returned data to have a fresh index starting at 0, instead of keeping the index numbers as before (with the removed rows missing). 
#     """
#     if not inplace:
#         data = data.dropna(inplace=False, ignore_index=ignore_index)
#     else:
#         data.dropna(inplace=True, ignore_index=ignore_index)
#     if len(data) == 0:
#         print("Warning: The new dataframe is empty.")
#     return data

# def organize_data(data: pd.DataFrame, col_map: dict = None, rtol=1e-5, atol=1e-8, copy=False):
#     """
#     Maps column names to standard names for mcpl data, checks for constant/zero/missing columns, and returns the organized data.
    
   
#     - rtol and atol: relative and absolute tolerance parameters passed to np.isclose() in order to find almost-constant columns (see below).
#     - copy: Set True if you want the returned dictionary to contain copies of the dataframe columns instead of views/references of them.

#     Returns
#     ----
#     inv_col_map, columns_len, columns_data

#     - inv_col_map: Dictionary with the standard column names as keys and the respective input columns' names as values.
#     - columns_len: Dictionary (with standard name keys) indicating for each column whether it is missing (0), constant (positive integer length of the column), or variable (negative integer column length).
#     The column length will of course be the length of the input data.
#     - columns_data: Dictionary (with standard name keys) of the column data, can be None (missing), or a scalar (constant) number, or a variable pandas series.
#     The series are views on the dataframe columns, unless copy = True.

#     Note: "Almost"-constancy of columns is checked via np.isclose(column, np.mean(column), rtol=rtol, atol=atol).
#     If True, then the column is considered constant with the mean as the constant value.

#     Note: The method does not handle NaN values in the dataframe. Better remove them beforehand using remove_na() method.
#     """

#     # check inputs
    
#     # build other outputs
#     columns_len = {} 
#     columns_data = {}
#     len_data = len(data)
#     for std_name, col in inv_col_map.items():
#         if col is None:
#             columns_len[std_name] = 0
#             columns_data[std_name] = None
#             continue
#         else:
#             series = data[col]
#             if len(series) == 0:
#                 columns_len[std_name] = 0 # zero -> missing column
#                 columns_data[std_name] = None # None value
#             elif series.nunique() == 1:
#                 first = series.iloc[0]
#                 columns_len[std_name] = len_data # positive -> constant column
#                 columns_data[std_name] = first # scalar value
#             else:
#                 mean = series.mean()
#                 if np.all(np.isclose(series, mean, rtol=rtol, atol=atol)):
#                     columns_len[std_name] = len_data # positive -> constant column
#                     columns_data[std_name] = mean # scalar value
#                 else:
#                     columns_len[std_name] = -len_data # negative -> variable column
#                     if copy:
#                         columns_data[std_name] = series.copy() # series value
#                     else:
#                         columns_data[std_name] = series # series value
    
#     return inv_col_map, columns_len, columns_data
