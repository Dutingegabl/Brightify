# This module should allow the brightify user to:
# I. Check the statistics of his original input data, and if it's sufficient for the desired resolution of the brightify calculations.
# II. Find out the geometry of his data automatically, or specify it manually.
# III. Generate (resample) more "synthetic" data that is similar to the original input data, using kdsource, in order to increase the statistics.
# IV. Check how close the generated data is to other data (in particular to the original input data, i.e. if it converges)

# required methods
# 1. read mcpl into dataframe, only the needed columns
# 2. write dataframe to mcpl
# 3. detect geometry/metrics of data
# 4. perform kds fit based on a geometry and other parameters
# 5. resample to mcpl based on original mcpl and fit output
# 6. comparing of two dataframes, using of different metrics: see data_comparison.py

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
from brightify.data_comparison import *

# ----- CONSTANTS -----

DEFAULT_COLUMNS_ORDER = ["ekin",
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

DEFAULT_COLUMNS_VALUES = {
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

DEFAULT_COLUMNS_ALIASES_LOWERCASE = {
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

def remove_nans_of_df(
    data: pd.DataFrame, 
    exempt_columns: list = None, 
    ignore_index: bool = False,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Removes rows containing NaN values from a DataFrame.
    Checks all columns by default, except those listed in 'exempt_columns'.
    """
    if exempt_columns is None:
        exempt_columns = []

    df = data if inplace else data.copy()
            
    # Use lowercase set lookup for performance
    exempt_lower = {str(e).lower() for e in exempt_columns}
    cols_to_check = [
        col for col in df.columns 
        if str(col).lower() not in exempt_lower
    ]
    
    if inplace:
        df.dropna(subset=cols_to_check, inplace=True)
        if ignore_index:
            df.reset_index(drop=True, inplace=True)
        return df
    else:
        return df.dropna(subset=cols_to_check, ignore_index=ignore_index)

def homogenize_df(
    data: pd.DataFrame, 
    explanation: dict = None, 
    exempt_columns: list = None,
    fill_defaults: bool = True,
    inplace=False,
) -> pd.DataFrame:
    """
    Cleans, renames, and filters a DataFrame regarding MCPL file standard.

    1. Uses an optional `explanation` dictionary for explicit renamings (e.g., {'my_col': 'ekin'}).
    2. Maps remaining columns automatically using global `DEFAULT_COLUMNS_ALIASES_LOWERCASE` (e.g. 'E_MeV' will be recognized to be 'ekin').
    3. Discards unnecessary columns unless they are listed in `exempt_columns`.
    4. Optionally fills missing standard MCPL columns with `DEFAULT_COLUMNS_VALUES`. 

    The returned dataframe is guaranteed to only contain default column names, plus any `exempt_columns`.
    
    Edge Case: If you pass an explanation that maps a column to a custom name not found in DEFAULT_COLUMNS_VALUES nor in `exempt_columns`, then that column will be discarded. 
    """
    if explanation is None:
        explanation = {}
    if exempt_columns is None:
        exempt_columns = []

    df = data if inplace else data.copy()
    
    rename_map = {}
    
    alias_to_standard = {}
    for std_col, aliases in DEFAULT_COLUMNS_ALIASES_LOWERCASE.items():
        for alias in aliases:
            alias_to_standard[alias.lower()] = std_col
            
    exempt_lower = {str(e).lower() for e in exempt_columns}
    
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        
        if col_str in explanation:
            rename_map[col] = explanation[col_str]
        elif col_lower in alias_to_standard:
            rename_map[col] = alias_to_standard[col_lower]
            
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    
    # Filter columns efficiently 
    valid_standard_cols = set(DEFAULT_COLUMNS_VALUES.keys())
    cols_to_keep = [
        col for col in df.columns 
        if col in valid_standard_cols or str(col).lower() in exempt_lower
    ]
    
    # Only re-index/filter if columns actually need dropping to avoid unnecessary copying
    if len(cols_to_keep) < len(df.columns):
        df = df[cols_to_keep]
    
    # Fill missing standard columns using scalar broadcasting (saves massive RAM)
    if fill_defaults:
        for std_col, default in DEFAULT_COLUMNS_VALUES.items():
            if std_col not in df.columns:
                df[std_col] = default  # Scalar assignment avoids np.full() memory allocation
                    
    return df

# ----- 1. -----
def read_df_from_mcpl(mcpl_file_path, n: int =-1, columns: set = {"ekin","pdgcode","x","y","z","ux","uy","uz","weight","time","polx","poly","polz","userflags"}):
    """
    Read up to n particles from the mcpl file into a pandas dataframe. n==-1 means read all.
    Only the specified columns are read.
    """
    columns = sorted(columns, key=lambda col: DEFAULT_COLUMNS_ORDER.index(col))
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
    Respected column names are: "ekin", "pdgcode", "x", "y", "z", "ux", "uy", "uz", "weight", "time", "polx", "poly", "polz", "userflags".
    Missing columns will be set to default values found in DEFAULT_VALUES,
    except "polx", "poly", "polz" and "userflags", which may be set to None in the mcpl file format.
    Note that writing will fail if e.g. the direction vectors are not normalized or entirely zero, or energy is negative etc.
    """
    write_data = dict()
    N = len(data)
    for col, default in DEFAULT_COLUMNS_VALUES.items():
        if col in data.columns:
            write_data[col] = data[col]
        elif default is not None:
            write_data[col] = np.full(N, default)
        else:
            write_data[col] = None
    # call the write_mcpl method using the process_func wrapper to avoid kernel crash and see reasons for failure (e.g. bad columns, negative energy etc.)
    process_func(kds.write_mcpl, filename=str(Path(mcpl_file_path).absolute()), **write_data, double_prec=double_prec, force=force)

# ----- 3. -----
def automatic_metrics(df: pd.DataFrame, trasl=None, rot=None, use_wavelength_for_neutrons=True, prefer_cartesian_output: bool = True):
    """
    Analyzes an MCPL pandas DataFrame to automatically find suitable metrics 
    (as input to kdsource Geometry() constructor), and an importance vector,
    omitting constant (zero-variance) variables entirely.

    Includes advanced heuristics for Wavelength, SurfR2, SurfCircle, Polar, and PolarMu.

    1. Transform input coordinates coordinate system using trasl and rot (without mutating input dataframe).
    2. Compute the weighted variance of variables in the transformed frame.
    3. Automatically select and instantiate the best matching metrics classes from the kdsource geom module,
       omitting unnecessary/constant/zero-variance variables.
    
    Returns
    -------
    metrics : list
        List of instantiated metrics corresponding to active dimensions.
    importance: list
        List of importance values (float), same order as `Geometry(metrics).varnames`.

    Assumes "weight", "ekin", "x", "y", "z", "ux", "uy", "uz", and "time" 
    are always present in df with valid non-NaN values.
    """
    vars_list = ["ekin", "x", "y", "z", "ux", "uy", "uz", "time"]
    weights = df["weight"].values
    
    # ---------------------------------------------------------
    # 1. Apply User Transformation (trasl & rot) to Data First
    # ---------------------------------------------------------
    pos = df[["x", "y", "z"]].values.astype(float)
    dir_vecs = df[["ux", "uy", "uz"]].values.astype(float)
    
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

    if trasl is not None:
        trasl_arr = np.array(trasl).reshape(1, 3)
        pos -= trasl_arr
    
    if rot is not None:
        pos = rot.apply(pos, inverse=True)
        dir_vecs = rot.apply(dir_vecs, inverse=True)

    transformed_data = np.column_stack([
        df["ekin"].values,
        pos,
        dir_vecs,
        df["time"].values
    ])

    # ---------------------------------------------------------
    # 2. Compute Variable Weighted Variances in Aligned Frame
    # ---------------------------------------------------------
    is_variable = {}
    for i, var in enumerate(vars_list):
        col_data = transformed_data[:, i]
        w_mean = np.average(col_data, weights=weights)
        w_var = np.average((col_data - w_mean) ** 2, weights=weights)
        if w_var < 1e-14 or np.isnan(w_var):
            is_variable[var] = False
        else:
            is_variable[var] = True
            
    x_val = pos[:, 0]
    y_val = pos[:, 1]
    z_val = pos[:, 2]

    # ---------------------------------------------------------
    # 3. Detect and Select Optimal Metrics & Build Importance List
    # ---------------------------------------------------------
    metrics = []

    # --- A. Energy Dimension (ekin) ---
    if is_variable["ekin"]:
        ekin_vals = df["ekin"].values
        
        is_thermal_neutron = use_wavelength_for_neutrons and np.median(ekin_vals) < 1e-6
        
        if is_thermal_neutron:
            metrics.append(kds.geom.Wavelength()) # l
        elif np.all(ekin_vals > 0) and (np.max(ekin_vals) / np.min(ekin_vals) > 10):
            E0 = float(np.max(ekin_vals))
            metrics.append(kds.geom.Lethargy(E0=E0))
        else:
            metrics.append(kds.geom.Energy())

    # --- B. Position & Direction Dimensions ---
    pos_active = is_variable["x"] or is_variable["y"] or is_variable["z"]
    xy_active = is_variable["x"] or is_variable["y"]
    z_is_constant = not is_variable["z"]
    dir_active = is_variable["ux"] or is_variable["uy"] or is_variable["uz"]

    guide_added = False
    if xy_active and not z_is_constant and dir_active:
        xwidth = float(np.max(np.abs(x_val)) * 2)
        yheight = float(np.max(np.abs(y_val)) * 2)
        zmax = float(np.max(z_val))
        if xwidth > 0 and yheight > 0:
            metrics.append(kds.geom.Guide(xwidth=xwidth, yheight=yheight, zmax=zmax))
            guide_added = True
    
    if not guide_added:
        # Position Metric Selection
        if pos_active:
            if z_is_constant:
                z_fixed = float(np.mean(z_val))
                xmin, xmax = float(np.min(x_val)), float(np.max(x_val))
                ymin, ymax = float(np.min(y_val)), float(np.max(y_val))
                
                r_vals = np.sqrt(x_val**2 + y_val**2)
                is_polar = np.std(r_vals) < (np.mean(r_vals) * 0.5) and np.min(r_vals) == 0
                
                if is_polar:
                    rho_max = float(np.max(r_vals))
                    
                    # Choice 1: Do you want the metric variables to be polar (rho, psi)?
                    # Use SurfR / SurfR2.
                    # Choice 2: Do you want the metric variables to stay Cartesian (x, y) 
                    # despite having a circular boundary? Use SurfCircle!
                                        
                    if prefer_cartesian_output:
                        metrics.append(kds.geom.SurfCircle(rho_min=0, rho_max=rho_max, z=z_fixed))
                    else:
                        expected_mean_r2 = 0.5 * (rho_max ** 2)
                        actual_mean_r2 = np.average(r_vals**2, weights=weights)
                        
                        if abs(actual_mean_r2 - expected_mean_r2) / expected_mean_r2 < 0.2:
                            metrics.append(kds.geom.SurfR2(rho_min=0, rho_max=rho_max, z=z_fixed))
                        else:
                            metrics.append(kds.geom.SurfR(rho_min=0, rho_max=rho_max, z=z_fixed))
                else:
                    metrics.append(kds.geom.SurfXY(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, z=z_fixed))
            else:
                xmin, xmax = float(np.min(x_val)), float(np.max(x_val))
                ymin, ymax = float(np.min(y_val)), float(np.max(y_val))
                zmin, zmax = float(np.min(z_val)), float(np.max(z_val))
                metrics.append(kds.geom.Vol(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax))
            
        # Direction Metric Selection
        if dir_active:
            uz_vals = dir_vecs[:, 2]
            uz_clipped = np.clip(uz_vals, -1.0, 1.0)
            theta_vals = np.arccos(uz_clipped) * (180.0 / np.pi)
            theta_w_mean = np.average(theta_vals, weights=weights)
            theta_w_std = np.sqrt(np.average((theta_vals - theta_w_mean) ** 2, weights=weights))
            if theta_w_std < 15.0:
                metrics.append(kds.geom.Polar())
            else:
                metrics.append(kds.geom.PolarMu())                

    # --- C. Time Dimension (time) ---
    if is_variable["time"]:
        time_vals = df["time"].values
        if np.all(time_vals > 0) and (np.max(time_vals) / np.min(time_vals) > 10):
            metrics.append(kds.geom.Decade())
        else:
            metrics.append(kds.geom.Time())
    
    return metrics

# ----- 4. -----
def fit(mcpl_file_path, xml_file_path,
        bw_file_path = None,
        trasl = None,
        rot = None,
        switch_x2z = False,
        geometry: kds.geom.Geometry | None = None,
        bw = "auto",
        J = 1,
        kernel = "gaussian",
        N = -1,
        skip = 0,
        scaling = None,
        importance = None,
        bw_kwargs = {},
        adjust_N = True):
    """
    This makes a fit and saves a model xml file.
    Neutrons only.

    Parameters
    ---------
    - mcpl_file_path: input mcpl(.gz) file path
    - xml_file_path: output model file path
    - bw_file_path: file path for save() method (only if variable bandwidth method; default = xml path with '.bws' ending)
    - trasl:
    - rot:
    - switch_x2z:
    - geometry: A kdsource.geom.Geometry object (with the same trasl and rot as entered here), or None.
    If None, it will be attempted to detect a suiting Geometry using up to 100k particles. 
    Depending on the translation and rotation, the geometry detected will be better or worse.
    - bw: bandwidth method "auto", "silv", "knn" or "mlcv"
    - J:
    - kernel: "gaussian", "epa" or "box"
    - N: number of particles used at most for the fitting
    - skip:
    - scaling: length 8 vector (for variables ekin, x, y, z, ux, uy, uz, time)
    - importance: length 8 vector (for variables ekin, x, y, z, ux, uy, uz, time). 
    Note that importance is automatically decided when geometry is None, but that can be overridden by this parameter.
    - bw_kwargs: kwargs for optimize_bw() and bw_auto(), bw_silv(), bw_knn(), or bw_mlcv()
    - adjust_N: 

    Returns
    -------
    string path to xml file
    """
    standard_variable_names = ["ekin", "x", "y", "z", "ux", "uy", "uz", "time"]

    # check file path parameters:
    mcpl_file_path = str(Path(mcpl_file_path).absolute())
    xml_file_path = str(Path(xml_file_path).absolute())
    if bw_file_path is not None:
        bw_file_path = str(Path(bw_file_path).absolute())
    else:
        bw_file_path = str(Path(xml_file_path).with_suffix(".bws").absolute())

    # filter bw_kwargs so nothing crashes
    allowed_bw_kwargs_all = {"weightfun", "maskfun"} # optimize_bw()
    allowed_bw_kwargs_auto = {"grid", "Nmlcv", "Nbatch", "Nknn", "show", "n_splits"} # bw_auto()
    allowed_bw_kwargs_silv = set()
    allowed_bw_kwargs_knn = {"K_eff", "k", "batch_size"} # bw_knn()
    allowed_bw_kwargs_mlcv = {"n_splits", "seed", "grid", "show"} # bw_mlcv()
    bw_kwargs = {k:v for k,v in bw_kwargs.items()
                 if k in allowed_bw_kwargs_all
                 or (bw == "auto" and k in allowed_bw_kwargs_auto)
                 or (bw == "silv" and k in allowed_bw_kwargs_silv)
                 or (bw == "knn" and k in allowed_bw_kwargs_knn)
                 or (bw == "mlcv" and k in allowed_bw_kwargs_mlcv)}
    print("bw_kwargs that were accepted are:", str(bw_kwargs))

    # make plist, geometry, source and then fit and save model xml:
    plist = kds.PList(str(Path(mcpl_file_path).absolute()), readformat="mcpl", pt="n", trasl=trasl, rot=rot, switch_x2z=switch_x2z, set_params=True)
    if geometry is None:
        parts, ws = plist.get(N=100_000)
        test_df = pd.DataFrame(parts, columns = standard_variable_names)
        test_df["weight"] = ws
        auto_metrics = automatic_metrics(df=test_df, trasl=None, rot=None) # no trasl and rot entered here because the data obtained from plist.get() has them already applied
        print("automatically found metrics:", str(auto_metrics))
        geometry = kds.geom.Geometry(metrics=auto_metrics, trasl=trasl, rot=rot)
        print("automatically generated geometry's properties:")
        print("varnames:", geometry.varnames)
        print("units:", geometry.units)
        print("volunits:", geometry.volunits)
        print("dim:", geometry.dim)
    source = kds.KDSource(plist=plist, geom=geometry, bw=bw, J=J, kernel=kernel)
    source.fit(N=N, skip=skip, scaling=scaling, importance=importance, bw_method=None, **bw_kwargs)
    source.save(xmlfilename=xml_file_path, bwfile=bw_file_path, adjust_N=adjust_N)
    if Path(bw_file_path).exists():
        return xml_file_path, bw_file_path
    else:
        return xml_file_path, None

# ----- 5. -----
def resample(xml_file_path, mcpl_file_path, amount,
             rng=None,
             overwrite_if_exists=False):
    params = {"kds_sourcefile": str(Path(xml_file_path).absolute()),
              "destination_mcpl": str(Path(mcpl_file_path).absolute()),
              "nout": amount,
              "rng": rng,
              "force": overwrite_if_exists}
    process_func(kds.resample_to_mcpl, **params)

# ----- 6. -----
# see data_comparison.py


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
