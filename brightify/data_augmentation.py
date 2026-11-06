# This module allows the brightify user to:
# 1. Check the statistics of his original input data, and if it's sufficient for the desired resolution of the brightify calculations.
# 2. Find out the geometry of his data automatically, or specify it manually.
# 3. Generate (resample) more "synthetic" data that is similar to the original input data, using kdsource, in order to increase the statistics.

# ----- IMPORTS -----
import kdsource.api as kds
import mcpl
import pandas as pd
from pathlib import Path
import tempfile
import numpy as np
import sys
import os
import multiprocessing

# ----- CONSTANTS -----
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

# ----- DATA I/O -----

class OrganizedData:
    def __init__(self, data: pd.DataFrame, col_map: dict = None, rtol=1e-5, atol=1e-8):
        _, self.columns_len, self.columns_data = self.organize_data(data=data, col_map=col_map, rtol=rtol, atol=atol, copy=False)

    @classmethod
    def from_mcpl_file(cls, 
                  file_path, n:int=-1, columns:set={"ekin","pdgcode","x","y","z","ux","uy","uz","weight","time","polx","poly","polz","userflags"},
                  rtol=1e-5, atol=1e-8):
        data = cls.read_data(file_path=file_path, n=n, columns=columns)
        return cls(data=data, col_map=None, rtol=rtol, atol=atol)

    @staticmethod
    def read_data(file_path, n: int =-1, columns: set = {"ekin","pdgcode","x","y","z","ux","uy","uz","weight","time","polx","poly","polz","userflags"}):
        """
        Read up to n particles from the mcpl file into a pandas dataframe. n==-1 means read all.
        Only the specified columns are read.
        """
        blocklength = 100000
        gen_file = mcpl.MCPLFile(str(Path(file_path).absolute()), blocklength=blocklength)
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
    
    @staticmethod
    def organize_data(data: pd.DataFrame, col_map: dict = None, rtol=1e-5, atol=1e-8, copy=False):
        """
        Maps column names to standard names for mcpl data, checks for constant/zero/missing columns, and returns the organized data.
        
        Parameters
        ----
        - data: pandas DataFrame
        - col_map (optional): A dictionary explaining any uncommon column names in the input data.
        Most are automatically recognized, but you can specify here e.g. col_map = {"my_coord_x": "x", "my_coord_y": "y"}.
        The dictionary values must be among the following standard names: "ekin", "pdgcode", "x", "y", "z", "ux", "uy", "uz", "weight", "time", "polx", "poly", "polz", "userflags"
        Note that this allows you to swap columns, e.g. col_map = {"my_x":"y", "my_y":"x"}.
        - rtol and atol: relative and absolute tolerance parameters passed to np.isclose() in order to find almost-constant columns (see below).
        - copy: Set True if you want the returned dictionary to contain copies of the dataframe columns instead of views/references of them.

        Returns
        ----
        inv_col_map, columns_len, columns_data

        - inv_col_map: Dictionary with the standard column names as keys and the respective input columns' names as values.
        - columns_len: Dictionary (with standard name keys) indicating for each column whether it is missing (0), constant (positive integer length of the column), or variable (negative integer column length).
        The column length will of course be the length of the input data.
        - columns_data: Dictionary (with standard name keys) of the column data, can be None (missing), or a scalar (constant) number, or a variable pandas series (including any NaN values).
        The series are views on the dataframe columns, unless copy = True.

        Note: "Almost"-constancy of columns is checked via np.isclose(column, np.mean(column), rtol=rtol, atol=atol).
        If True, then the column is considered constant with the mean as the constant value.
        NaN values are ignored in the calculations, unless the entire column is NaN, then the column will be considered a missing column. 
        Variable columns are such that have least two different non-NaN values that are not close enough to pass the np.isclose() test.
        """

        # check inputs
        if len(set(data.columns)) < len(data.columns):
            raise ValueError("Input DataFrame must not have the same column name more than once")
        col_map = col_map if col_map is not None else {}
        for col,std_name in col_map.items():
            if col not in data.columns:
                raise ValueError("The key columns in col_map must exist in the data")
            if std_name not in COLUMN_ALIASES_LOWERCASE.keys():
                raise ValueError("The value columns in col_map must be among the standard names")
        if len(set(col_map.values())) < len(col_map.values()):
            raise ValueError("No standard column name may be used more than once in the values of col_map")

        # build inv_col_map
        inv_col_map = {std_name:col for col,std_name in col_map.items()}
        for std_name, aliases in COLUMN_ALIASES_LOWERCASE.items():
            if std_name in inv_col_map.keys():
                continue
            else:
                columns_matching = []
                for col in data.columns:
                    if col.lower() in aliases:
                        columns_matching.append(col)
                if len(columns_matching) == 0:
                    inv_col_map[std_name] = None
                elif len(columns_matching) == 1:
                    inv_col_map[std_name] = columns_matching[0]
                else:
                    raise ValueError("More than one column is matching the standard column name '"+std_name+"', namely: "+str(columns_matching))

        # build other outputs
        columns_len = {} 
        columns_data = {}
        len_data = len(data)
        for std_name, col in inv_col_map.items():
            if col is None:
                columns_len[std_name] = 0
                columns_data[std_name] = None
                continue
            else:
                series = data[col]  # might contain NaN values
                series_without_NaN = series.dropna()
                if len(series_without_NaN) == 0:
                    columns_len[std_name] = 0 # zero -> missing column
                    columns_data[std_name] = None # None value
                elif series_without_NaN.nunique() == 1:
                    first = series_without_NaN.iloc[0]
                    columns_len[std_name] = len_data # positive -> constant column
                    columns_data[std_name] = first # scalar value
                else:
                    mean = series_without_NaN.mean()
                    if np.all(np.isclose(series_without_NaN, mean, rtol=rtol, atol=atol)):
                        columns_len[std_name] = len_data # positive -> constant column
                        columns_data[std_name] = mean # scalar value
                    else:
                        columns_len[std_name] = -len_data # negative -> variable column
                        if copy:
                            columns_data[std_name] = series.copy() # series value
                        else:
                            columns_data[std_name] = series # series value
        
        return inv_col_map, columns_len, columns_data

    def write_to_mcpl(self, file_path, double_prec=False, force=False):
        """
        Write data to a mcpl file using kdsource.
        Missing (len = 0) columns will be set to default values: weight=1, pdgcode=2112 (neutron), all other=0,
        except polx, poly, polz and userflags, which may be None.
        Where needed, scalar values are converted to rows before being passed to the kds.write_mcpl() method.
        """
        assert len(self.columns_len) == len(self.columns_data) == len(COLUMN_ALIASES_LOWERCASE)
        write_data = self.columns_data.copy() # makes a shallow copy and assigns it to new local dictionary. it contains links to the same values (identical object references). reassigning of keys does not change those original values, nor the original dict.
        file_path = str(Path(file_path).absolute())
        abs_lens_set = set([abs(l) for l in self.columns_len.values()])
        longest = max(abs_lens_set)
        assert not longest == 0 # Longest column is zero length
        assert not len(abs_lens_set) > 2 and not (len(abs_lens_set) == 2 and not 0 in abs_lens_set) # Some columns are different length
        for col, col_len in self.columns_len.items():
            # allowed scalar but not None: weight pdgcode
            if col == "weight" and col_len == 0:
                write_data[col] = 1 # default
            elif col == "pdgcode" and col_len == 0:
                write_data[col] = 2112 # default
            # allowed None but not scalar: polx poly polz userflags
            elif col in ("polx", "poly", "polz", "userflags"):
                if col_len > 0: # constant column
                    val = write_data[col]
                    write_data[col] = np.full(longest, val)
                # other cases: col_len = 0 (None) is allowed as write parameter, col_len < 0 (variable series) is covered below.
            # others:
            else:
                if col_len == 0: # missing column
                    write_data[col] = np.zeros(longest) # default
                elif col_len > 0: # constant column
                    val = write_data[col]
                    write_data[col] = np.full(longest, val)
                elif col_len < 0: # variable column
                    pass # write_data[col] already ok
                else:
                    raise Exception("Impossible column length encountered.")
        # call the write_mcpl method using the process_func wrapper to avoid kernel crash and see reasons for failure (e.g. bad columns, negative energy etc.)
        process_func(kds.write_mcpl, filename=file_path, **write_data, double_prec=double_prec, force=force)

# ----- DATA MUTATION -----


def normalize_vector_columns(data: pd.DataFrame):

    # all_columns = constant_columns | variable_columns | missing_columns # this is a shallow copy combining all keys (the values are references to the same objects as in the two individual dictionaries)

    # # 3. Check normalization of direction vectors
    # ux, uy, uz = (all_columns[inv_col_map[c]] for c in ("ux", "uy", "uz"))
    # if not (ux is None and uy is None and uz is None): # Only skip if all directions are missing (i.e. None). Otherwise, assume that the missing directions are zero.
    #     if ux is None: ux = 0
    #     if uy is None: uy = 0
    #     if uz is None: uz = 0
    #     mag = np.sqrt(ux**2 + uy**2 + uz**2) # this might be a pandas series or a single number
    #     check = mag if isinstance(mag, (int, float, np.number)) else mag.dropna()
    #     if np.allclose(check, 1.0):
    #         print("Direction vectors are already normalized.")
    #     else:
    #         print("Direction vectors are not already normalized.")
    #     if do_normalize:
    #         if isinstance(mag, (int, float, np.number)) and mag == 0:
    #             raise ValueError("Cannot normalize zero-magnitude vector")
    #         else:
    #             mag = mag.replace(0, np.nan)
    #         ux = ux / mag # this may cast an ux that is previously just a number to a series/array, if mag is a series.
    #         uy = uy / mag # on the other hand, if it is a series already, and mag is just a number, it results a series too.
    #         uz = uz / mag # finally, two series are also becoming a series, but two numbers become just a number.
            
    #         all_columns[...] = ...

    #         print("Direction vectors have been normalized.")
    
    # else: # all directions are None
    #     if do_normalize:
    #         print("Direction vectors cannot be normalized as all components are missing.")

    return all_columns


def find_geometry(inputs):
    """
    This attempts to find a kdsource geometry that suits the inputs
    """
    # We must look at:
    # 1. missing or constant columns
    return geometry


def augment_data(inputs):
    """
    Expands data using kdsource
    """
    # 1. find geometry
    # 2. construct model
    # 3. fit model
    # 4. resample to disk
    # 5. read from disk and append to input data?
    return augmented_data





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
#     inputFile,
#     output_path=None,
#     overwrite_output=False,
#     n_gen=1_000,
#     pt="n",
#     n_fit=-1,
#     kernel="gaussian",
#     bw="silv",
#     pCurrent=1.0,
#     model_path=None,
#     overwrite_model=False,
#     generator=False,
# ):
#     """
#     Generates data similar to inputFile using kdsource, which uses KDE (Kernel Density Estimation) modeling.

#     Parameters
#     ---
#     - inputFile: file path to input data; requires ending `.mcpl` or `.mcpl.gz`
#     - output_path: file path for generated data (optional, see "Returns" section); file ending will be `.mcpl.gz`
#     - overwrite_output: whether already existing file at output_path may be overwritten
#     - n_gen: amount of particles to generate
#     - pt: particle type ("n", "p", or "e") of generated particles and of input particles used for fitting the KDE model
#     - n_fit: amount of input particles (with type `pt`) used for fitting of the source model (-1 means all)
#     - kernel: kernel shape ("gaussian", "box", or "epa" (Epanechnikov))
#     - bw: method for finding kernel bandwidth ("silv", "knn" or "mlcv")
#     - pCurrent: total source particle current
#     - model_path: where to save the model xml file (None makes it temporary)
#     - overwrite_model: whether already existing file at model_path may be overwritten
#     - generator: whether you want a generator for dataframes instead of one dataframe. `n_gen` will be the amount per yield.

#     Returns
#     ---
#     - If output_path is specified: None. (The data will be saved to the file.)
#     - If output_path is None: Pandas dataframe. Or, if `generator` is True: Generator of pandas dataframes.
#     """

#     inputFile = str(Path(inputFile).absolute())

#     return_a_df = False
#     output_is_temp = False
#     model_is_temp = False

#     if output_path is not None:
#         output_path = Path(output_path).absolute().with_suffix(".mcpl.gz")
#         if output_path.exists() and not overwrite_output:
#             raise ValueError(
#                 "Output file already exists and overwriting is not allowed"
#             )
#         output_path = str(output_path)
#     else:
#         return_a_df = True
#         output_path = str(Path(tempfile.gettempdir()) / "brightify_kdsource.mcpl.gz")
#         output_is_temp = True

#     if model_path is not None:
#         model_path = Path(model_path).absolute().with_suffix(".xml")
#         if model_path.exists() and not overwrite_model:
#             raise ValueError("Model file already exists and overwriting is not allowed")
#         model_path = str(model_path)
#     else:
#         model_path = str(Path(tempfile.gettempdir()) / "brightify_kdsource.xml")
#         model_is_temp = True

#     my_input_plist = kds.PList(
#         filename=inputFile,
#         readformat="mcpl",
#         pt=pt,
#         trasl=None,
#         rot=None,
#         switch_x2z=False,
#         set_params=True,
#     )
#     # print(vars(my_input_plist))

#     my_energy = kds.geom.Energy()
#     my_time = kds.geom.Time()
#     my_position = kds.geom.Vol()
#     my_directions = kds.geom.Isotrop()
#     my_geometry = kds.Geometry([my_energy, my_time, my_position, my_directions])
#     # print(vars(my_geometry))

#     my_source = kds.KDSource(
#         plist=my_input_plist, geom=my_geometry, bw=bw, J=pCurrent, kernel=kernel
#     )
#     # print(vars(my_source))

#     if n_fit == -1 or n_fit > my_input_plist.N:
#         n_fit = my_input_plist.N
#     # print("Amount of particles for fitting:", n_fit)

#     my_source.fit(N=n_fit)

#     model_path = str(Path(my_source.save(xmlfilename=model_path)).absolute())
#     # if not model_is_temp:
#     #     print("Source model saved to", model_path)
#     # else:
#     #     print("Temporary source model saved to", model_path)

#     if not return_a_df:
#         kds.resample_to_mcpl(
#             kds_sourcefile=model_path,
#             destination_mcpl=output_path,
#             nout=n_gen,
#             rng=None,
#             force=True,
#         )
#         if model_is_temp:
#             Path(model_path).unlink()
#         return output_path
#     else:
#         if not generator:
#             return resample_df(
#                 source=model_path,
#                 dest=output_path,
#                 nout=n_gen,
#                 rng=None,
#                 force=True,
#                 no_df=False,
#                 del_dest=output_is_temp,
#                 del_source=model_is_temp,
#             )
#         else:
#             return resample_df_gen(
#                 source=model_path,
#                 dest=output_path,
#                 nout=n_gen,
#                 rng=None,
#                 del_dest=output_is_temp,
#                 del_source=model_is_temp,
#             )
