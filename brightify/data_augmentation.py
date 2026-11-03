import kdsource.api as kds
import mcpl
import pandas as pd
from pathlib import Path
import tempfile
import numpy as np


DEFAULT_ALIASES_LOWERCASE = {
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

def organize_data(data: pd.DataFrame, col_map: dict = None, rtol=1e-5, atol=1e-8):
    """
    Maps columns, checks for constant/zero values, and returns the organized data.
    
    Parameters
    ----
    - data: pandas DataFrame (e.g. created from a mcpl file)
    - col_map (optional): A dictionary explaining any uncommon column names in the input data.
    Most are automatically recognized, but you can specify here e.g. col_map = {"my_coord_x": "x", "my_coord_y": "y"}.
    The values must be among the following standard names: "ekin", "pdgcode", "x", "y", "z", "ux", "uy", "uz", "weight", "time", "polx", "poly", "polz", "userflags"
    Note that this allows you to swap columns, e.g. col_map = {"my_x":"y", "my_y":"x"}.
    - rtol and atol: relative and absolute tolerance parameters passed to np.isclose() in order to find almost-constant columns (see below).

    Returns
    ----
    inv_col_map, constant_columns, variable_columns

    - inv_col_map: Dictionary with all standard column names as keys and the respective input columns' names as values.
    - constant_columns: Dictionary of constant or not-found columns (with standard names as keys), e.g. {"weight": 1, "z":0.6, "time": 0, "polx": None}. 
    Also contains almost-constant columns' mean values, calculated via np.isclose(column, np.mean(column), rtol=rtol, atol=atol).
    NaN values are ignored in the calculations except when the entire column is NaN, then the constant yielded is None.
    - variable_columns: Dictionary of numpy arrays for the remaining columns (with standard names as keys).
    """

    # check inputs
    if len(set(data.columns)) < len(data.columns):
        raise ValueError("Input DataFrame must not have the same column name more than once")
    col_map = col_map if col_map is not None else {}
    for col,std_name in col_map.items():
        if col not in data.columns:
            raise ValueError("The key columns in col_map must exist in the data")
        if std_name not in DEFAULT_ALIASES_LOWERCASE.keys():
            raise ValueError("The value columns in col_map must be among the standard names")
    if len(set(col_map.values())) < len(col_map.values()):
        raise ValueError("No standard column name may be used more than once in the values of col_map")

    # build inv_col_map
    inv_col_map = {std_name:col for col,std_name in col_map.items()}
    for std_name, aliases in DEFAULT_ALIASES_LOWERCASE.items():
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
    constant_columns = {}
    variable_columns = {}    
    for std_name, col in inv_col_map.items():
        if col is None:
            constant_columns[std_name] = None
            continue
        else:
            series = data[col]
            series_without_NaN = series.dropna()
            if len(series_without_NaN) == 0:
                constant_columns[std_name] = None
            elif series_without_NaN.nunique() == 1:
                first = series_without_NaN.iloc[0]
                constant_columns[std_name] = first
            else:
                mean = series_without_NaN.mean()
                if np.all(np.isclose(series_without_NaN, mean, rtol=rtol, atol=atol)):
                    constant_columns[std_name] = mean
                else:
                    variable_columns[std_name] = series.copy() # WITH the NaN values

    all_columns = constant_columns | variable_columns # this is a shallow copy combining all keys (the values are references to the same objects as in the two individual dictionaries)

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
    return geometry


def augment_data(inputs):
    """
    Expands data using kdsource
    """
    return augmented_data



# def read_data(file_path, n=-1):
#     """
#     Read up to n particles from the mcpl file. n==-1 means read all.
#     """
#     blocklength = 10000
#     gen_file = mcpl.MCPLFile(file_path, blocklength=blocklength)
#     datas = []
#     for i, p in enumerate(gen_file.particle_blocks):
#         data = pd.DataFrame(
#             {
#                 "ekin": p.ekin,  # Kinetic energy
#                 "x": p.x,  # x-position
#                 "y": p.y,  # y-position
#                 "z": p.z,  # z-position
#                 "ux": p.ux,  # Velocity in x-direction
#                 "uy": p.uy,  # Velocity in y-direction
#                 "uz": p.uz,  # Velocity in z-direction
#                 "time": p.time,
#                 "weight": p.weight,  # Weight of the particle
#                 "pdgcode": p.pdgcode,  # Particle PDG code
#             }
#         )
#         datas.append(data)
#         if n != -1 and (i + 1) * blocklength >= n:
#             break
#     data = pd.concat(datas, ignore_index=True)
#     if n != -1 and len(data) > n:
#         data = data[:n]
#     return data

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
