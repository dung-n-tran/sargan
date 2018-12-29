import pickle
import numpy as np
import os
DATA_PATH = "/data/dung/sargan"
freq_band = (300e6, 3000e6)
def load_data_dict(dict_type="20"):  
    float_type = np.float32
    dict_filename = "sar_dict_target_distance_" + dict_type
    with open(os.path.join(DATA_PATH, dict_filename + "_no_downsampling" + ".pickle"), 'rb') as handle:
        sar_dict_data = pickle.load(handle)
    sar_dict_mat = sar_dict_data["sar_dict_mat"].astype(float_type)
    transmitted_pulse = sar_dict_data["transmistted_pulse"].astype(float_type)
    n_samples = sar_dict_data["n_samples"]
    n_apertures = sar_dict_data["n_apertures"]
    sampling_period = sar_dict_data["transmistted_pulse_sample_period"].astype(float_type)
    n_atoms = sar_dict_mat.shape[1]
    
    return dict_filename, sar_dict_mat, n_samples, n_apertures, n_atoms, transmitted_pulse, sampling_period

def load_deterministic_scene_data(scene_type = "uniform"):
    scene_data_filename = "deterministic_" + scene_type + "_scene_dict_atom_distance_20"
    with open(os.path.join(DATA_PATH, scene_data_filename + ".pickle"), 'rb') as handle:
        sar_scene_data = pickle.load(handle)
    scene_raw_data_mat = sar_scene_data["scene_raw_data_mat"]
    scene_image = sar_scene_data["scene_image"]
    n_samples = sar_scene_data["n_samples"]
    n_apertures = sar_scene_data["n_apertures"]
    sampling_period = sar_scene_data["transmistted_pulse_sample_period"]
    return scene_raw_data_mat, scene_image, n_samples, n_apertures, sampling_period