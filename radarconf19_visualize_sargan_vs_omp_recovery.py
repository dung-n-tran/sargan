import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use("seaborn-poster") ### Use this for figures used in posters
plt.style.use("seaborn-paper") ### Use this for figures used in paper
# plt.style.use("seaborn-talk") ### Use this for figures used in presentations/talks
from sargan_models import SARGAN
import tensorflow as tf
import numpy as np
import time
import sys, os
from tqdm import tqdm
from sar_utilities import to_dB, visualize_raw_data, get_spectrum, \
    visualize_raw_data_spectrum, insert_freq_gaps, \
    generate_raw_data_from_dict, downsample, snr, \
    add_gaussian_noise, preprocess_train, preprocess_test, \
    generate_raw_data, get_frequency_parameters, get_frequency_gap_indices
from sargan_config import Config as conf
import math
from alert_utilities import send_images_via_email
import scipy.io as sio
from chart_utilities import config_figures, color_palette
import seaborn as sns
from sar_data_utilities import DATA_PATH, load_data_dict, load_deterministic_scene_data

config_figures(mpl, color_palette)
ori_color = color_palette[0]
corrupted_color = color_palette[1]
reconstructed_color = color_palette[2]
omp_color = color_palette[7]

def main(scene_type, dict_type, savefile=False):
    
    data_output_path = os.path.join(DATA_PATH, "radarconf19_v4/outputs")
    scene_matfile_path = os.path.join(data_output_path, scene_type + "_dict_" + dict_type + "_scene_rec")
    
    float_type = np.float32
    img_rec_matfile_path = scene_matfile_path + "_img"
    img_rec_data = sio.loadmat(img_rec_matfile_path + ".mat")
    I_original = img_rec_data["I_original"].astype(float_type)
    I_corrupted = img_rec_data["I_corrupted"].astype(float_type)
    I_sargan = img_rec_data["I_sargan"].astype(float_type)
    I_omp = img_rec_data["I_omp"].astype(float_type)
    corrupted = img_rec_data["corrupted"].astype(float_type)
    sargan_rec = img_rec_data["sargan_rec"].astype(float_type)
    omp_rec = img_rec_data["omp_rec"].astype(float_type)
    original = img_rec_data["original"].astype(float_type)
    missing_rates = img_rec_data["missing_rates"][0].astype(float_type)
    print("\nMISSING RATES:", missing_rates)
    n_missing_rates = missing_rates.shape[0]
        
    # Get scene data
    scene_raw_data_mat, scene_image, n_samples, n_apertures, sampling_period = load_deterministic_scene_data(scene_type)
    scene_raw_data_batch = np.expand_dims(scene_raw_data_mat, axis=0)
    
    # Get frequency parameters
    n_fft, sampling_freq, df, freq = get_frequency_parameters(n_samples, sampling_period)
    
    # Get frequency gaps
    from sar_data_utilities import freq_band
    freq_gaps = get_frequency_gap_indices(freq_band, df, missing_rates)

    db_range = 50
    for i_missing_rate in range(n_missing_rates):
        fig = plt.figure(figsize=(20, 9))

        missing_rate = missing_rates[i_missing_rate]
        missing_rate_str = "{:.0f}".format(missing_rate*100)
        corrupted_i = corrupted[i_missing_rate]
        sargan_rec_i = sargan_rec[i_missing_rate]
        omp_rec_i = omp_rec[i_missing_rate]
        I_corrupted_i = I_corrupted[i_missing_rate]
        I_sargan_i = I_sargan[i_missing_rate]
        I_omp_i = I_omp[i_missing_rate]        
        
        # Original raw data
        ax = fig.add_subplot(3, 4, 1)
        ax = visualize_raw_data(original, ax, db_range=db_range)

        # Corrupted raw data
        ax = fig.add_subplot(3, 4, 2)
        ax = visualize_raw_data(corrupted_i, ax, db_range=db_range)

        # SARGAN-recovered raw data
        ax = fig.add_subplot(3, 4, 3)
        ax = visualize_raw_data(sargan_rec_i, ax, db_range=db_range)

        # OMP-recovered raw data
        ax = fig.add_subplot(3, 4, 4)
        ax = visualize_raw_data(omp_rec_i, ax, db_range=db_range)

        # Original image
        ax = fig.add_subplot(3, 4, 5)
        ax = visualize_raw_data(I_original[0:250, 50:450], ax, db_range=db_range)

        # Corrupted image
        ax = fig.add_subplot(3, 4, 6)
        ax = visualize_raw_data(I_corrupted_i[0:250, 50:450], ax, db_range=db_range)

        # SARGAN-recovered image
        ax = fig.add_subplot(3, 4, 7)
        ax = visualize_raw_data(I_sargan_i[0:250, 50:450], ax, db_range=db_range)

        # OMP-recovered image
        ax = fig.add_subplot(3, 4, 8)
        ax = visualize_raw_data(I_omp_i[0:250, 50:450], ax, db_range=db_range)

        # Time domain - one aperture
        i_aperture = 0; lw = 1.5
        ax = fig.add_subplot(349)
        ax.plot(original[:, i_aperture], color=ori_color, lw=lw, label="Original")
        ax.plot(corrupted_i[:, i_aperture], color=corrupted_color, lw=lw, label="Corrupted")
        ax.plot(sargan_rec_i[:, i_aperture], color=reconstructed_color, lw=lw, label="Recovered by SARGAN")
        ax.set_xlim([0, n_samples])
        ax.set_xlabel("Sample"); ax.set_ylabel("Magnitude")
        legend_handler = ax.legend(); plt.setp(legend_handler.get_texts(), color='gray')

        ax = fig.add_subplot(3, 4, 10)
        ax.plot(original[:, i_aperture], color=ori_color, lw=lw, label="Original")
        ax.plot(corrupted_i[:, i_aperture], color=corrupted_color, lw=lw, label="Corrupted")
        ax.plot(omp_rec_i[:, i_aperture], color=omp_color, lw=lw, label="Recovered by OMP")
        ax.set_xlim([0, n_samples])
        legend_handler = ax.legend(); plt.setp(legend_handler.get_texts(), color='gray')
        ax.set_xlabel("Sample"); ax.set_ylabel("Magnitude")

        ori_spectrum = get_spectrum(original[:, i_aperture], n_fft); ori_spectrum_dB = to_dB(ori_spectrum)
        corrupted_spectrum = get_spectrum(corrupted_i[:, i_aperture], n_fft); corrupted_spectrum_dB = to_dB(corrupted_spectrum)
        corrupted_spectrum_dB[corrupted_spectrum_dB < -200] = -50
        sargan_spectrum = get_spectrum(sargan_rec_i[:, i_aperture], n_fft); sargan_spectrum_dB = to_dB(sargan_spectrum)
        omp_spectrum = get_spectrum(omp_rec_i[:, i_aperture], n_fft); omp_spectrum_dB = to_dB(omp_spectrum)
        zoom_factor = 8
        x_range = [0.2, 3]; y_min = 20
        ax = fig.add_subplot(3, 4, 11)
        ax.plot(freq[0:n_fft // zoom_factor], ori_spectrum_dB[0:n_fft//zoom_factor], color=ori_color, lw=lw, label="Original")
        ax.plot(freq[0:n_fft // zoom_factor], corrupted_spectrum_dB[0:n_fft//zoom_factor], color=corrupted_color, lw=lw, label="Corrupted")
        ax.plot(freq[0:n_fft // zoom_factor], sargan_spectrum_dB[0:n_fft//zoom_factor], color=reconstructed_color, lw=lw, label="Recovered by SARGAN")
        ax.fill_between(freq[0:n_fft // zoom_factor], corrupted_spectrum_dB[0:n_fft//zoom_factor], y_min, color=corrupted_color, alpha=0.1)
        ax.set_ylim(ymin=y_min); ax.set_xlim(x_range)
        # ax.set_title("Spectrum: deterministic scene", color="gray")
        ax.set_xlabel("Frequency [GHz]"); ax.set_ylabel("Magnitude [dB]")
        legend_handler = ax.legend(); plt.setp(legend_handler.get_texts(), color='gray')

        ax = fig.add_subplot(3, 4, 12)
        ax.plot(freq[0:n_fft // zoom_factor], ori_spectrum_dB[0:n_fft//zoom_factor], color=ori_color, lw=lw, label="Original")
        ax.plot(freq[0:n_fft // zoom_factor], corrupted_spectrum_dB[0:n_fft//zoom_factor], color=corrupted_color, lw=lw, label="Corrupted")
        ax.plot(freq[0:n_fft // zoom_factor], omp_spectrum_dB[0:n_fft//zoom_factor], color=omp_color, lw=lw, label="Recovered by OMP")
        ax.fill_between(freq[0:n_fft // zoom_factor], corrupted_spectrum_dB[0:n_fft//zoom_factor], y_min, color=corrupted_color, alpha=0.1)
        ax.set_ylim(ymin=y_min); ax.set_xlim(x_range)
        # # ax.set_title("Spectrum: deterministic scene", color="gray")
        ax.set_xlabel("Frequency [GHz]"); ax.set_ylabel("Magnitude [dB]")
        legend_handler = ax.legend(); plt.setp(legend_handler.get_texts(), color='gray')

        plt.tight_layout()
        if savefile:
            fig_output_path = os.path.join(DATA_PATH, "radarconf19_paper/22074730sjjkfjywrsmx/figures")
            recovery_filename = scene_type + "_dict_" + dict_type + "_" + missing_rate_str + "_missing_sargan_omp_image_raw_one_aperture.jpg"
            plt.savefig(os.path.join(fig_output_path, recovery_filename), dpi=300)
            
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", "-d", help="SAR dictionary type", default="20")
    parser.add_argument("--scene", "-s", help="Deterministic scene type", default="uniform")
    parser.add_argument("--save", help="Save output to mat file", action="store_true")
    args = parser.parse_args()
    
    dict_type = args.dict
    scene_type = args.scene
    savefile = args.save
    main(scene_type, dict_type, savefile)