import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("seaborn-poster") ### Use this for figures used in posters
# plt.style.use("seaborn-paper") ### Use this for figures used in paper
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

def run_omp_recovery(corrupted_data, pulse, freq_gap, n_nonzero_coefs):
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from scipy.linalg import circulant
    
    D = circulant(pulse)
    D_batch = np.expand_dims(D, axis=0)
    D_gap, _ = insert_freq_gaps(D_batch, D.shape, freq_gap)
    D_gap = D_gap[0, :, :]

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(D_gap, corrupted_data)
    coef = omp.coef_
    omp_reconstructed = np.transpose(np.dot(coef, D))
    return omp_reconstructed


def main(scene_type, dict_type, missing_rates, model_trained_epoch, gpu, savefile=False):

    # Get scene data
    scene_raw_data_mat, scene_image, n_samples, n_apertures, sampling_period = load_deterministic_scene_data(scene_type)
    scene_raw_data_batch = np.expand_dims(scene_raw_data_mat, axis=0)
    
    # Get SAR dictionary data    
    dict_filename, sar_dict_mat, n_samples, n_apertures, n_atoms, transmitted_pulse, transmitted_pulse_sampling_period = load_data_dict(dict_type=dict_type)
    processed_transmitted_pulse = np.concatenate((transmitted_pulse, np.zeros((900, 1))))
    
    # Get frequency parameters
    n_fft, sampling_freq, df, freq = get_frequency_parameters(n_samples, sampling_period)
    
    # Get frequency gaps
    from sar_data_utilities import freq_band
    n_missing_rates = missing_rates.shape[0]
    freq_gaps = get_frequency_gap_indices(freq_band, df, missing_rates)

    # SARGAN model
    img_size = (n_samples, n_apertures)
    batch_size = conf.batch_size
    model = SARGAN(img_size, batch_size)
    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    # model_path_test = conf.trained_models_path + "/checkpoint_backup/" + "%s_%s_model_%s.ckpt" % (data_name, experiment_name, model_trained_epoch)
#     model_path_test = conf.trained_models_path + "/checkpoint/" + "sar_dict_small_5_freq_corrupted_real_det_0_1_coef_range_dict_dist_20_model_199" + ".ckpt"
    model_path_test = os.path.join(DATA_PATH, "radarconf19_v3/trained_models/checkpoint/", "sar_dict_target_distance_{}_5_freq_corrupted_real_det_0_1_coef_range_dict_dist_{}_model_{}.ckpt".format(dict_type, dict_type, model_trained_epoch))
    fig_output_path = os.path.join(DATA_PATH, "radarconf19_paper/22074730sjjkfjywrsmx")
    data_output_path = os.path.join(DATA_PATH, "radarconf19_v4/outputs")
    saver = tf.train.Saver()    
    
    # GPU options
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=gpu)
    config=tf.ConfigProto(gpu_options=gpu_options)
    
    # RECOVERY
    in_snrs = []
    sargan_snr_gains = []
    sargan_out_snrs = []
    omp_out_snrs = []
    omp_snr_gains = []
    start_time = time.time()
    pbar = tqdm(range(n_missing_rates), unit="missing rate")
    scene_matfile_dict = {}
    corrupted = []
    sargan_rec = []
    omp_rec = []
    with tf.Session(config=config) as sess:

        saver.restore(sess, model_path_test)
        img = scene_raw_data_batch
        cond = img.copy()
        for i_missing_rate in pbar:
            freq_gap = freq_gaps[i_missing_rate]
            pcond, _ = insert_freq_gaps(cond, (n_samples, n_apertures), freq_gap)
            pimg, pcond = preprocess_test(img, pcond, (n_samples, n_apertures))

            # Recovery using SARGAN
            gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
            in_snr, sargan_out_snr, sargan_snr_gain = snr(pimg, pcond, gen_img, norm_flag=1, mag_flag=0)
            in_snrs.append(in_snr)
            sargan_out_snrs.append(sargan_out_snr)
            sargan_snr_gains.append(sargan_snr_gain)

            # Recovery using OMP
            omp_reconstructed = run_omp_recovery(cond[0, :, :], processed_transmitted_pulse, freq_gap, n_nonzero_coefs=100)
            omp_reconstructed = np.roll(omp_reconstructed, 300, axis=0)
            _, omp_out_snr, omp_snr_gain = snr(pimg[0, :, :, 0], pcond[0, :, :, 0], omp_reconstructed, norm_flag=1, mag_flag=0)
            omp_out_snrs.append(omp_out_snr)
            omp_snr_gains.append(omp_snr_gain)

            corrupted.append(pcond[0, :, :, 0])
            sargan_rec.append(gen_img[0, :, :, 0])
            omp_rec.append(omp_reconstructed)

    scene_matfile_dict = {
        "original": pimg[0, :, :, 0],
        "corrupted": corrupted,
        "sargan_rec": sargan_rec,
        "sargan_out_snr": sargan_out_snrs,
        "omp_out_snr": omp_out_snrs,
        "sargan_snr_gain": sargan_snr_gains,
        "omp_snr_gain": omp_snr_gains,
        "omp_rec": omp_rec,
        "missing_rates": missing_rates,
        "I_original": scene_image
    }
    scene_matfile_path = os.path.join(data_output_path, scene_type + "_dict_" + dict_type + "_scene_rec")
    if savefile:
        sio.savemat(scene_matfile_path + ".mat", scene_matfile_dict)
        print("Saved output to ", scene_matfile_path)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="gpu id", default=0)
    parser.add_argument("--dict", "-d", help="SAR dictionary type", default="20")
    parser.add_argument("--scene", "-s", help="Deterministic scene type", default="uniform")
    parser.add_argument("--epoch", "-e", help="Model trained epoch", default="160")
    parser.add_argument("--save", help="Save output to mat file", action="store_true")
    args = parser.parse_args()
    
    dict_type = args.dict
    scene_type = args.scene
#     missing_rates = np.asarray([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    missing_rates = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9])
    model_trained_epoch = args.epoch
    gpu = str(args.gpu)
    savefile = args.save
    main(scene_type, dict_type, missing_rates, model_trained_epoch, gpu, savefile)