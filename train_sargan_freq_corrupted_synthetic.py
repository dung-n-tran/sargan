import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("seaborn-poster") ### Use this for figures used in posters
# plt.style.use("seaborn-paper") ### Use this for figures used in paper
# plt.style.use("seaborn-talk") ### Use this for figures used in presentations/talks
import scipy.misc
from sargan_models import SARGAN
from utils import imsave
import tensorflow as tf
import numpy as np
import time
import sys
from tqdm import tqdm
from sar_utilities import to_dB, visualize_raw_data, get_spectrum, \
    visualize_raw_data_spectrum, insert_freq_gaps, \
    generate_raw_data_from_dict, downsample, snr, \
    add_gaussian_noise, preprocess_train, preprocess_test, \
    generate_raw_data
import math
from alert_utilities import send_images_via_email
import time

mpl.rcParams["axes.grid"] = False
mpl.rcParams["grid.color"] = "#f5f5f5"
mpl.rcParams["axes.facecolor"] = "#ededed"
mpl.rcParams["axes.spines.left"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.bottom"] = False
mpl.rcParams['axes.labelcolor'] = "grey"
mpl.rcParams['xtick.color'] = 'grey'
mpl.rcParams['ytick.color'] = 'grey'
from cycler import cycler
import seaborn as sns
color_palette = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(color_palette)
mpl.rcParams["axes.prop_cycle"] = cycler('color', color_palette)

import pickle

DATA_PATH = "/data/dung/sargan"
import os
data_filename = "sar_dict_small"
with open(os.path.join(DATA_PATH, data_filename + ".pickle"), 'rb') as handle:
    sar_dict_data = pickle.load(handle)
    
sar_dict_mat = sar_dict_data["sar_dict_mat"]
ori_n_samples = sar_dict_data["n_samples"]
ori_n_apertures = sar_dict_data["n_apertures"]
sampling_period = sar_dict_data["transmistted_pulse_sample_period"]
n_atoms = sar_dict_mat.shape[1]


downsample_factor = 4
DOWNSAMPLE = False
if DOWNSAMPLE:
    n_samples = math.ceil(ori_n_samples / downsample_factor)
    n_apertures = math.ceil(ori_n_apertures / downsample_factor)
else:
    n_samples = ori_n_samples
    n_apertures = ori_n_apertures
    
n_fft = ori_n_samples
sampling_freq = 1. / (sampling_period + 1e-6)
df = sampling_freq / n_fft
freq = [df*i / 1e6 for i in range(0, n_fft)]    

freq_band = (0.005e6, 0.08e6)
freq_band_width = freq_band[1] - freq_band[0]
missing_rate = 0.8
missing_band_width = round(freq_band_width * missing_rate)
gap_width = missing_band_width
f_start = np.array([freq_band[0] + round(0.1*freq_band_width)])
f_end = f_start + gap_width
f_start_idx = np.int_(np.ceil(f_start / df))
f_end_idx = np.int_(np.ceil(f_end / df))
freq_gaps = [f_start_idx, f_end_idx]


from sargan_config import Config as conf
    
img_size = (n_samples, n_apertures)
train_size = img_size
batch_size = conf.batch_size
img_channel = conf.img_channel
conv_channel_base = conf.conv_channel_base

learning_rate = conf.learning_rate
beta1 = conf.beta1
max_epoch = conf.max_epoch
L1_lambda = conf.L1_lambda
save_per_epoch = conf.save_per_epoch

trained_models_path = conf.trained_models_path
data_name = 'synthetic_sar'
experiment_name = "freq_corrupted"
db_range = 9 # For visualizing raw dta

output_path = os.path.join(conf.output_path, experiment_name)


ori_color = color_palette[1]
corrupted_color = color_palette[0]
reconstructed_color = color_palette[5]

def main():
#     if DOWNSAMPLE:
#         model = SARGANDOWNSAMPLE()
#     else:
#         model = SARGAN()
    print("IMG SIZE: ", img_size)
    model = SARGAN(img_size, batch_size)
    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    saver = tf.train.Saver()

    start_time = time.time()
    if not os.path.exists(trained_models_path + "/checkpoint"):
        os.makedirs(trained_models_path + "/checkpoint")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

#     gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list='1')
    gpu_options = tf.GPUOptions(allow_growth=True)
    config=tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    n_train_iters_each_epoch = 1000
    n_test_iters = 200
    n_targets = 3
    sparsity_rate = 0.2
    max_epoch = 100
        
    pbar = tqdm(range(max_epoch), unit="epoch")
    train_d_loss_values = []
    train_g_loss_values = []
    snr_gains = []
    with tf.Session(config=config) as sess:
        if conf.model_path_train == "":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, conf.model_path_train)    
        for epoch in pbar:
            counter = 0
            epoch_snr_gain = []
            epoch_start_time = time.time()
            for i_train in range(n_train_iters_each_epoch):
#                 img = generate_raw_data_from_dict(sar_dict_mat, n_targets=n_targets, batch_size=batch_size)
                img = generate_raw_data(sar_dict_mat, sparsity_rate, batch_size)
                cond = img.copy()
#                 cond = add_gaussian_noise(cond, sd=noise_standard_deviation)
                cond, _ = insert_freq_gaps(cond, (ori_n_samples, ori_n_apertures), freq_gaps)
                img, cond = preprocess_train(img, cond, (ori_n_samples, ori_n_apertures), DOWNSAMPLE, downsample_factor)                
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, M = sess.run([g_opt, model.g_loss], feed_dict={model.image:img, model.cond:cond})            
                train_d_loss_values.append(m)
                train_g_loss_values.append(M)
                counter += 1
    #             if counter > 1:
    #                 break
                if counter % 100 == 0:
                    print("\rEpoch [%d], Iteration [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, counter, time.time() - start_time, m, M), end="")
            if (epoch) % conf.save_per_epoch == 0:
                save_path = saver.save(sess, conf.trained_models_path + "/checkpoint/" + "%s_%s_model_%s.ckpt" % (data_name, experiment_name, epoch+1))
                print("Model saved in file: %s" % save_path)
                
                for i_test in range(n_test_iters):
                    #img = generate_raw_data_from_dict(sar_dict_mat, n_targets=n_targets, batch_size=batch_size)
                    img = generate_raw_data(sar_dict_mat, sparsity_rate, batch_size)
                    cond = img.copy()
                    pcond, _ = insert_freq_gaps(cond, (ori_n_samples, ori_n_apertures), freq_gaps)
                    pimg, pcond = preprocess_test(img, pcond, (ori_n_samples, ori_n_apertures), DOWNSAMPLE, downsample_factor)                    
                    gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
                    in_snr, out_snr, snr_gain = snr(pimg, pcond, gen_img, norm_flag=0, mag_flag=0)
                    epoch_snr_gain.append(snr_gain)
                
                ave_epoch_snr_gain = sum(epoch_snr_gain) / float(len(epoch_snr_gain))
                snr_gains.append(ave_epoch_snr_gain)
                
                in_snr, out_snr, snr_gain = snr(pimg[0, :, :, 0], pcond[0, :, :, 0], 
                                                gen_img[0, :, :, 0], norm_flag=0, mag_flag=0)
                fig = plt.figure(figsize=(14, 4)); ax = fig.add_subplot(131)
                ax = visualize_raw_data(pimg[0, :, :, 0], ax, db_range)
                ax.set_title("Original", color='grey')
                ax = fig.add_subplot(132)
                ax = visualize_raw_data(pcond[0, :, :, 0], ax, db_range)
                ax.set_title("Corrupted SNR = {:.2f} [dB]".format(in_snr), color='grey')
                ax = fig.add_subplot(133)
                ax = visualize_raw_data(gen_img[0, :, :, 0], ax, db_range)
                ax.set_title("Reconstructed SNR = {:.2f} [dB]".format(out_snr), color='grey')
                plt.tight_layout()
                test_filename_raw = os.path.join(output_path, '%s_epoch_%s_raw_batchsize_%s.jpg' % (experiment_name, epoch, batch_size))
                plt.savefig(test_filename_raw, dpi=300)

                # Time domain visualization: original vs corrupted vs recovered
                line_idx = 10
                fig = plt.figure(figsize=(16, 9)); ax = fig.add_subplot(111);
                ax.plot(pimg[0, :, line_idx, 0], color=ori_color, lw=2.0)
                ax.plot(pcond[0, :, line_idx, 0], color=corrupted_color, lw=2.0)
                ax.plot(gen_img[0, :, line_idx, 0], color=reconstructed_color, lw=2.0)
                ax.fill_between(range(pimg[0, :, line_idx, 0].shape[0]), pimg[0, :, line_idx, 0],
                                gen_img[0, :, line_idx, 0], color=reconstructed_color, alpha=0.3)
                ax.set_title("Time domain - One aperture", color='grey')
                legend = ax.legend(["Original", "Corrupted", "Reconstructed"]); plt.setp(legend.get_texts(), color='grey')
                ax.set_xlim([0, pimg[0, :, line_idx, 0].shape[0]])
                test_filename_one_aperture = os.path.join(output_path, '%s_epoch_%s_one_aperture_batchsize_%s.jpg' % (experiment_name, epoch, batch_size))
                plt.savefig(test_filename_one_aperture, dpi=300)
                                                
                fig = plt.figure(figsize=(16, 9)); 
                ax = fig.add_subplot(211)
                ax.plot(train_d_loss_values)
                ax.set_title("D loss", color='grey')
                ax.set_xlabel("Iteration")

                ax = fig.add_subplot(212)
                ax.set_title("G loss", color='grey')
                ax.plot(train_g_loss_values)
                ax.set_xlabel("Iteration")
                plt.tight_layout()
#                 plt.savefig(os.path.join(output_path, '%s_training_losses_epoch_%s.jpg' % (experiment_name, epoch)))

                fig = plt.figure(figsize=(16, 9));
                ax = fig.add_subplot(111)
                ax.plot(snr_gains); ax.set_title("PSNR Gain", color='grey'); ax.set_xlabel("Epoch")
                test_filename_snr_gain = os.path.join(output_path, '%s_epoch_%s_gain_batchsize_%s.jpg' % (experiment_name, epoch, batch_size))
                plt.savefig(test_filename_snr_gain, dpi=300)
                
                # Spectrum visualization: original vs corrupted vs reconstructed
                n_fft = n_samples
                zoom_factor = 1
                fig = plt.figure(figsize=(16, 9)); ax = fig.add_subplot(111);
                raw_spectrum = get_spectrum(pimg[0, :, :, 0], n_fft)
                raw_spectrum_dB = to_dB(np.mean(raw_spectrum[0:n_fft//zoom_factor, :], axis=1))
                corrupted_spectrum = get_spectrum(pcond[0, :, :, 0], n_fft)
                corrupted_spectrum_dB = to_dB(np.mean(corrupted_spectrum[0:n_fft//zoom_factor, :], axis=1))
                corrupted_spectrum_dB[corrupted_spectrum_dB==-np.inf] = -170
                recovered_spectrum = get_spectrum(gen_img[0, :, :, 0], n_fft)
                recovered_spectrum_dB = to_dB(np.mean(recovered_spectrum[0:n_fft//zoom_factor, :], axis=1))
                ax.plot(freq[0:n_fft // zoom_factor], raw_spectrum_dB, color=ori_color, lw=2.0)
                ax.plot(freq[0:n_fft // zoom_factor], corrupted_spectrum_dB, color=corrupted_color, lw=2.0)
                ax.plot(freq[0:n_fft // zoom_factor], recovered_spectrum_dB, color=reconstructed_color, lw=3.0)
                ax.set_ylim(ymin=-50); ax.set_xlim([0.005, 0.08])

                ax.set_title("Spectrum: original vs corrupted vs recovered", color='grey')
                legend = ax.legend(["Original", "Corrupted", "Recovered"]); plt.setp(legend.get_texts(), color='grey')
                ax.set_xlabel("Frequency [MHz]"); ax.set_ylabel("dB")
                ax.fill_between(freq[0:n_fft // zoom_factor], corrupted_spectrum_dB, -90, color=corrupted_color, alpha=0.1)
                test_filename_spectrum = os.path.join(output_path, '%s_epoch_%s_spectrum_batchsize_%s.jpg' % (experiment_name, epoch, batch_size))
                plt.savefig(test_filename_spectrum, dpi=300)
                
                epoch_running_time = time.time() - epoch_start_time
                email_alert_subject = "Epoch %s %s SARGAN unnormalized train results" % (epoch, experiment_name.upper())                
                email_alert_text = """                
                    Epoch [{}/{}] SNR Gain {:.2f} [dB]
                    
                    EXPERIMENT PARAMETERS:
                        Frequency missing rate: {}
                        Learning rate: {}
                        Batch size: {}
                        Max epoch number: {}
                        Training iterations each epoch: {}
                        
                    Running time: {:.2f} [s]
                """.format(epoch+1, max_epoch, ave_epoch_snr_gain,
                           missing_rate, learning_rate, batch_size,
                           max_epoch, n_train_iters_each_epoch,
                           epoch_running_time)
                attachments = [
                    test_filename_raw, 
                    test_filename_one_aperture, 
                    test_filename_spectrum, 
                    test_filename_snr_gain
                ]
                send_images_via_email(email_alert_subject,
                                     email_alert_text,
                                     attachments,
                                     sender_email="ozawamariajp@gmail.com", 
                                     recipient_emails=["ngocdungxyz@gmail.com"])
                plt.close("all")
                
if __name__=="__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu=':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1][4:])
        print("GPU options:", str(sys.argv[1][4:]))
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(0)

    main()
