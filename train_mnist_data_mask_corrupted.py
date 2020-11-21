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
    generate_raw_data, rotate_by_90_degree
import math
from alert_utilities import send_images_via_email
import time
import scipy.io as sio
import skimage.measure as ski_mem
import random
#import cv2
# import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

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

DATA_PATH = "/scratch/hle/data/trained_models"
import os
dict_filename = "mnist_small_mask_corrupted"
# dict_filename = "sar_dict_small"
# mask sizes and position of mask in the mnist images
mask_sizes = [4,5,6]
left_corner_indices = [12,13,14]

downsample_factor = 4
DOWNSAMPLE = False
n_samples = 28
n_apertures = 28
    
n_fft = n_samples



from sargan_config import Config as conf
    
img_size = (28, 28) #img_size = (n_samples, n_apertures)
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
# data_name = 'synthetic_sar'
data_name = dict_filename
experiment_name = "mnist_images_corrupted_by_center_mask"

output_path = os.path.join(conf.output_path, experiment_name)
print("OUTPUT PATH: {}".format(output_path))
NOISE_RANGE = [0.4,0.6]

ori_color = color_palette[1]
corrupted_color = color_palette[0]
reconstructed_color = color_palette[5]

def main(args):
#     if DOWNSAMPLE:
#         model = SARGANDOWNSAMPLE()
#     else:
#         model = SARGAN()
    gpu_id = args.gpu
    enable_email_alert = args.alert
    
    print("GPU ID: {}".format(gpu_id))
    print("IMG SIZE: ", img_size)
    print("BATCH SIZE: ", batch_size)
    model = SARGAN(img_size, batch_size)
    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    saver = tf.train.Saver(max_to_keep=20)
    
    if not os.path.exists(trained_models_path + "/checkpoint"):
        os.makedirs(trained_models_path + "/checkpoint")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(gpu_id))
#     gpu_options = tf.GPUOptions(allow_growth=True)
    config=tf.ConfigProto(gpu_options=gpu_options)
#     config.gpu_options.allow_growth = True
    n_train_iters_each_epoch = 1000
    n_test_iters = 50
    n_targets = 3
    sparsity_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    max_epoch = 100
#     coefficient_range = (0, 1)    
    coefficient_range = (-1, 1)    
    pbar = tqdm(range(max_epoch), unit="epoch")
    train_d_loss_values = []
    train_g_loss_values = []
    snr_gains = []
    real_snr_gains = []
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
                sparsity_rate = sparsity_rates[np.random.randint(0, len(sparsity_rates))]
#                 sparsity_rate = 0.2
                batch_img, _ = mnist.train.next_batch(20)
                reshape_batch_img = [np.array(img.reshape((28,28))) for img in batch_img]
                img = reshape_batch_img[0]
                cond = img.copy()
                #cond = np.transpose(cond)
                cond = add_gaussian_noise(cond, sd=np.random.uniform(NOISE_RANGE[0], NOISE_RANGE[1]))
                #mask_size = random.choice(mask_sizes)
                #left_corner_index = random.choice(left_corner_indices)
                #cond[left_corner_index : left_corner_index + mask_size, left_corner_index : left_corner_index + mask_size] = 0
                img = img.reshape(1, 784)
                cond = cond.reshape(1,784)
                img, cond = preprocess_train(img, cond, (28, 28), DOWNSAMPLE, downsample_factor)                
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
                      % (epoch, counter, time.time() - epoch_start_time, m, M), end="")
            
            if (epoch) % 5 == 0:
                save_path = saver.save(sess, conf.trained_models_path + "/checkpoint/" + "%s_%s_model_%s.ckpt" % (data_name, experiment_name, epoch+1))
                print("Model saved in file: %s" % save_path)
                
                # SYNTHETIC TESTING
                for i_test in range(n_test_iters):
                    sparsity_rate = sparsity_rates[np.random.randint(0, len(sparsity_rates))]
                    batch_img, _ = mnist.train.next_batch(64)
                    reshape_batch_img = [np.array(img.reshape((28,28))) for img in batch_img]
                    img = reshape_batch_img[0]
                    cond = img.copy()
                    #cond = np.transpose(cond)
                    cond = add_gaussian_noise(cond, sd=np.random.uniform(NOISE_RANGE[0], NOISE_RANGE[1]))
                    #mask_size = random.choice(mask_sizes)
                    #left_corner_index = random.choice(left_corner_indices)


                    #cond[left_corner_index : left_corner_index + mask_size, left_corner_index : left_corner_index + mask_size] = 0

                    img = img.reshape(1, 784)
                    pcond = cond.reshape(1,784)
                    pimg, pcond = preprocess_test(img, pcond, (28, 28), DOWNSAMPLE, downsample_factor)                    
                    gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
                    in_snr, out_snr, snr_gain = snr(pimg, pcond, gen_img, norm_flag=0, mag_flag=0)
                    epoch_snr_gain.append(snr_gain)
                
                ave_epoch_snr_gain = sum(epoch_snr_gain) / float(len(epoch_snr_gain))
                snr_gains.append(ave_epoch_snr_gain)
                
                in_snr, out_snr, snr_gain = snr(pimg[0, :, :, 0], pcond[0, :, :, 0], 
                                                gen_img[0, :, :, 0], norm_flag=0, mag_flag=0)
                
                fig = plt.figure(figsize=(14, 4)); ax = fig.add_subplot(131)
                ax.imshow(pimg[0, :, :, 0])
                ax.set_title("Original", color='grey')
                ax = fig.add_subplot(132)
                ax.imshow(pcond[0, :, :, 0])
                ax.set_title("Corrupted SNR = {:.2f} [dB]".format(in_snr), color='grey')
                ax = fig.add_subplot(133)
                ax.imshow(gen_img[0, :, :, 0])
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
                
                
                # REAL DATA TESTING
                batch_img, _ = mnist.train.next_batch(64)
                reshape_batch_img = [np.array(img.reshape((28,28))) for img in batch_img]
                img = reshape_batch_img[0]
                cond = img.copy()
                #cond = np.transpose(cond)
                cond = add_gaussian_noise(cond, sd=np.random.uniform(NOISE_RANGE[0], NOISE_RANGE[1]))
                #mask_size = random.choice(mask_sizes)
                #left_corner_index = random.choice(left_corner_indices)


                #cond[left_corner_index : left_corner_index + mask_size, left_corner_index : left_corner_index + mask_size] = 0


                img = img.reshape(1, 784)
                pcond = cond.reshape(1,784)
                #img = calibrated_real_raw_data_batch[0:batch_size, :, :]
                #cond = img.copy()
                #pcond, _ = insert_freq_gaps(cond, (ori_n_samples, ori_n_apertures), freq_gaps)
                pimg, pcond = preprocess_test(img, pcond, (28, 28), DOWNSAMPLE, downsample_factor)                    
                gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
                real_in_snr, real_out_snr, real_snr_gain = snr(pimg, pcond, gen_img, norm_flag=0, mag_flag=0)
                real_snr_gains.append(real_snr_gain)
                psnr_value = ski_mem.compare_psnr(pimg, gen_img, 1)
                # Time domain visualization: original vs corrupted vs recovered
                fig = plt.figure(figsize=(14, 4)); ax = fig.add_subplot(131)
                ax.imshow(pimg[0, :, :, 0])
                ax.set_title("Original", color='grey')
                ax = fig.add_subplot(132)
                ax.imshow(pcond[0, :, :, 0])
                ax.set_title("Corrupted SNR = {:.2f} [dB]".format(real_in_snr), color='grey')
                ax = fig.add_subplot(133)
                ax.imshow(gen_img[0, :, :, 0])
                ax.set_title("Reconstructed SNR = {:.2f} [dB]".format(real_out_snr), color='grey')
                plt.tight_layout()
                test_filename_raw_real = os.path.join(output_path, '%s_epoch_%s_raw_batchsize_%s_real.jpg' % (experiment_name, epoch, batch_size))
                plt.savefig(test_filename_raw_real, dpi=300)
                
                # Time domain visualization - one aperture: original vs corrupted vs recovered
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
                test_filename_one_aperture_real = os.path.join(output_path, '%s_epoch_%s_one_aperture_batchsize_%s_real.jpg' % (experiment_name, epoch, batch_size))
                plt.savefig(test_filename_one_aperture_real, dpi=300)
                

                # Real SNR
                fig = plt.figure(figsize=(16, 9));
                ax = fig.add_subplot(111)
                ax.plot(real_snr_gains); ax.set_title("Real PSNR Gain", color='grey'); ax.set_xlabel("Epoch")
                test_filename_snr_gain_real = os.path.join(output_path, '%s_epoch_%s_gain_batchsize_%s_real.jpg' % (experiment_name, epoch, batch_size))
                plt.savefig(test_filename_snr_gain_real, dpi=300)
                
                           
                epoch_running_time = time.time() - epoch_start_time
                
                ##### EMAIL ALERT
                email_alert_subject = "Epoch %s %s SARGAN %s" % (epoch+1, experiment_name.upper(), dict_filename)
                email_alert_text = """                
                    Epoch [{}/{}] Real SNR Gain {:.2f} [dB]
                    
                    Synthetic SNR Gain {:.2f} [dB]
                    
                    EXPERIMENT PARAMETERS:
                        Learning rate: {}
                        Batch size: {}
                        Max epoch number: {}
                        Training iterations each epoch: {}
                        Sparsity rates: {}
                    Running time: {:.2f} [s]
		    PSNR VALUE: {}
                """.format(epoch+1, max_epoch, real_snr_gain, ave_epoch_snr_gain,
                           learning_rate, batch_size,
                           max_epoch, n_train_iters_each_epoch,
                           str(sparsity_rates),
                           epoch_running_time, psnr_value)
                attachments = [
                    test_filename_raw, 
                    test_filename_one_aperture, 
                    test_filename_snr_gain, 
                    test_filename_raw_real,
                    test_filename_one_aperture_real,
                    test_filename_snr_gain_real
                ]
                send_images_via_email(email_alert_subject,
                                     email_alert_text,
                                     attachments,
                                     sender_email="hieule246@gmail.com", 
                                     recipient_emails=["hieule246@gmail.com"])
                plt.close("all")
                
if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="gpu id", default=0)
    parser.add_argument("--alert", help="Enable email alert", action="store_true")
    args = parser.parse_args()
    main(args)
