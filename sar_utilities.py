import numpy as np
import scipy.io as sio
import glob
import os
import matplotlib.pyplot as plt

def to_dB(s):
    return 20*np.log10(np.abs(s))

def get_fft_num(n, up_factor):
    return up_factor*int(pow(2, np.ceil(np.log2(n))))

def get_frequency_parameters(n_samples, sampling_period):
    n_fft = n_samples
    sampling_freq = 1. / (sampling_period + 1e-32)
    df = sampling_freq / n_fft
    freq = [df*i / 1e9 for i in range(0, n_fft)] 
    return n_fft, sampling_freq, df, freq

def get_frequency_gap_indices(freq_band, df, missing_rates):
    freq_band_width = freq_band[1] - freq_band[0]
    n_missing_rates = missing_rates.shape[0]
    freq_gaps = []
    for missing_rate in missing_rates:
        missing_band_width = round(freq_band_width * missing_rate)
        gap_width = missing_band_width
        f_start = np.array([freq_band[0] + round(0.5 * (freq_band_width - missing_band_width))])
        f_end = f_start + gap_width
        f_start_idx = np.int_(np.ceil(f_start / df))
        f_end_idx = np.int_(np.ceil(f_end / df))
        freq_gap = [f_start_idx, f_end_idx]
        freq_gaps.append(freq_gap)
    return freq_gaps
    
def rms(x):
    from numpy import mean, sqrt, square, arange
    return sqrt(mean(square(x)))

def snr(truth, x_in, x_out, norm_flag=0, mag_flag=0):
    import numpy as np
    from scipy.signal import hilbert
    if mag_flag == 0:
        maxtruth = np.abs(hilbert(truth)).max()
        maxin = np.abs(hilbert(x_in)).max()
        maxout = np.abs(hilbert(x_out)).max()
    else:
        maxtruth = truth.max()
        maxin = x_in.max()
        maxout = x_out.max()
    
    if norm_flag:
        truth = truth/maxtruth;
        x_in = x_in/maxin;
        x_out = x_out/maxout;
        
    snr_in = 20*np.log10( rms(truth) / rms(x_in - truth) )
    snr_out = 20*np.log10( rms(truth) / rms(x_out - truth) )
    snr_gain = snr_out - snr_in
    return snr_in, snr_out, snr_gain

def visualize_raw_data(raw_data, ax=None, db_range=20, set_ticks=False):
    raw_data_mat_db = to_dB(raw_data)
    if not ax:
        fig = plt.figure(); ax = fig.add_subplot(111)
    ax.imshow(raw_data_mat_db, vmax=max(raw_data_mat_db.flatten()), vmin=max(raw_data_mat_db.flatten()) - db_range, cmap='jet', aspect="auto")
    if not set_ticks:
        ax.set_xticks([]); ax.set_yticks([])
    return ax
def get_spectrum(raw_data, n_fft):
    return np.fft.fft(raw_data, n_fft, axis=0)

def visualize_raw_data_spectrum(raw_data, freq, n_fft, zoom_factor=1):    
    fig = plt.figure(); ax = fig.add_subplot(111);
    raw_data_spectrum = get_spectrum(raw_data, n_fft)
    ax.plot(freq[0:n_fft // zoom_factor], to_dB(np.mean(abs(raw_data_spectrum[0:n_fft // zoom_factor, :]), axis=1)), color='#333333', lw=2.0)
    return ax

def insert_freq_gaps(ori_img, img_size, freq_gaps):
    if len(ori_img.shape) == 1:
        img = np.reshape(ori_img, (1, img_size[0], img_size[1]))
    else:
        img = ori_img.reshape(ori_img.shape[0], img_size[0], img_size[1])
    n_fft = img_size[0]
    identity = np.identity(n_fft)
    dft_mtx = np.fft.fft(identity, n_fft)
    idft_mtx = np.fft.ifft(identity, n_fft)
    mask = np.ones((n_fft, 1))
    f_start_idx, f_end_idx = freq_gaps
    for i in range(len(f_start_idx)):    
        mask[f_start_idx[i] : f_end_idx[i], :] = 0.0 + 0.0j
    mask[n_fft // 2 + 1: n_fft - 1, :] = np.conj(mask[n_fft // 2 - 1: 1: -1, :]) 
    masked_freq_dict = np.multiply(mask, dft_mtx)
    corrupted_imgs = []
    corrupted_spectrum = []
    for i in range(img.shape[0]):
        tmp = np.dot(masked_freq_dict, img[i, :, :])
        corrupted_spectrum.append(tmp)
        corrupted_imgs.append(np.dot(idft_mtx, tmp).real[0:img_size[0], :])
    corrupted_spectrum = np.asarray(corrupted_spectrum)
    corrupted_imgs = np.asarray(corrupted_imgs)
    return corrupted_imgs, corrupted_spectrum

def generate_raw_data_from_dict(sar_dict_mat, n_targets, batch_size):
    from random import randint
    from random import shuffle
    n_atoms = sar_dict_mat.shape[1]
    raw_data_batch = []
    for i_batch in range(batch_size):
        atom_range = np.arange(n_atoms)
        shuffle(atom_range)
        target_indices = atom_range[:n_targets]
        raw_data = np.dot(sar_dict_mat[:, target_indices], np.random.uniform(0.5, 1, size=n_targets))
        raw_data_batch.append(raw_data)
    return np.asarray(raw_data_batch)

def generate_raw_data(sar_dict_mat, sparsity_rate, batch_size, coefficient_range=(0, 1)):
    
    from scipy import stats
    from scipy import sparse
#     class CustomRandomState(object):
#         def randint(self, k):
#             i = np.random.randint(k)
#             return i - i % 2
#     rs = CustomRandomState()
    uniform_loc = coefficient_range[0]
    uniform_scale = coefficient_range[1] - coefficient_range[0]
    rvs = stats.uniform(loc=uniform_loc, scale=uniform_scale).rvs
    coef_mtx = sparse.random(batch_size, sar_dict_mat.shape[1],
                             format="csr", density=sparsity_rate, data_rvs=rvs)
    raw_data_batch = coef_mtx.A.dot(np.transpose(sar_dict_mat))
    return raw_data_batch

def downsample(img, axis_0_factor=None, axis_1_factor=None):
    import scipy.signal as ss
    if axis_0_factor is not None:
        img = ss.decimate(img, axis_0_factor, axis=0)        
    if axis_1_factor is not None:
        img = ss.decimate(img, axis_1_factor, axis=1)
    return img

def add_gaussian_noise(img, sd=1):
    mean = 0
    noise = np.random.normal(mean, sd, img.shape)
    return img + noise

def preprocess_train(img, cond, img_size, DOWNSAMPLE=False, downsample_factor=1, img_channel=1):
    img = img.reshape(img.shape[0], img_size[0], img_size[1])
    cond = cond.reshape(cond.shape[0], img_size[0], img_size[1])
    img = img.reshape(img.shape[0], img_size[0], img_size[1], img_channel)
    cond = cond.reshape(img.shape[0], img_size[0], img_size[1], img_channel)
    downsampled_img = []
    downsampled_cond = []
    if DOWNSAMPLE:
        for i in range(img.shape[0]):
            downsampled_img.append(downsample(img[i, :, :], axis_0_factor=downsample_factor, axis_1_factor=downsample_factor))
            downsampled_cond.append(downsample(cond[i, :, :], axis_0_factor=downsample_factor, axis_1_factor=downsample_factor))
        downsampled_img = np.asarray(downsampled_img)
        downsampled_cond = np.asarray(downsampled_cond)
    else:
        downsampled_img = img
        downsampled_cond = cond
    
    return downsampled_img, downsampled_cond

def preprocess_test(img, cond, train_size, DOWNSAMPLE=False, downsample_factor=1, img_channel=1):

    img = img.reshape(img.shape[0], train_size[0], train_size[1])
    cond = cond.reshape(cond.shape[0], train_size[0], train_size[1])
    img = img.reshape(img.shape[0], train_size[0], train_size[1], img_channel)
    cond = cond.reshape(cond.shape[0], train_size[0], train_size[1], img_channel)    
    downsampled_img = []
    downsampled_cond = []
    if DOWNSAMPLE:
        for i in range(img.shape[0]):
            downsampled_img.append(downsample(img[i, :, :], axis_0_factor=downsample_factor, axis_1_factor=downsample_factor))
            downsampled_cond.append(downsample(cond[i, :, :], axis_0_factor=downsample_factor, 
                                               axis_1_factor=downsample_factor))
        downsampled_img = np.asarray(downsampled_img)
        downsampled_cond = np.asarray(downsampled_cond)
    else:
        downsampled_img = img
        downsampled_cond = cond
    
    return downsampled_img, downsampled_cond