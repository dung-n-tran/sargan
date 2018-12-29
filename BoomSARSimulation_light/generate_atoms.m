clc; close all; clear;

addpath('utils');
pulse_data = load('DataSets/TxPulse.mat');
load('para.mat');
radar_center_location_idx = ceil(length(xRadar)/2);
xdelta = 0.145;
ydelta = 0.145;
xlen = 500;
ylen = 750;
xcenter = -645;
ycenter = 75;

xrange = 0.5*xlen*xdelta;
yrange = 0.5*ylen*ydelta;

grid_x = xcenter-xrange:xdelta:xcenter+xrange;
grid_y = ycenter-yrange:ydelta:ycenter+yrange;
[g_x, g_y] = meshgrid(grid_x, grid_y);
g_z = zeros(size(g_x));

% Generates coordinates of a target for each grid point 
% within a box on the scene
% Amplitude of each target is K*10dB
box_x_center_idx = ceil(length(grid_x)/2);
box_y_center_idx = ceil(length(grid_y)/2);
box_x_half_length = floor(length(grid_x)/2)-100;
box_x_half_length = 50;
box_y_half_length = box_x_half_length;

%%%%%%%% TUNE THIS FOR DICTIONARY RESULTION
target_x_distance = 5;

target_y_distance = target_x_distance;
box_x_locations = grid_x(box_x_center_idx - box_x_half_length : ...
    target_x_distance : box_x_center_idx + box_x_half_length);
box_y_locations = grid_y(box_y_center_idx - 6*box_y_half_length : ...
    target_y_distance : box_y_center_idx - 4*box_y_half_length);
[box_grid_x, box_grid_y] = meshgrid(box_x_locations, box_y_locations);
box_grid_z = zeros(size(box_grid_x));
K = 1;

% x = (xrange(2)-xrange(1)).*rand(1, N)+xrange(1);
% y = (yrange(2)-yrange(1)).*rand(1, N)+yrange(1);
% 
% 
a = 10*10.^(K*ones(size(box_grid_x))); %amplitudes
% a = 0.1 * 0.1.^(K*ones(size(box_grid_x))); %amplitudes
% 
targets = [box_grid_x(:) box_grid_y(:) box_grid_z(:) a(:)];
% targets = [grid_x(251) grid_y(251) 0 a(1)];
n_targets = size(targets, 1);

s = pulse_data.st;
n_samples = 1500;%16000;
% num_samples = length(s);
pulse = [s; zeros(n_samples-length(s), 1)];
size(s);

fs = 8*10^9; %Sampling rate = 8GHz
[m, I] = max(pulse);
t0 = (I-1)/fs;

% Raw time-domain data
raw_data_matrices = [];
radar_half_num_locations = 150;
radar_location_indices = radar_center_location_idx - radar_half_num_locations : ...
    radar_center_location_idx + radar_half_num_locations;
%radar_location_indices = 1:length(xRadar);
xRadar = xRadar(radar_location_indices);
yRadar = yRadar(radar_location_indices);
zRadar = zRadar(radar_location_indices);
sum_raw = zeros(n_samples, length(xRadar));
n_apertures = length(xRadar);
sar_dict = zeros(n_samples * n_apertures, n_targets);
for i_target = 1:size(targets, 1)
    raw_data = generateTDRawData(xRadar, yRadar, zRadar,...
        xRadar, yRadar, zRadar, targets(i_target, :),...
        pulse, t0, fs, tRadar(1));
    sar_dict(:, i_target) = raw_data(:);
%     raw_data_matrices = [raw_data_matrices, raw_data];
    sum_raw = sum_raw + raw_data;
%     I_p = backProject(raw_data, xRadar, yRadar, zRadar,...
%         xRadar, yRadar, zRadar, g_x, g_y, g_z,...
%         fs, tRadar(1), 10); %Back projection
% %     figure;imagesc(raw_data(1:7500,:)-min(min(raw_data))); set(gca,'YDir','normal');title('Raw Data');
%     colormap(gca, jet)
% %     set(gca,'FontSize',18, 'FontName','Times');
% %     xlabel('Apertures', 'FontSize', 28, 'FontName','Times');
% %     ylabel('Time', 'FontSize', 28, 'FontName','Times');
% %     figure;
%     imagedB(grid_x, grid_y, I_p, -40, 0);
%     xlabel('Cross Range'); ylabel('Down Range');
end

% I_p = backProject(reshape(sar_dict(:, 5), n_samples, n_apertures), xRadar, yRadar, zRadar,...
%         xRadar, yRadar, zRadar, g_x, g_y, g_z,...
%         fs, tRadar(1), 1); %Back projection
% I_p = backProject(reshape(sum_raw, n_samples, n_apertures), xRadar, yRadar, zRadar,...
%         xRadar, yRadar, zRadar, g_x, g_y, g_z,...
%         fs, tRadar(1), 1); %Back projection    
% figure; imagedB(grid_x, grid_y, I_p, -40, 0);
% colormap(gca, jet);
% xlabel('Cross Range'); ylabel('Down Range');

figure; imagesc(sum_raw)
colormap(jet(3))

pulse = pulse_data.st;
pulse_sampling_period = pulse_data.ts;
save("sar_dict_target_distance_5.mat",...
    "sar_dict", "n_samples", "n_apertures", "pulse",  "pulse_sampling_period");

zoom_factor = 8;
fs = 1 / pulse_data.ts;
nfft = n_samples;
df = fs / nfft;
freq = df * (0:nfft-1) / 1e6;


pulse_S = fft(pulse, nfft);
plot(freq(1:floor(nfft/zoom_factor)),...
    20*log10(abs(pulse_S(1:floor(nfft/zoom_factor)))), 'go');

dict_atom = reshape(sar_dict(:, 2), n_samples, n_apertures);
dict_atom_one_aperture = dict_atom(:, 1);
dict_atom_one_aperture_S = fft(dict_atom_one_aperture, nfft);
dict_atom_one_aperture_S_dB = 20*log10(abs(dict_atom_one_aperture_S));
hold on;
plot(freq(1:floor(nfft/zoom_factor)),...
    dict_atom_one_aperture_S_dB(1:floor(nfft/zoom_factor)), 'rx')

real_data = load('Datasets/real_sar_data/C1.mat');
real_s = real_data.Data(:, 1);
real_s = [zeros(300, 1); real_s; zeros(600, 1)];
real_S = fft(real_s, nfft);
real_S_dB = 20*log10(abs(real_S));
hold on;
plot(freq(1:floor(nfft/zoom_factor)), real_S_dB(1:floor(nfft/zoom_factor)), 'b')


figure; 
% plot(pulse_data.st); title('Transmitted pulse'); hold on;
plot(1e4*real_s); hold on;
plot(dict_atom_one_aperture);

figure;
atom1 = sar_dict(:, 13);
atom2 = sar_dict(:, 14);
sum = atom1 + atom2;
sum = reshape(sum, n_samples, n_apertures);
plot(sum(:, 100))
