clc; close all; clear;

addpath('/home/dung/Development/sargan_v4/BoomSARSimulation_light/utils');
pulse_data = load('/home/dung/Development/sargan_v4/BoomSARSimulation_light/Datasets/TxPulse.mat');
load('/home/dung/Development/sargan_v4/BoomSARSimulation_light/para.mat');
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
target_x_distance = 20;

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
a = 10*10.^(K*rand(size(box_grid_x))); %amplitudes
% 
dict_targets = [box_grid_x(:) box_grid_y(:) box_grid_z(:) a(:)];
% targets = [grid_x(251) grid_y(251) 0 a(1)];
n_dict_targets = size(dict_targets, 1);

s = pulse_data.st;
n_samples = 1500;%16000;
pulse = [s; zeros(n_samples-length(s), 1)];
size(s);

fs = 8*10^9; %Sampling rate = 8GHz
[m, I] = max(pulse);
t0 = (I-1)/fs;

% Raw time-domain data
radar_half_num_locations = 150;
radar_location_indices = radar_center_location_idx - radar_half_num_locations : ...
    radar_center_location_idx + radar_half_num_locations;
%radar_location_indices = 1:length(xRadar);
xRadar = xRadar(radar_location_indices);
yRadar = yRadar(radar_location_indices);
zRadar = zRadar(radar_location_indices);
n_apertures = length(xRadar);

DATA_PATH = "/data/dung/sargan"
scene_type = "clustered";
dict_type = num2str(target_x_distance);
data_output_path = strcat(DATA_PATH, "/radarconf19_v4/outputs");
scene_matfile_path = strcat(data_output_path, "/", scene_type, "_dict_", dict_type, "_scene_rec")
scene_rec_data = load(scene_matfile_path + ".mat");

I_corrupted_cell = {};
I_sargan_cell = {};
I_omp_cell = {};
for i = 1 : length(scene_rec_data.missing_rates)
    corrupted = squeeze(scene_rec_data.corrupted(i, :, :));
    sargan_rec = squeeze(scene_rec_data.sargan_rec(i, :, :));
    omp_rec = squeeze(scene_rec_data.omp_rec(i, :, :));
    I_corrupted_cell{end+1} = backProject(corrupted, xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, g_x, g_y, g_z, fs, tRadar(1), 10); %Back projection
    I_sargan_cell{end+1} = backProject(sargan_rec, xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, g_x, g_y, g_z, fs, tRadar(1), 10); %Back projection
    I_omp_cell{end+1} = backProject(omp_rec, xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, g_x, g_y, g_z, fs, tRadar(1), 10); %Back projection
end

original = scene_rec_data.original;
corrupted = scene_rec_data.corrupted;
sargan_rec = scene_rec_data.sargan_rec;
omp_rec = scene_rec_data.omp_rec;
missing_rates = scene_rec_data.missing_rates;
sargan_out_snr = scene_rec_data.sargan_out_snr;
omp_out_snr = scene_rec_data.omp_out_snr;
sargan_snr_gain = scene_rec_data.sargan_snr_gain;
omp_snr_gain = scene_rec_data.omp_snr_gain;
I_original = scene_rec_data.I_original;
I_corrupted =  permute(cat(3, I_corrupted_cell{:}), [3, 1, 2]);
I_sargan = permute(cat(3, I_sargan_cell{:}), [3, 1, 2]);
I_omp =  permute(cat(3, I_omp_cell{:}), [3, 1, 2]);


image_matfile_path = scene_matfile_path + "_img";
save(image_matfile_path + ".mat", "original", "I_original",...
    "I_corrupted", "I_sargan", "I_omp",...
    "corrupted", "sargan_rec", "omp_rec", "missing_rates",...
    "sargan_out_snr", "omp_out_snr", "sargan_snr_gain", "omp_snr_gain");
%{
db_range = -50;
figure;
subplot(221); imagedB(grid_x, grid_y, I_corrupted, db_range, 0);
title("Corrupted"); xlabel('Cross Range'); ylabel('Down Range');
colormap(gca, jet)

subplot(222); imagedB(grid_x, grid_y, I_sargan, db_range, 0);
title("Recovered by SARGAN"); xlabel('Cross Range'); ylabel('Down Range');
colormap(gca, jet)

subplot(223); imagedB(grid_x, grid_y, I_omp, db_range, 0);
title("Recovered by OMP"); xlabel('Cross Range'); ylabel('Down Range');
colormap(gca, jet)
%}
exit;


