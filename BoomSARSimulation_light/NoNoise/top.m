function [] = top()


addpath('../utils');

%% -------------------------------
load('../DataSets/TxPulse.mat');
load('../DataSets/SAR_raw_data1.mat'); %Replace this

[NumSamps NumAps] = size(data);
pulse = [s; zeros(NumSamps-length(s), 1)];
[s I] = max(pulse);
t0 = (I-1)/fs;
tAcq = tRadar(1);

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

oversample = 2;
%% -- Reduce the data size

NumSamps = 5200; % data is zero after 5200 samples. So this doesn't affect quality
ds = 4; % we use only every ds'th data record

data = data(1:2:NumSamps, 1:ds:end);
xRadar = xRadar(1:ds:end);
yRadar = yRadar(1:ds:end);
zRadar = zRadar(1:ds:end);
fs = fs/2;
pulse = pulse(1:2:NumSamps);


%% ---Generate raw data

I_p = backProject(data, xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, g_x, g_y, g_z, fs, tAcq, oversample);
figure, imagedB(grid_x, grid_y, I_p, -40, 0); title('Original Image');
xlabel('Cross Range');ylabel('Down Range');


%% ---Recover RawPatch ---
rgrid_x = grid_x(1:1:end);
rgrid_y = grid_y(1:1:end-580);
[rg_x, rg_y] = meshgrid(rgrid_x, rgrid_y);
rg_z = zeros(size(rg_x));

K = 10; %Number of apertures processed at a time
Noverlap = 0; %Number of apertures that overlap
sparsity = 500;
tic;
data_r =  SparseImagingPfd2D_RawPatch(data, pulse, t0, fs, tAcq, ...
    xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, rg_x, rg_y, rg_z, sparsity, K, Noverlap);
toc;

I_p_out = backProject(data_r, xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, g_x, g_y, g_z, fs, tAcq, oversample);
figure;imagedB(grid_x, grid_y, I_p_out, -40, 0); title('Reconstructed Image');
xlabel('Cross Range');ylabel('Down Range');

disp(['SNR : ', num2str(20*log10(norm(data(:))/norm(data(:)-data_r(:))))]);


end