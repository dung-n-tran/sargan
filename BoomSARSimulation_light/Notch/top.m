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


%% ----Notched pulse

pulse_fft = fft(pulse);
N = length(pulse_fft);
p_fft = 10*log10(abs(pulse_fft(1:NumSamps/2)));
figure;plot(0:fs/N:fs*(NumSamps/2-1)/N, p_fft); title('FFT of the pulse');
ylim([-50 10]);

passband = find(p_fft>=-3); %passband
bw =  length(passband);

%Random band
pp = 0.75;
id = passband(randperm(bw, round(pp*bw)));

%Continuous band
% pp = 0.25;
% [m mi] = max(p_fft);
% id = mi-round(bw*pp/2):mi+round(bw*pp/2);

pulse_fft(id) = 0;
pulse_fft(NumSamps-id+2) = 0;

notched_pulse = ifft(pulse_fft, 'symmetric');
p_fft = pulse_fft(1:NumSamps/2);
idx = find(p_fft==0);
p_fft(idx) = 10^-16;
figure;plot(0:fs/N/1e6:fs*(NumSamps/2-1)/N/1e6, 10*log10(abs(p_fft)), 'LineWidth',1);
title('FFT of the notched pulse');
ylim([-35 7]);
xlim([0 3e3]);
xlabel('Frequency (MHz)' , 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude (dB)' , 'FontSize', 12, 'FontWeight', 'bold');


d_fft = fft(data);
d_fft(id, :) = 0;
d_fft(NumSamps-id+2, :) = 0;

data_n = (ifft(d_fft, 'symmetric'));

%% -- Reduce the data size

NumSamps = 5200; % data is zero after 5200 samples. So this doesn't affect quality
ds = 4; % we use only every ds'th data record

data = data(1:2:NumSamps, 1:ds:end);
data_n = data_n(1:2:NumSamps, 1:ds:end);
xRadar = xRadar(1:ds:end);
yRadar = yRadar(1:ds:end);
zRadar = zRadar(1:ds:end);
fs = fs/2;
pulse = pulse(1:2:NumSamps);
notched_pulse = notched_pulse(1:2:NumSamps);

%% 

I_p = backProject(data, xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, g_x, g_y, g_z, fs, tAcq, oversample);
figure, imagedB(grid_x, grid_y, I_p, -40, 0); title('Original Image');
xlabel('Cross Range');ylabel('Down Range');

I_p = backProject(data_n, xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, g_x, g_y, g_z, fs, tAcq, oversample);
figure, imagedB(grid_x, grid_y, I_p, -40, 0);
xlabel('Cross Range');ylabel('Down Range');
title('Image with missing bands');


%% ---Recover RawPatch ---
rgrid_x = grid_x(1:1:end);
rgrid_y = grid_y(1:1:end-580);
[rg_x, rg_y] = meshgrid(rgrid_x, rgrid_y);
rg_z = zeros(size(rg_x));

K = 10; %Number of apertures processed at a time
Noverlap = 0; %Number of apertures that overlap
sparsity = 500;
tic;
data_r =  SparseImagingPfd2D_RawPatch(data_n, pulse, notched_pulse, t0, t0, fs, tAcq, ...
    xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, rg_x, rg_y, rg_z, sparsity, K, Noverlap);
toc;

I_p_out = backProject(data_r, xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, g_x, g_y, g_z, fs, tAcq, oversample);
figure;imagedB(grid_x, grid_y, I_p_out, -40, 0);
  
disp(['SNR : ', num2str(20*log10(norm(data(:))/norm(data(:)-data_r(:))))]);


end