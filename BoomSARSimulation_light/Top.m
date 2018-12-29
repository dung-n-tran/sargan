function [] = Top()

%%
close all;
tic
addpath('utils');
pulse_data = load('DataSets/TxPulse.mat');
load('para.mat');

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

% 10 random point targets with various reflectivity
num_targets = 2;
p_target  = genTargets1(num_targets, [grid_x(151) grid_x(350)], [grid_y(251) grid_y(500)]);

num_samples = 8000;%16000;

s = pulse_data.st;
pulse = [s; zeros(num_samples-length(s), 1)];
size(s);

fs = 8*10^9; %Sampling rate = 8GHz
[m, I] = max(pulse);
t0 = (I-1)/fs;

% Raw time-domain data
y = generateTDRawData(xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, p_target, pulse, t0, fs, tRadar(1)); %Raw data generation

%figure;imagesc(mat2gray(y)); set(gca,'YDir','normal');% title('Raw Data');
% figure;imagesc(y(1:7500,:)-min(min(y))); set(gca,'YDir','normal');title('Raw Data');
figure; imagesc(y - min(min(y))); set(gca, 'YDir', 'normal'); title('Raw Data');
colormap(gca, jet)

%colormap(gray);  
set(gca,'FontSize',18, 'FontName','Times');
 xlabel('Apertures', 'FontSize', 28, 'FontName','Times');
 ylabel('Time', 'FontSize', 28, 'FontName','Times');

% image formation
I_p = backProject(y, xRadar, yRadar, zRadar, xRadar, yRadar, zRadar, g_x, g_y, g_z, fs, tRadar(1), 10); %Back projection
figure;imagedB(grid_x, grid_y, I_p, -40, 0);
xlabel('Cross Range'); ylabel('Down Range');
colormap(gca, bone)

running_time = toc;

fprintf('Runing time:'); disp(running_time);
end