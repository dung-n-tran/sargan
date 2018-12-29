function [Y_hat] = SparseImagingPfd2D(S, s0, s0_notch, t0, t0_notch, fs, tAcq, ...
    tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, grid_x, grid_y, grid_z, threshold, K, Noverlap)
%SparseImagingPfd2D_RawPatch : Processes K apertures at a time, instead of the entire data
%
%Input parameters
%    S           : Input raw data(NSamples, NApertures)
%    s0          : Impulse waveform
%    s0_notch    : Impulse waveform with missing frequency bands
%    t0          : Centre of s0 in seconds (e.g. 1*10^-9)
%    fs          : Sampling rate
%    tAcq        : Acquisition delay in seconds
%    xR, yR, zR  : Radar path(NApertures)
%    grid_x,
%    grid_y
%    grid_z      : The imaging grid(NR, NXR)
%    threshold   : Threshold parameter for OMP
%    K           : Number of apertures processed at a time
%    Noverlap    : Number of apertures overlapping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[NR, NXR] = size(grid_x);
[NumSamples NumAps] = size(S);
NumGrid = NR*NXR; %Number of grid points

Y_hat = zeros(NumSamples, NumAps); %D*S
scale_factor = zeros(1, NumAps);
grid_x = grid_x(:);
grid_y = grid_y(:);
grid_z = grid_z(:);

i = 1;
while 1
    j = min(i+K-1, NumAps);
    display(['Apertures ' num2str(i) ' to ' num2str(j)]);

    Y = S(:, i:j);
    D = single(zeros(size(Y, 2)*NumSamples, NumGrid));
    makeDictionary_1(tx_x(i:j), tx_y(i:j), tx_z(i:j), rx_x(i:j), rx_y(i:j), rx_z(i:j));
    X = omp_S(D, Y(:), threshold);
    clear D;
    
    Idx = find(X~=0);
    p_tgt = [grid_x(Idx) grid_y(Idx) grid_z(Idx) X(Idx)];
    Y_r = generateTDRawData(tx_x(i:j), tx_y(i:j), tx_z(i:j), ...
        rx_x(i:j), rx_y(i:j), rx_z(i:j), p_tgt, s0, t0, fs, tAcq);
    
    Y_hat(:, i:j) = Y_hat(:, i:j)+Y_r;
    scale_factor(:, i:j) = scale_factor(:, i:j) + ones(1, j-i+1);
    
    i = j+1-Noverlap;
    if j==NumAps
        scale_factor(find(scale_factor==0)) = 1;
        Y_hat = Y_hat*(diag(1./scale_factor));
        break;
    end
end


    function [] = makeDictionary(tx_x, tx_y, tx_z, rx_x, rx_y, rx_z)
        %MAKEDICTIONARY : Makes dictionary for denoising SAR images (the parabolas). An
        % atom in the dictionary is the raw data corresponding to a point
        % on the grid.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        for n = 1:NumGrid
            p_target = [grid_x(n) grid_y(n) grid_z(n) 1];
            data = single(generateTDRawData(tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, p_target, s0_notch, t0_notch, fs, tAcq));
                  
            D(:, n) = data(:);
        end
    end

    function[] = makeDictionary_1(tx_x, tx_y, tx_z, rx_x, rx_y, rx_z)
        %MAKEDICTIONARY : Makes dictionary for denoising SAR images (the parabolas). An
        % atom in the dictionary is the raw data corresponding to a point
        % on the grid.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        c = 3*10^8;
        
        ref_dat = generateTDRawData(tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, [grid_x(1) grid_y(1) grid_z(1) 1], s0_notch, t0_notch, fs, tAcq);
        dat_fft = fft(ref_dat);
        
        D(:, 1) = single(ref_dat(:));
        ref_t = (sqrt((tx_x-grid_x(1)).^2+(tx_y-grid_y(1)).^2+(tx_z-grid_z(1)).^2)+...
            sqrt((rx_x-grid_x(1)).^2+(rx_y-grid_y(1)).^2+(rx_z-grid_z(1)).^2))/c;
                
        for n = 2:NumGrid
            td = (sqrt((tx_x-grid_x(n)).^2+(tx_y-grid_y(n)).^2+(tx_z-grid_z(n)).^2)+...
                sqrt((rx_x-grid_x(n)).^2+(rx_y-grid_y(n)).^2+(rx_z-grid_z(n)).^2))/c;
            
            data = single(shiftSig(dat_fft, (td-ref_t)', fs));
            D(:, n) = data(:);
        end
    end

end

function [sig_out] = shiftSig(sig_fft, td, fs)
%SHIFTSIG
N = size(sig_fft, 1);

%sfft = fft(sig, N);
%t = (0:N-1)';
exp1 = exp((0:N-1)'*(-2*pi*i*fs*td/(N)));
sfft = sig_fft.*exp1;

sig_out = real(ifft(sfft, 'symmetric'));
end
