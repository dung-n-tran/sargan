function [s] = generateRawDataForAp(tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, p_target, pulse, t0, fs, tAcq)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GENERATERAWDATAFORAP : SAR image formation (Raw Image) for single aperture
%point and multiple targets. The shifted pulses for targets are not added
%together. The output is num_samples x num_targets
%
%Input parameters
%    Tx_range    : Transmitter cordinate (single aperture)
%    Rx_range    : Receiver cordinate
%    p_target    : Cordinates of the targets Cordinates of the targets 
%                  (It can be an array of cordinates, 
%                   each row corresponding to a target)
%    pulse       : Input pulse
%    t0          : Center of the pulse in seconds (e.g. t0 = 20*10^-9)
%    fs          : Sampling rate
%    num_samples : Number of samples in the pulse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_tgt     = size(p_target, 1); %Number of targets
num_samples = length(pulse);

c = 3*10^8; %Velocity of the signal in m/s
pulse_fft = fft(pulse, 2*num_samples); %Required for calculating pulse shift

s = zeros(num_samples, num_tgt); %output raw image

%distance from Tx and Rx to targets
r = sqrt((tx_x-p_target(:, 1)).^2+(tx_y-p_target(:, 2)).^2+(tx_z-p_target(:, 3)).^2)+...
    sqrt((rx_x-p_target(:, 1)).^2+(rx_y-p_target(:, 2)).^2+(rx_z-p_target(:, 3)).^2);

t = r/c;

s = pulseshift(t'-tAcq-t0)*diag(p_target(:, 4));


    function [shifted_pulse] = pulseshift(td)

    N = size(pulse_fft, 1);

    m = (0:N-1)'*(-2*pi*i*fs*td/N);
    p_f = bsxfun(@times, pulse_fft, exp(m));
    shifted_pulse = real(ifft(p_f, 'symmetric'));

    shifted_pulse(1+num_samples:end,:) = [];
    end

end

