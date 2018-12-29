function [s] = generateTDRawData(tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, p_target, pulse, t0, fs, tAcq)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GENERATERAWDATA : SAR image formation (Raw Image)
%
%Input parameters
%    Tx_range    : Transmitter cordinates
%    Rx_range    : Receiver cordinates
%    p_target    : Cordinates of the targets Cordinates of the targets 
%                  (It can be an array of cordinates, 
%                   each row corresponding to a target)
%    pulse       : Input pulse
%    t0          : Center of the pulse in seconds (e.g. t0 = 20*10^-9)
%    fs          : Sampling rate
%    num_samples : Number of samples in the pulse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_tgt   = size(p_target, 1); %Number of targets
num_ap = size(tx_x, 1); %Number of apertures
num_samples = length(pulse);

c = 3*10^8; %Velocity of the signal in m/s
pulse_fft = fft(pulse, num_samples); %Required for calculating pulse shift

s = zeros(num_samples, num_ap); %output raw image

for k = 1:num_tgt

    %distance from Tx and Rx to targets
    r1 = sqrt((tx_x-p_target(k, 1)).^2+(tx_y-p_target(k, 2)).^2+(tx_z-p_target(k, 3)).^2);
    r2 = sqrt((rx_x-p_target(k, 1)).^2+(rx_y-p_target(k, 2)).^2+(rx_z-p_target(k, 3)).^2);

    t = (r1+r2)/c;

    %Add up the returned pulses
    %s = s+p_target(k, 4)*pulsegen(num_samples, fs, t'-tAcq);
    s = s+p_target(k, 4)*pulseshift(t'-tAcq-t0);
end
% if num_tgt~= 0
%     s = s./num_tgt;
% end
    
    function [shifted_pulse] = pulseshift(td)

    m = (0:num_samples-1)'*(-2*pi*i*fs*td/num_samples);
    p_f = bsxfun(@times, pulse_fft, exp(m));
    shifted_pulse = real(ifft(p_f, 'symmetric'));
  
    end

end

