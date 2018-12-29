function [pulse] = pulsegen(num_samples, fs, delta_t)
%PULSEGEN : p(t) = cos(2*pi*f*(t-delta_t))exp(-(alpha*f*(t-delta_t))^2).
%
% num_samples : Length of the pulse
% fs          : Sampling rate (e.g. 8GHz)
% delta_t     : Pulse delay (e.g. 8*10^-9s). It can be a single valuse in which
%           the output is a single delayed pulse, or it can be an array of 
%           in which case the output is an array of delayed pulses
%
%


alpha = 2;
f = 800*10^6; %Pulse frequency (e.g. 800MHz)


t = 0:num_samples-1;
delta_t = delta_t*fs;

n = repmat(t', 1, length(delta_t))-repmat(delta_t, length(t), 1);

pulse = cos(2*pi*f/fs*n).*exp(-(alpha*f/fs*n).^2);

end
