function [shifted_pulse] = pulseshift(pulse, fs, td)
%PULSEDEL 

%format long;

N = 2*length(pulse);
pulse_fft = fft(pulse, N);



m = (0:N-1)'*(-2*pi*i*fs*td/N);
p_f = bsxfun(@times, pulse_fft, exp(m));
shifted_pulse = real(ifft(p_f, 'symmetric'));

shifted_pulse(1+N/2:end,:) = [];

end

