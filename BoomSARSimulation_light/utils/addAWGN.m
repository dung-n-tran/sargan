
 function [yn] = addAWGN(y, SNR)
    %Adds AWGN to columns of y, so that each column has the given SNR(dB)

    scale_factor = sqrt(10^(-SNR/10));
 
    noise = normalize(randn(size(y)))*scale_factor;
    yn = normalize(y) + noise;
    
end