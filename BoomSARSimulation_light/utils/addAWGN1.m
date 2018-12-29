
 function [yn] = addAWGN1(y, SNR)
    %Adds AWGN to columns of y, so that each column has the given SNR(dB)

    scale_factor = sqrt(sum(y.^2, 1))*sqrt(10^(-SNR/10));
 
    noise = randn(size(y))*diag(scale_factor);
    yn = y + noise;
    
end