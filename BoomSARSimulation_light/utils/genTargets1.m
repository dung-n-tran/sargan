function [targets] = genTargets(N, xrange, yrange)
%GENTARGETS : Generates N target cordinates within grid given by grid_x at 
%random locations not on grid points.
%Amplitude of targets go from 0 to K*10dB

K = 1;

x = (xrange(2)-xrange(1)).*rand(1, N)+xrange(1);
y = (yrange(2)-yrange(1)).*rand(1, N)+yrange(1);


a =  10*10.^(K*rand(1, N)); %amplitude

targets = [(x(1:N))' (y(1:N))' zeros(N, 1) a'];

end

