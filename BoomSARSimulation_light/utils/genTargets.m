function [targets] = genTargets(N, grid_x, grid_y)
%GENTARGETS : Generates N target cordinates within grid given by grid_x and
%grid_y. Amplitude of targets go from 0 to K*10dB

K = 0.8;
x = grid_x(randperm(length(grid_x)));
y = grid_y(randperm(length(grid_y)));

a =  10.^(K*rand(1, N)); %amplitude

targets = [(x(1:N))' (y(1:N))' zeros(N, 1) a'];

end

