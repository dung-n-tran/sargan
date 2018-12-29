function [ v ] = var(x)

x = abs(x);
l = length(x);
m = mean(x);
x = x./m;

v = norm(x-1);


end

