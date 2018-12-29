function [x] = omp(A, y, threshold)
%#codegen

%OMP : Orthogonal Matching Pursuit

%Inputs
%    y : Measured vector
%    A : Sensing matrix
%    x : The sparse output

%Initialization
yr = y; %residue
si = []; %sparse index set


[m, I] = max(abs(A' * yr));

while m > threshold
    si = [si I];
    
    yr = y - (A(:, si) * (A(:, si)\y));
    [m, I] = max(abs(A' * yr));
end
si = unique(si);
x = zeros(size(A,2), 1);

x(si) = A(:,si)\y;

end

