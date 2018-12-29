function [x] = omp(A, y, S)
%OMP : Orthogonal Matching Pursuit

%Inputs
%    y : Measured vector
%    A : Sensing matrix
%    x : The sparse output

%Initialization
yr = y; %residue
si = []; %sparse index set


%Note: I am using the support 's' size as the stopping criterian,
%as we know it already
while length(si) < S
%while norm(yr) > 10^-5
    [m, I] = max(abs(A' * yr));
    si = [si I];
     
    yr = y - (A(:, si) * (A(:, si)\y));
end

si = unique(si);
x = zeros(size(A,2), 1);

x(si) = A(:,si)\y;
end

