function [x] = omp(A, y, threshold)
%OMP : Orthogonal Matching Pursuit

%Inputs
%    y : Measured vector
%    A : Sensing matrix
%    x : The sparse output

%Initialization
yr = y; %residue
x = zeros(size(A, 2), 1);

n = 0;
[m, I] = max(abs(A' * yr));
m
while m > threshold && n < 5
%while norm(yr)>=0.1
%while(length(si)<=10)
    
    x(I) = m;
    yr = yr - (m*A(:, I));
    
    A(:, I) =  [];
    [m, I] = max(abs(A' * yr));
    m
    
    n = n+1;
end

x(find(x ~=0))
end

