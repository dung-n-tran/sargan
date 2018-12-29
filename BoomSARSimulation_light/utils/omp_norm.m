function [x] = omp(A, y, epsilon)
%OMP : Orthogonal Matching Pursuit

%Inputs
%    y : Measured vector
%    A : Sensing matrix
%    x : The sparse output

%Initialization
yr = y; %residue
si = []; %sparse index set
As = [];
x = zeros(size(A, 2), 1);


[m, I] = max(abs(A' * yr));

while norm(yr)>=epsilon*norm(y)
    si = [si I];
    As = [As A(:, I)];
    
    x(si) = As \ y;
    yr = y - (As * x(si));
    [m, I] = max(abs(A' * yr));
end


end

