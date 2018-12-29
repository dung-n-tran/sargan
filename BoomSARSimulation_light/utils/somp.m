function [x] = somp(A, y, N, threshold)
%SOMP : Simultaneous Orthogonal Matching Pursuit

%Inputs
%    y : Measured vector
%    A : Sensing matrix
%    x : The sparse output

%Initialization
yr = y; %residue
si = []; %sparse index set
x = zeros(size(A, 2), 1);


[m, I] = max(abs(A' * yr));

while m > threshold
    si = [si I];
    for i=1:N:length(y)
        As = A(i:i+N-1, si);
        yr(i:i+N-1) = y(i:i+N-1) - (As * (As \ y(i:i+N-1)));
    end
    [m, I] = max(abs(A' * yr));
end

x(si) = A(:, si) \ y;

end

