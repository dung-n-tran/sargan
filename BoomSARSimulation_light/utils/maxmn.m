function [maxValues, maxIndices] = maxmn(A, N)
%MAXMN : Returns max 'N' values of array A 

[maxValues, maxIndices] = sort(A(:), 'descend');

maxValues = maxValues(1:N);
maxIndices = maxIndices(1:N);
end