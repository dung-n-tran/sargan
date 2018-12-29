function [maxValues, maxIndices] = maxn(A, N)
%MAXN : Returns max 'N' values of vector A 

[maxValues, maxIndices] = sort(A, 'descend');

maxValues = maxValues(1:N);
maxIndices = maxIndices(1:N);

end

