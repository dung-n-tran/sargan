function [A, normA] = normalize(Ain)
%NORMALIZE : Normalizes columns of input matrix

%normA = sqrt(diag(Ain'*Ain));
normA = sqrt(sum(Ain.^2, 1));
normA(find(normA==0)) = 1; %to avoid divide by zero

%A = Ain*diag(1./normA);
A = bsxfun(@rdivide, Ain, normA); 

end

