function [X] = omp_CV(A, Y)
%OMP : Orthogonal Matching Pursuit with cross validation

%Inputs
%    Y : Measured vector
%    A : Sensing matrix
%    x : The sparse output

%Parameters
alpha = 0.9;
L = round(0.5*length(Y)); %Number of measurements in the estimation set
Ie = randperm(L);
Ie(L+1:end) = [];
Icv = setdiff([1:length(Y)], Ie);

%Estimation set
Ae = (A(Ie, :));
Ye = (Y(Ie));

%Cross Validation set
Acv = (A(Icv, :));
Ycv = (Y(Icv));

%Initialization
m = alpha*max(Ae'*Ye)
X = omp(Ae, Ye, m);
m_new = max(Acv'*(Ycv-Acv*X));

while(m_new<m)
    m = m_new
    X = omp(Ae, Ye, m);
    m_new = max(Acv'*(Ycv-Acv*X));
end


end

