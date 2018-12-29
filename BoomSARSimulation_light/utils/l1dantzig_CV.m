function [X] = l1dantzig_CV(A, Y)

%Inputs
%    Y : Measured vector
%    A : Sensing matrix
%    x : The sparse output

%Parameters
alpha = 0.99;
L = round(0.8*length(Y)); %Number of measurements in the estimation set
Ie = randperm(length(Y));
Ie(L+1:end) = [];
Icv = setdiff([1:length(Y)], Ie);

%Estimation set
Ae = normalize(A(Ie, :));
Ye = (Y(Ie));

%Cross Validation set
Acv = normalize(A(Icv, :));
Ycv = (Y(Icv));

%Initialization
epsilon = alpha*max(Ae'*Ye);
[X, lam] = DS_homotopy_function(Ae, Ye, epsilon, 10);
epsilon_new = max(Acv'*(Ycv-Acv*X));

while(epsilon_new<epsilon)
    epsilon = epsilon_new;
    [X, lam] = DS_homotopy_function(Ae, Ye, epsilon, 10);
    epsilon_new = max(Acv'*(Ycv-Acv*X));
end

[X, lam] = DS_homotopy_function(A, Y, epsilon, 10);
%X = omp(A, Y, epsilon*1.4);

end

