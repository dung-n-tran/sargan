function [x, k] = SolveOMPF(A, y, varargin)
% SolveOMPF: Orthogonal Matching Pursuit. General version.
%            Can handle complex-valued data.
% Usage
%   [alpha, iters] = SolveOMPF(A, y, 'maxIteration', k, 'lambda', minCorr, 'isNonnegative', false);
% Inputs
%	A: Either an explicit nxN matrix, with rank(A) = min(N,n) 
%               by assumption, or a string containing the name of a 
%               function implementing an implicit matrix (see below for 
%               details on the format of the function).
%	y           observation vector
% Outputs
%	 x          sparse solution of OMP
%    k          number of iterations performed
%   minCorr     minimum correlation b/w A and residue permitted

DEBUG = 0;
STOPPING_GROUND_TRUTH = -1;
STOPPING_DUALITY_GAP = 1;
STOPPING_SPARSE_SUPPORT = 2;
STOPPING_OBJECTIVE_VALUE = 3;
STOPPING_SUBGRADIENT = 4;
STOPPING_DEFAULT = STOPPING_OBJECTIVE_VALUE;
stoppingCriterion = STOPPING_DEFAULT;

OptTol = 1e-10;
lambdaStop = 0;
maxIters = length(y);
[n,N]= size(A);

% Parameters for linsolve function
% Global variables for linsolve function
global opts opts_tr machPrec
opts.UT = true; 
opts_tr.UT = true; 
opts_tr.TRANSA = true; 
machPrec = 1e-5;

% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' must be passed in pairs.']);
end
parameterCount = length(varargin)/2;

for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'lambda'
            lambdaStop = parameterValue;
        case 'maxiteration'
            if parameterValue>maxIters
                if DEBUG>0
                    warning('Parameter maxIteration is larger than the possible value: Ignored.');
                end
            else
                maxIters = parameterValue;
            end
        case 'tolerance'
            OptTol = parameterValue;
        case 'stoppingcriterion'
            stoppingCriterion = parameterValue;
        case 'groundtruth'
            xG = parameterValue;
        case 'isnonnegative'
            isNonnegative = parameterValue;
        otherwise
            error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''.']);
    end
end

% Initialize
x = zeros(N,1);
k = 0;
R_I = [];
activeSet = [];
res = y;
normy = norm(y);
resnorm = normy;
done = 0;

while ~done && k<maxIters
    corr = A'*res;
    if isNonnegative
        [maxcorr i] = max(corr);
    else
        [maxcorr i] = max(abs(corr));
    end
    
    if maxcorr<=lambdaStop
        done = 1;
    else
        newIndex = i(1);
        
        % Update Cholesky factorization of A_I
        [R_I, done] = updateChol(R_I, n, N, A, activeSet, newIndex);
    end
    
    if ~done
        activeSet = [activeSet newIndex];
        
        % Solve for the least squares update: (A_I'*A_I)dx_I = corr_I
        dx = zeros(N,1);
        z = linsolve(R_I,corr(activeSet),opts_tr);
        dx(activeSet) = linsolve(R_I,z,opts);
        x(activeSet) = x(activeSet) + dx(activeSet);
        
        % Compute new residual
        res = y - A(:,activeSet) * x(activeSet);
        
        switch stoppingCriterion
            case STOPPING_SUBGRADIENT
                error('Subgradient is not a valid stopping criterion for OMP.');
            case STOPPING_DUALITY_GAP
                error('Duality gap is not a valid stopping criterion for OMP.');
            case STOPPING_SPARSE_SUPPORT
                error('Sparse support is not a valid stopping criterion for OMP.');
            case STOPPING_OBJECTIVE_VALUE
                resnorm = norm(res);
                
                if ((resnorm <= OptTol*normy) || ((lambdaStop > 0) && (maxcorr <= lambdaStop)))
                    done = 1;
                end
            case STOPPING_GROUND_TRUTH
                done = norm(xG-x)<OptTol;
            otherwise
                error('Undefined stopping criterion');
        end

        
        if DEBUG>0
            fprintf('Iteration %d: Adding variable %d\n', k, newIndex);
        end
        
        k = k+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [R, flag] = updateChol(R, n, N, A, activeSet, newIndex)
% updateChol: Updates the Cholesky factor R of the matrix 
% A(:,activeSet)'*A(:,activeSet) by adding A(:,newIndex)
% If the candidate column is in the span of the existing 
% active set, R is not updated, and flag is set to 1.

global opts_tr machPrec
flag = 0;

newVec = A(:,newIndex);

if isempty(activeSet),
%     R = sqrt(sum(newVec.^2));
    R = norm(newVec,2);
else
    p = linsolve(R,A(:,activeSet)'*A(:,newIndex),opts_tr); 

%     q = sum(newVec.^2) - sum(p.^2);
     q = sum(abs(newVec).^2) - sum(abs(p).^2);
    
    if (q <= machPrec) % Collinear vector
        flag = 1;
    else
        R = [R p; zeros(1, size(R,2)) sqrt(q)];
    end
end

