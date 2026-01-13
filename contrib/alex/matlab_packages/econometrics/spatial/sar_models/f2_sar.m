function llike = f2_sar(parm,y,x,W,detval)
% PURPOSE: evaluates log-likelihood -- given ML estimates
%  spatial autoregressive model using sparse matrix algorithms
% ---------------------------------------------------
%  USAGE:llike = f2_sar(parm,y,X,W,ldet)
%  where: parm = vector of maximum likelihood parameters
%                parm(1:k-2,1) = b, parm(k-1,1) = rho, parm(k,1) = sige
%         y    = dependent variable vector (n x 1)
%         X    = explanatory variables matrix (n x k)
%         W    = spatial weight matrix
%         ldet = matrix with [rho log determinant] values
%                computed in sar.m using one of Kelley Pace's routines  
% ---------------------------------------------------
%  RETURNS: a  scalar equal to minus the log-likelihood
%           function value at the ML parameters
%  --------------------------------------------------
%  NOTE: this is really two functions depending
%        on nargin = 4 or nargin = 5 (see the function)
% ---------------------------------------------------
%  SEE ALSO: sar, f2_far, f2_sac, f2_sem
% ---------------------------------------------------

% written by: James P. LeSage 1/2000
% University of Toledo
% Department of Economics
% Toledo, OH 43606
% jlesage@spatial.econometrics.com

n = length(y); 
k = length(parm);
b = parm(1:k-2,1);
rho = parm(k-1,1);
sige = parm(k,1);

if nargin == 4 % case of no lndet approximation
               % we compute the full thing using
               % sparse algorithms
spparms('tight'); 
z = speye(n) - 0.1*sparse(W);
p = colmmd(z);
z = speye(n) - rho*sparse(W);
[l,u] = lu(z(:,p));
detval = sum(log(abs(diag(u))));
eD = z*y-x*b;
tmp2 = 1/(2*sige);
epe = eD'*eD;
llike = -(n/2)*log(pi) - (n/2)*log(sige) + detval - tmp2*epe;

elseif nargin == 5 % case of an lndet approximation
                   % computed in sar.m using either 
                   % lndetmc() or lndetint()  
gsize = detval(2,1) - detval(1,1);
i1 = find(detval(:,1) <= rho + gsize);
i2 = find(detval(:,1) <= rho - gsize);
i1 = max(i1);
i2 = max(i2);
index = round((i1+i2)/2);
if isempty(index)
index = 1;
end;
detm = detval(index,2);
z = speye(n) - rho*sparse(W);
eD = z*y-x*b;
tmp2 = 1/(2*sige);
epe = eD'*eD;
llike = -(n/2)*log(pi) - (n/2)*log(sige) + detm - tmp2*epe;

else
error('f2_sar: Wrong # of input arguments');
end;

