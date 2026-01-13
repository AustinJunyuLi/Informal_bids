function llike = f2_sdm(parm,y,x,W,detval)
% PURPOSE: evaluates llike for the spatial durbin model 
%          using sparse matrix algorithms
% ---------------------------------------------------
%  USAGE:llike = f2_sdm(parm,y,x,W,detm)
%  where: parm  = ML parameters (beta, rho, sige)
%          y    = dependent variable vector
%          x    = data matrix
%          W    = spatial weight matrix
%         detm =  matrix with [rho log determinant] values
%                computed in sdm.m using one of 
%                Kelley Pace's routines  
% ---------------------------------------------------
%  RETURNS: a  scalar equal to minus the log-likelihood
%           function value at the parameter rho
%  --------------------------------------------------
%  NOTE: this is really two functions depending
%        on nargin = 4 or nargin = 5 (see the function)
% --------------------------------------------------- 
%  SEE ALSO: sdm, f_far, f_sac, f_sem
% ---------------------------------------------------

% written by: James P. LeSage 4/2002
% University of Toledo
% Department of Economics
% Toledo, OH 43606
% jlesage@spatial-econometrics.com


[n k] = size(x); 
npar = length(parm);
beta = parm(1:npar-2,1);
rho = parm(npar-1,1);
sige = parm(npar,1);

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


xdx = [ x(:,2:k) W*x(:,2:k) ones(n,1)];
e = y-xdx*beta-rho*sparse(W)*y;
epe = e'*e;
tmp2 = 1/(2*sige);
llike = -(n/2)*log(pi) - (n/2)*log(sige) + detm - tmp2*epe;
