function results = sdm(y,x,W,info)
% PURPOSE: computes spatial durbin model estimates
%         (I-rho*W)y = a + X*B1 + W*X*B2 + e, using sparse algorithms
% ---------------------------------------------------
%  USAGE: results = sdm(y,x,W,info)
%  where: y = dependent variable vector
%         x = independent variables matrix
%         W = contiguity matrix (standardized)
%       info = an (optional) structure variable with input options:
%       info.rmin = (optional) minimum value of rho to use in search  
%       info.rmax = (optional) maximum value of rho to use in search    
%       info.convg = (optional) convergence criterion (default = 1e-8)
%       info.maxit = (optional) maximum # of iterations (default = 500)
%       info.lflag = 0 for full lndet computation (default = 1, fastest)
%                  = 1 for MC lndet approximation (fast for very large problems)
%                  = 2 for Spline lndet approximation (medium speed)
%       info.order = order to use with info.lflag = 1 option (default = 50)
%       info.iter  = iterations to use with info.lflag = 1 option (default = 30) 
%       info.lndet = a matrix returned by sar, sar_g, sarp_g, etc.
%                    containing log-determinant information to save time
% ---------------------------------------------------
%  RETURNS: a structure
%         results.meth  = 'sdm'
%         results.beta  = bhat [a B1 B2]' a k+(k-1) x 1 vector
%         results.rho   = rho 
%         results.tstat = t-statistics (last entry is rho)
%         results.yhat  = yhat
%         results.resid = residuals
%         results.sige  = sige
%         results.rsqr  = rsquared
%         results.rbar  = rbar-squared
%         results.lik   = log likelihood
%         results.nobs  = nobs
%         results.nvar  = nvars (includes lam)
%         results.y     = y data vector
%         results.iter  = # of iterations taken
%         results.rmax  = 1/max eigenvalue of W (or rmax if input)
%         results.rmin  = 1/min eigenvalue of W (or rmin if input)
%         results.lflag = lflag from input
%         results.miter = info.iter option from input
%         results.order = info.order option from input
%         results.limit = matrix of [rho lower95,logdet approx, upper95] intervals
%                         for the case of lflag = 1
%         results.time1 = time for log determinant calcluation
%         results.time2 = time for eigenvalue calculation
%         results.time3 = time for hessian or information matrix calculation
%         results.time4 = time for optimization
%         results.time  = total time taken       
%         results.lndet = a matrix containing log-determinant information
%                          (for use in later function calls to save time)
%  --------------------------------------------------
%  SEE ALSO: prt(results)
% ---------------------------------------------------
%  NOTES: constant term should be in 1st column of the x-matrix
%         constant is excluded from B2 estimates
%  if you use lflag = 1 or 2, info.rmin will be set = -1 
%                             info.rmax will be set = 1
%  For n < 1000 you should use lflag = 0 to get exact results  
% ---------------------------------------------------
% REFERENCES: R. Kelley Pace and R. Barry, 
% ``Simulating mixed regressive spatial autoegressive estimators'', 
% Computational Statistics, 1998 Vol. 13, pp. 397-418.
% For lndet information see: Ronald Barry and R. Kelley Pace, 
% "A Monte Carlo Estimator of the Log Determinant of Large Sparse Matrices", 
% Linear Algebra and its Applications", Volume 289, Number 1-3, 1999, pp. 41-54.
% ---------------------------------------------------

% written by:
% James P. LeSage, 4/2002
% Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com

% NOTE: much of the speed for large problems comes from:
% the use of methods pioneered by Pace and Barry.
% R. Kelley Pace was kind enough to provide functions
% lndetmc, and lndetint from his spatial statistics toolbox
% for which I'm very grateful.


time1 = 0; 
time2 = 0;
time3 = 0;
time4 = 0;

timet = clock; % start the clock for overall timing

% if we have no options, invoke defaults
if nargin == 3
    info.lflag = 1;
end;

% parse input options
[rmin,rmax,convg,maxit,detval,ldetflag,eflag,order,miter,options] = sdm_parse(info);

results.miter = miter;
results.order = order;
results.lflag = ldetflag;

% check size of user inputs for comformability
[n nvar] = size(x); 
[n1 n2] = size(W);
if n1 ~= n2
error('sdm: wrong size weight matrix W');
elseif n1 ~= n
error('sdm: wrong size weight matrix W');
end;
[nchk junk] = size(y);
if nchk ~= n
error('sdm: wrong size y vector input');
end;

results.meth = 'sdm';

results.y = y;
results.nobs = n;
results.nvar = nvar; 

% compute eigenvalues or limits
[rmin,rmax,time2] = sdm_eigs(eflag,W,rmin,rmax,n);

% do log-det calculations
[detval,time1] = sdm_lndet(ldetflag,W,rmin,rmax,detval,order,miter);

results.rmax = rmax;      
results.rmin = rmin;
results.lflag = ldetflag;

t0 = clock;

options = optimset('fminbnd');
[rho,like,exitflag,output] = fminbnd('f_sdm',rmin,rmax,options,y,x,W,detval);
   
time4 = etime(clock,t0);
results.lik = -like;

if exitflag == 0
fprintf(1,'sdm: convergence not obtained in %4d iterations \n',options(14));
end;
results.iter = output.iterations;

% find beta hats
rho2 = rho*rho;
dy=W*y;
xdx=[ x(:,2:nvar) W*x(:,2:nvar) ones(n,1)];
xdxtxdx=(xdx'*xdx);
xdxinv=inv(xdxtxdx);
xdxy=xdx'*y;
xdxdy=xdx'*dy;
bmat=xdxtxdx\[xdxy xdxdy];       
bols=bmat(:,1);
bolsd=bmat(:,2);
beta = bols - rho*bolsd;
results.yhat = xdx*beta + rho*sparse(W)*y;
eo=y-xdx*bols;
ed=dy-xdx*bolsd;
e2o=(eo'*eo);
edo=(ed'*eo);
e2d=(ed'*ed);
epe = (e2o-2*rho*edo+rho2*e2d);
sige = epe/n;

% compute results 
results.rho = rho;
results.resid = y - results.yhat;
results.sige = sige;

% use numerical hessian
t0 = clock;

parm = [beta
        rho
        sige];
nb = length(parm);
% t-stats using numerical hessian
hessn = hessian('f2_sdm',parm,y,x,W,detval);
time3 = etime(clock,t0);

xpxi = invpd(hessn); 
xpxi = abs(diag(xpxi(1:nb-1,1:nb-1)));
tmp = [beta
       rho];
nb = length(beta);
ttmp = tmp./sqrt(xpxi);
bhat = zeros(nb,1);
bhat(1,1) = beta(nb,1);
bhat(2:nb,1) = beta(1:nb-1,1);
tstat = zeros(nb+1,1);
tstat(1,1) = ttmp(nb,1);
tstat(2:nb,1) = ttmp(1:nb-1,1);
tstat(nb+1,1) = ttmp(nb+1,1);
results.beta= bhat;
results.tstat = tstat;

ym = y - mean(y);
rsqr1 = epe;
rsqr2 = ym'*ym;
results.rsqr = 1.0 - rsqr1/rsqr2; % r-squared
rsqr1 = rsqr1/(n-2*nvar+1);
rsqr2 = rsqr2/(n-1.0);
results.rbar = 1 - (rsqr1/rsqr2); % rbar-squared

results.time = etime(clock,timet);
results.time1 = time1;
results.time2 = time2;
results.time3 = time3;
results.time4 = time4;
results.lndet = detval;



function [rmin,rmax,convg,maxit,detval,ldetflag,eflag,order,iter,options] = sdm_parse(info)
% PURPOSE: parses input arguments for far, far_g models
% ---------------------------------------------------
%  USAGE: [rmin,rmax,convg,maxit,detval,ldetflag,eflag,order,iter] = far_parse(info)
% where info contains the structure variable with inputs 
% and the outputs are either user-inputs or default values
% ---------------------------------------------------

% set defaults
options = zeros(1,18); % optimization options for fminbnd
options(1) = 0; 
options(2) = 1.e-6; 
options(14) = 500;

eflag = 1;     % default to not computing eigenvalues
ldetflag = 1;  % default to 1999 Pace and Barry MC determinant approx
order = 50;    % there are parameters used by the MC det approx
iter = 30;     % defaults based on Pace and Barry recommendation
rmin = -1;     % use -1,1 rho interval as default
rmax = 1;
detval = 0;    % just a flag
convg = 0.0001;
maxit = 500;

fields = fieldnames(info);
nf = length(fields);
if nf > 0
    
 for i=1:nf
    if strcmp(fields{i},'rmin')
        rmin = info.rmin;  eflag = 1;
    elseif strcmp(fields{i},'rmax')
        rmax = info.rmax;  eflag = 1;
    elseif strcmp(fields{i},'convg')
        options(2) = info.convg;
    elseif strcmp(fields{i},'maxit')
        options(14) = info.maxit;  
    elseif strcmp(fields{i},'lndet')
    detval = info.lndet;
    ldetflag = -1;
    eflag = 1;
    rmin = detval(1,1);
    nr = length(detval);
    rmax = detval(nr,1);
    elseif strcmp(fields{i},'lflag')
        tst = info.lflag;
        if tst == 0,
        ldetflag = 0; eflag = 0; % compute eigenvalues
        elseif tst == 1,
        ldetflag = 1; eflag = 1; % reset this from default
        elseif tst == 2,
        ldetflag = 2; eflag = 1; % reset this from default
        else
        error('sdm: unrecognizable lflag value on input');
        end;
    elseif strcmp(fields{i},'order')
        order = info.order;  
    elseif strcmp(fields{i},'iter')
        iter = info.iter; 
    end;
 end;
 
else, % the user has input a blank info structure
      % so we use the defaults
end; 

function [rmin,rmax,time2] = sdm_eigs(eflag,W,rmin,rmax,n);
% PURPOSE: compute the eigenvalues for the weight matrix
% ---------------------------------------------------
%  USAGE: [rmin,rmax,time2] = far_eigs(eflag,W,rmin,rmax,W)
% where eflag is an input flag, W is the weight matrix
%       rmin,rmax may be used as default outputs
% and the outputs are either user-inputs or default values
% ---------------------------------------------------


if eflag == 0
t0 = clock;
opt.tol = 1e-3; opt.disp = 0;
lambda = eigs(sparse(W),speye(n),1,'SR',opt);  
rmin = 1/lambda;   
rmax = 1;
time2 = etime(clock,t0);
else
time2 = 0;
end;


function [detval,time1] = sdm_lndet(ldetflag,W,rmin,rmax,detval,order,iter);
% PURPOSE: compute the log determinant |I_n - rho*W|
% using the user-selected (or default) method
% ---------------------------------------------------
%  USAGE: detval = far_lndet(lflag,W,rmin,rmax)
% where eflag,rmin,rmax,W contains input flags 
% and the outputs are either user-inputs or default values
% ---------------------------------------------------


% do lndet approximation calculations if needed
if ldetflag == 0 % no approximation
t0 = clock;    
out = lndetfull(W,rmin,rmax);
time1 = etime(clock,t0);
tt=rmin:.001:rmax; % interpolate a finer grid
outi = interp1(out.rho,out.lndet,tt','spline');
detval = [tt' outi];
    
elseif ldetflag == 1 % use Pace and Barry, 1999 MC approximation

t0 = clock;    
out = lndetmc(order,iter,W,rmin,rmax);
time1 = etime(clock,t0);
results.limit = [out.rho out.lo95 out.lndet out.up95];
tt=rmin:.001:rmax; % interpolate a finer grid
outi = interp1(out.rho,out.lndet,tt','spline');
detval = [tt' outi];

elseif ldetflag == 2 % use Pace and Barry, 1998 spline interpolation

t0 = clock;
out = lndetint(W,rmin,rmax);
time1 = etime(clock,t0);
tt=rmin:.001:rmax; % interpolate a finer grid
outi = interp1(out.rho,out.lndet,tt','spline');
detval = [tt' outi];

elseif ldetflag == -1 % the user fed down a detval matrix
    time1 = 0;
        % check to see if this is right
        if detval == 0
            error('sdm: wrgon lndet input argument');
        end;
        [n1,n2] = size(detval);
        if n2 ~= 2
            error('sdm: wrong sized lndet input argument');
        elseif n1 == 1
            error('sdm: wrong sized lndet input argument');
        end;          
end;

