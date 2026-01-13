function results = sdm_gc(y,x,W,ndraw,nomit,prior)
% PURPOSE: Bayesian estimates of the heteroscedastic spatial durbin model C-MEX version
%         (I-rho*W)y = a + X*B1 + W*X*B2 + e, e = N(0,sige*V), V = diag(v1,v2,...vn)
%          r/vi = ID chi(r)/r, r = Gamma(m,k)
%          a, B1, B2 = diffuse
%          1/sige = Gamma(nu,d0), 
%          rho = Uniform(rmin,rmax) 
%-------------------------------------------------------------
% USAGE: results = sdm_gc(y,x,W,ndraw,nomit,prior)
% where: y = dependent variable vector (nobs x 1)
%        x = independent variables matrix (nobs x nvar), with constant term in 1st column
%        W = 1st order contiguity matrix (standardized, row-sums = 1)
%    ndraw = # of draws
%    nomit = # of initial draws omitted for burn-in            
%    prior = a structure variable with:
%            prior.rval  = r prior hyperparameter, default=4
%            prior.novi  = 1 turns off sampling for vi, producing homoscedastic model            
%            prior.m     = informative Gamma(m,k) prior on r
%            prior.k     = (default: not used)
%            prior.nu    = informative Gamma(nu,d0) prior on sige
%            prior.d0    = default: nu=0,d0=0 (diffuse prior)
%            prior.rmin  = (optional) min rho used in sampling (default = 0)
%            prior.rmax  = (optional) max rho used in sampling (default = 1)  
%            prior.lflag = 0 for full lndet computation (default = 1, fastest)
%                        = 1 for MC approx (fast for large problems)
%                        = 2 for Spline approx (medium speed)
%            prior.order = order to use with prior.lflag = 1 option (default = 50)
%            prior.iter  = iters to use with prior.lflag = 1 option (default = 30)   
%            prior.seed  = a numerical value to seed the random number generator
%                         (default is to use the system clock which produces
%                          different results on every run)
%            prior.lndet = a matrix returned by sdm, etc.
%                          containing log-determinant information to save time
%-------------------------------------------------------------
% RETURNS:  a structure:
%          results.meth   = 'sdm_gc'
%          results.bdraw  = bhat draws (ndraw-nomit x nvar)
%          results.pdraw  = rho  draws (ndraw-nomit x 1)
%          results.sdraw  = sige draws (ndraw-nomit x 1)
%          results.vmean  = mean of vi draws (nobs x 1) 
%          results.rdraw  = r draws (ndraw-nomit x 1) (if m,k input)
%          results.bmean  = b prior means, prior.beta from input
%          results.bstd   = b prior std deviations sqrt(diag(prior.bcov))
%          results.r      = value of hyperparameter r (if input)
%          results.nobs   = # of observations
%          results.nvar   = # of variables in x-matrix
%          results.ndraw  = # of draws
%          results.nomit  = # of initial draws omitted
%          results.y      = y-vector from input (nobs x 1)
%          results.yhat   = mean of posterior predicted (nobs x 1)
%          results.nu     = nu prior parameter
%          results.d0     = d0 prior parameter
%          results.time1  = time for eigenvalue calculation
%          results.time2  = time for log determinant calcluation
%          results.time3  = time for sampling
%          results.time   = total time taken  
%          results.rmax   = 1/max eigenvalue of W (or rmax if input)
%          results.rmin   = 1/min eigenvalue of W (or rmin if input)          
%          results.tflag  = 'plevel' (default) for printing p-levels
%                         = 'tstat' for printing bogus t-statistics 
%          results.lflag  = lflag from input
%          results.iter   = prior.iter option from input
%          results.order  = prior.order option from input
%          results.limit  = matrix of [rho lower95,logdet approx, upper95] 
%                           intervals for the case of lflag = 1
%          results.lndet = a matrix containing log-determinant information
%                          (for use in later function calls to save time)
%          results.seed   = seed (if input, or zero if not)
% --------------------------------------------------------------
% NOTES: constant term should be in 1st column of the x-matrix
%        constant is excluded from B2 estimates
%  - use either improper prior.rval 
%          or informative Gamma prior.m, prior.k, not both of them
% - if you use lflag = 1 or 2, prior.rmin will be set = 0 
%                              prior.rmax will be set = 1
% - for n < 1000 you should use lflag = 0 to get exact results  
% --------------------------------------------------------------
% SEE ALSO: (sar_gcd, sar_gcd2 demos) prt, sar_g
% --------------------------------------------------------------
% REFERENCES: James P. LeSage, `Bayesian Estimation of Spatial Autoregressive
%             Models',  International Regional Science Review, 1997 
%             Volume 20, number 1\&2, pp. 113-129.
% For lndet information see: Ronald Barry and R. Kelley Pace, 
% "A Monte Carlo Estimator of the Log Determinant of Large Sparse Matrices", 
% Linear Algebra and its Applications", Volume 289, Number 1-3, 1999, pp. 41-54.
% and: R. Kelley Pace and Ronald P. Barry 
% "Simulating Mixed Regressive Spatially autoregressive Estimators", 
% Computational Statistics, 1998, Vol. 13, pp. 397-418.
%----------------------------------------------------------------

% written by:
% James P. LeSage, 12/2001
% Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com


% NOTE: some of the speed for large problems comes from:
% the use of methods pioneered by Pace and Barry.
% R. Kelley Pace was kind enough to provide functions
% lndetmc, and lndetint from his spatial statistics toolbox
% for which I'm very grateful.

timet = clock;

% error checking on inputs
[n junk] = size(y);
results.y = y;
[n1 k] = size(x);
[n3 n4] = size(W);
time1 = 0;
time2 = 0;
time3 = 0;

if n1 ~= n
error('sdm_gc: x-matrix contains wrong # of observations');
elseif n3 ~= n4
error('sdm_gc: W matrix is not square');
elseif n3~= n
error('sdm_gc: W matrix is not the same size at y,x');
end;

if nargin == 5
    prior.lflag = 1;
end;

[nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,seed] = sdm_parse(prior);


results.order = order;
results.iter = iter;

[rmin,rmax,time1] = sdm_eigs(eflag,W,rmin,rmax,n);

results.rmax = rmax; 
results.rmin = rmin;

results.lflag = ldetflag;

[detval,time2] = sdm_lndet(ldetflag,W,rmin,rmax,detval,order,iter);


% ====== initializations
% compute this stuff once to save time
Wy = sparse(W)*y;
Wx = sparse(W)*x(:,2:k);
xdx = [ x(:,2:k) Wx ones(n,1)];
[j nk] = size(xdx);
vi = ones(n,1);

Wy = sparse(W)*y;

if seed ~= 0;
rseed = num2str(seed);
end;

% =====================================================
% The sampler starts here
% =====================================================


fprintf(' -- sdm_gc: MCMC sampling -- \n');
if seed == 0
time3 = clock; % start timing the sampler
[rout,bout,sout,rdraw,vmean,yhat] = ...
   sdm_gcc(y,xdx,Wy,detval,rval,ndraw,nomit,nu,d0,mm,kk,rho,sige,novi_flag);
time3 = etime(clock,time3);
else
time3 = clock; % start timing the sampler
[rout,bout,sout,rdraw,vmean,yhat] = ...
   sdm_gcc(y,xdx,Wy,detval,rval,ndraw,nomit,nu,d0,mm,kk,rho,sige,novi_flag,rseed);
time3 = etime(clock,time3);
end;    

% return only ndraw-nomit draws
    rout = rout(nomit+1:ndraw,1);
    bout = bout(nomit+1:ndraw,:);
    sout = sout(nomit+1:ndraw,1);
    rdraw = rdraw(nomit+1:ndraw,1);

% =====================================================
% The sampler ends here
% =====================================================

% rearrange bdraws with constant term first, then x, then Wx
[na nb] = size(bout);
btmp = zeros(na,nb);
btmp(:,1) = bout(:,nb);
btmp(:,2:nb) = bout(:,1:nb-1);

pmean = mean(rout);
bmean = mean(bout);
yhat = xdx*bmean' + pmean*Wy;
e = (y - yhat);
epe = e'*e;
ym = y - mean(y);
rsqr1 = epe;
rsqr2 = ym'*ym;
results.rsqr = 1.0 - rsqr1/rsqr2; % r-squared
[n nvar] = size(xdx);
rsqr1 = rsqr1/(n-nvar);
rsqr2 = rsqr2/(n-1.0);
results.rbar = 1 - (rsqr1/rsqr2); % rbar-squared
results.sige = epe/(n - nvar);


results.meth  = 'sdm_gc';
results.bdraw = btmp;
results.pdraw = rout;
results.sdraw = sout;
results.vmean = vmean;
results.yhat  = yhat;
results.nobs  = n;
results.nvar  = k;
results.ndraw = ndraw;
results.nomit = nomit;
results.time1 = time1;
results.time2 = time2;
results.time3 = time3;
results.nu = nu;
results.d0 = d0;
results.tflag = 'plevel';
results.rflag = eflag;
if mm~= 0
results.rdraw = rdraw;
results.m     = mm;
results.k     = kk;
else
results.r     = rval;
results.rdraw = 0;
end;

results.time = etime(clock,timet);
results.seed = seed;
results.lndet = detval;
results.novi = novi_flag;





function [nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,seed] = sdm_parse(prior)
% PURPOSE: parses input arguments for far, far_g models
% ---------------------------------------------------
%  USAGE: [[nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,seed] = 
%                           sdm_parse(prior,k)
% where info contains the structure variable with inputs 
% and the outputs are either user-inputs or default values
% ---------------------------------------------------

% set defaults

eflag = 1;     % default to not computing eigenvalues
ldetflag = 1;  % default to 1999 Pace and Barry MC determinant approx
order = 50;    % there are parameters used by the MC det approx
iter = 30;     % defaults based on Pace and Barry recommendation
rmin = -1;     % use -1,1 rho interval as default
rmax = 1;
detval = 0;    % just a flag
rho = 0.5;
sige = 1.0;
rval = 4;
mm = 0;
kk = 0;
nu = 0;
d0 = 0;
cc = 0.2;
novi_flag = 0; % do vi-estimates
seed = 0;


fields = fieldnames(prior);
nf = length(fields);
if nf > 0
 for i=1:nf
    if strcmp(fields{i},'nu')
        nu = prior.nu;
    elseif strcmp(fields{i},'d0')
        d0 = prior.d0;  
    elseif strcmp(fields{i},'rval')
        rval = prior.rval; 
    elseif strcmp(fields{i},'m')
        mm = prior.m;
        kk = prior.k;
        rval = gamm_rnd(1,1,mm,kk);    % initial value for rval   
    elseif strcmp(fields{i},'rmin')
        rmin = prior.rmin;  eflag = 1;
    elseif strcmp(fields{i},'rmax')
        rmax = prior.rmax;  eflag = 1;
    elseif strcmp(fields{i},'lndet')
    detval = prior.lndet;
    ldetflag = -1;
    eflag = 1;
    rmin = detval(1,1);
    nr = length(detval);
    rmax = detval(nr,1);
    elseif strcmp(fields{i},'lflag')
        tst = prior.lflag;
        if tst == 0,
        ldetflag = 0; eflag = 0; % compute eigenvalues
        elseif tst == 1,
        ldetflag = 1; eflag = 1; % reset this from default
        elseif tst == 2,
        ldetflag = 2; eflag = 1; % reset this from default
        else
        error('sdm_gc: unrecognizable lflag value on input');
        end;
    elseif strcmp(fields{i},'order')
        order = prior.order;  
    elseif strcmp(fields{i},'iter')
        iter = prior.iter; 
    elseif strcmp(fields{i},'novi')
        novi_flag = prior.novi;
    elseif strcmp(fields{i},'seed')
        seed = prior.seed;
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
            error('sdm_gc: wrgon lndet input argument');
        end;
        [n1,n2] = size(detval);
        if n2 ~= 2
            error('sdm_gc: wrong sized lndet input argument');
        elseif n1 == 1
            error('sdm_gc: wrong sized lndet input argument');
        end;          
end;




