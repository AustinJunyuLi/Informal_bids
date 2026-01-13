function results = sar_gc(y,x,W,ndraw,nomit,prior)
% PURPOSE: Bayesian estimates of the heteroscedastic spatial autoregressive model C-MEX version
%          y = rho*W*y + XB + e, e = N(0,sige*V), V = diag(v1,v2,...vn) 
%          r/vi = ID chi(r)/r, r = Gamma(m,k)
%          B = N(c,T), 
%          1/sige = Gamma(nu,d0), 
%          rho = Uniform(rmin,rmax)  
%-------------------------------------------------------------
% USAGE: results = sar_gc(y,x,W,ndraw,nomit,prior)
% where: y = dependent variable vector (nobs x 1)
%        x = independent variables matrix (nobs x nvar)
%        W = 1st order contiguity matrix (standardized, row-sums = 1)
%    ndraw = # of draws
%    nomit = # of initial draws omitted for burn-in            
%    prior = a structure variable with:
%            prior.beta  = prior means for beta,   c above (default 0)
%            priov.bcov  = prior beta covariance,  T above (default 1e+12)
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
%            prior.dflag = 0 for return of only nomit+1:ndraw draws (default)
%                        = 1 for return of all 1:ndraw draws
%-------------------------------------------------------------
% RETURNS:  a structure:
%          results.meth   = 'sar_gc'
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
%          results.seed   = seed (if input, or zero if not)
%          results.dflag  = dflag from input (or default value)
% --------------------------------------------------------------
% NOTES: - use either improper prior.rval 
%          or informative Gamma prior.m, prior.k, not both of them
% - if you use lflag = 1 or 2, prior.rmin will be set = -1 
%                              prior.rmax will be set = 1
% - for n < 1000 you should use lflag = 0 to get exact results  
% --------------------------------------------------------------
% SEE ALSO: (sar_gcd, sar_gcd2 demos) prt, sar_g
% --------------------------------------------------------------
% REFERENCES: James P. LeSage, `Bayesian Estimation of Spatial Autoregressive
%             Models',  International Regional Science Review, 1997 
%             Volume 20, number 1&2, pp. 113-129.
% For lndet information see: Ronald Barry and R. Kelley Pace, 
% "A Monte Carlo Estimator of the Log Determinant of Large Sparse Matrices", 
% Linear Algebra and its Applications", Volume 289, Number 1-3, 1999, pp. 41-54.
% and: R. Kelley Pace and Ronald P. Barry 
% "Simulating Mixed Regressive Spatially autoregressive Estimators", 
% Computational Statistics, 1998, Vol. 13, pp. 397-418.
%----------------------------------------------------------------

% written by:
% James P. LeSage, 4/2002
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
error('sar_gc: x-matrix contains wrong # of observations');
elseif n3 ~= n4
error('sar_gc: W matrix is not square');
elseif n3~= n
error('sar_gc: W matrix is not the same size at y,x');
end;

if nargin == 5
    prior.lflag = 1;
end;

[nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,c,T,prior_beta,seed,dflag] = sar_parse(prior,k);


results.order = order;
results.iter = iter;

% error checking on prior information inputs
[checkk,junk] = size(c);
if checkk ~= k
error('sar_gc: prior means are wrong');
elseif junk ~= 1
error('sar_gc: prior means are wrong');
end;

[checkk junk] = size(T);
if checkk ~= k
error('sar_gc: prior bcov is wrong');
elseif junk ~= k
error('sar_gc: prior bcov is wrong');
end;



results.lflag = ldetflag;


[detval,time2] = sar_lndet(ldetflag,W,rmin,rmax,detval,order,iter);

[rmin,rmax,time1] = sar_eigs(eflag,W,rmin,rmax,n);

results.rmax = rmax; 
results.rmin = rmin;


% ====== initializations
% compute this stuff once to save time
TI = inv(T);
TIc = TI*c;

Wy = sparse(W)*y;

% =====================================================
% The sampler starts here
% =====================================================
if seed ~= 0;
rseed = num2str(seed);
end;

if novi_flag == 1;
    rval = 0;
end;


fprintf(' -- sar_gc: MCMC sampling -- \n');
if seed == 0
time3 = clock; % start timing the sampler
[rout,bout,sout,rdraw,vmean] = ...
   sar_gcc(y,x,Wy,detval,rval,ndraw,nomit,nu,d0,mm,kk,TI,TIc,rho,sige);
time3 = etime(clock,time3);
else
time3 = clock; % start timing the sampler
[rout,bout,sout,rdraw,vmean] = ...
   sar_gcc(y,x,Wy,detval,rval,ndraw,nomit,nu,d0,mm,kk,TI,TIc,rho,sige,rseed);
time3 = etime(clock,time3);
end;

% return only ndraw-nomit draws
if dflag == 0
    rout = rout(nomit+1:ndraw,1);
    bout = bout(nomit+1:ndraw,1:k);
    sout = sout(nomit+1:ndraw,1);
    rdraw = rdraw(nomit+1:ndraw,1);
end;
% =====================================================
% The sampler ends here
% =====================================================

rho = mean(rout);
beta = mean(bout);

yhat  = rho*Wy + x*beta';

results.meth  = 'sar_gc';
results.bdraw = bout;
results.pdraw = rout;
results.sdraw = sout;
results.vmean = vmean;
results.yhat  = yhat;
results.bmean = c;
results.bstd  = sqrt(diag(T));
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
results.priorb = prior_beta;
if mm~= 0
results.rdraw = rdraw;
results.m     = mm;
results.k     = kk;
else
results.r     = rval;
results.rdraw = 0;
end;
results.seed = seed;
results.dflag = dflag;
results.time = etime(clock,timet);
results.novi = novi_flag;



function [nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,c,T,prior_beta,seed,dflag] = sar_parse(prior,k)
% PURPOSE: parses input arguments for far, far_g models
% ---------------------------------------------------
%  USAGE: [nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,c,T,prior_beta,cc,metflag,seed,dflag] = 
%                           sar_parse(prior,k)
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
c = zeros(k,1);   % diffuse prior for beta
T = eye(k)*1e+12;
prior_beta = 0;   % flag for diffuse prior on beta
cc = 0.2;
novi_flag = 0; % do vi-estimates
seed = 0;
dflag = 0;


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
    elseif strcmp(fields{i},'beta')
        c = prior.beta;
        prior_beta = 1; % flag for informative prior on beta
    elseif strcmp(fields{i},'bcov')
        T = prior.bcov;
        prior_beta = 1; % flag for informative prior on beta
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
        error('sar_g: unrecognizable lflag value on input');
        end;
    elseif strcmp(fields{i},'order')
        order = prior.order;  
    elseif strcmp(fields{i},'iter')
        iter = prior.iter; 
    elseif strcmp(fields{i},'novi')
        novi_flag = prior.novi;
    elseif strcmp(fields{i},'seed');
    seed = prior.seed;
    seedflag = 1;    
    elseif strcmp(fields{i},'dflag');
    dflag = prior.dflag;
    end;
 end;
 
else, % the user has input a blank info structure
      % so we use the defaults
end; 

function [rmin,rmax,time2] = sar_eigs(eflag,W,rmin,rmax,n);
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


function [detval,time1] = sar_lndet(ldetflag,W,rmin,rmax,detval,order,iter);
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
            error('sar_g: wrong lndet input argument');
        end;
        [n1,n2] = size(detval);
        if n2 ~= 2
            error('sar_g: wrong sized lndet input argument');
        elseif n1 == 1
            error('sar_g: wrong sized lndet input argument');
        end;          
end;



