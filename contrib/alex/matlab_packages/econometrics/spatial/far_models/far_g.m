function results = far_g(y,W,ndraw,nomit,prior)
% PURPOSE: Bayesian estimates for the 1st-order Spatial autoregressive model
%          y = rho*W*y + e,    e = N(0,sige*V), 
%          V = diag(v1,v2,...vn), r/vi = ID chi(r)/r, r = Gamma(m,k)
%          rho = uniform(rmin,rmax),  sige = gamma(nu,d0)    
%----------------------------------------------------------------
% USAGE: result =  far_g(y,W,ndraw,nomit,prior)
% where: y = nobs x 1 independent variable vector (mean = 0)
%        W = nobs x nobs 1st-order contiguity matrix (standardized)
%       ndraw = # of draws
%       nomit = # of initial draws omitted for burn-in
%       prior = a structure variable for prior information input
%        prior.nu,   = informative Gamma(nu,d0) prior on sige
%        prior.d0      default: nu=0,d0=0 (diffuse prior)
%        prior.rval, = r prior hyperparameter, default=4
%        prior.m,    = informative Gamma(m,k) prior on r
%        prior.k,    = informative Gamma(m,k) prior on r
%        prior.rmin, = (optional) min value of rho to use in sampling
%        prior.rmax, = (optional) max value of rho to use in sampling
%        prior.lflag = 0 for full lndet computation (default = 1, fastest)
%                    = 1 for MC approximation (fast for very large problems)
%                    = 2 for Spline approximation (medium speed)
%        prior.order = order to use with info.lflag = 1 option (default = 50)
%        prior.iter  = iters to use with info.lflag = 1 option (default = 30)   
%        prior.lndet = a matrix returned by sar, sar_g, sarp_g, etc.
%                      containing log-determinant information to save time
%---------------------------------------------------------------
% RETURNS: a structure:
%          results.meth   = 'far_g'
%          results.pdraw  = rho draws (ndraw-nomit x 1)
%          results.sdraw  = sige draws (ndraw-nomit x 1)
%          results.vmean  = mean of vi draws (nobs x 1)
%          results.rdraw  = r-value draws (ndraw-nomit x 1)
%          results.nu     = prior nu-value for sige (if prior input)
%          results.d0     = prior d0-value for sige (if prior input)
%          results.r      = value of hyperparameter r (if input)
%          results.m      = m prior parameter (if input)
%          results.k      = k prior parameter (if input)    
%          results.nobs   = # of observations
%          results.ndraw  = # of draws
%          results.nomit  = # of initial draws omitted
%          results.y      = actual observations
%          results.yhat   = mean of posterior for y-predicted (nobs x 1)
%          results.time   = total time taken
%          results.time1  = time for log determinant calcluation
%          results.time2  = time for eigenvalue calculation   
%          results.time3  = time taken for sampling                 
%          results.rmax   = 1/max eigenvalue of W (or rmax if input)
%          results.rmin   = 1/min eigenvalue of W (or rmin if input) 
%          results.tflag  = 'plevel' (default) for printing p-levels
%                         = 'tstat' for printing bogus t-statistics      
%          results.lflag  = lflag from input
%          results.liter  = info.iter option from input
%          results.order  = info.order option from input
%          results.limit  = matrix of [rho lower95,logdet approx, upper95] intervals
%                           (for the case of lflag = 1)      
%          results.lndet = a matrix containing log-determinant information
%                          (for use in later function calls to save time)
%----------------------------------------------------------------
% NOTES: - use either improper prior.rval 
%          or informative Gamma prior.m, prior.k, not both of them
% - if you use lflag = 1 or 2, prior.rmin will be set = -1 
%                              prior.rmax will be set = 1
% - for n < 1000 you should use lflag = 0 to get exact results        
%----------------------------------------------------------------
% SEE ALSO: (far_gd, far_gd2, far_gd3, demos) prt, plt
% --------------------------------------------------------------
% REFERENCES: James P. LeSage, `Bayesian Estimation of Spatial Autoregressive
%             Models',  International Regional Science Review, 1997 
%             Volume 20, number 1\&2, pp. 113-129.
% For lndet information see: Ronald Barry and R. Kelley Pace, 
% "A Monte Carlo Estimator of the Log Determinant of Large Sparse Matrices", 
% Linear Algebra and its Applications", Volume 289, Number 1-3, 1999, pp. 41-54.
% and: R. Kelley Pace and Ronald P. Barry "Simulating Mixed Regressive
% Spatially autoregressive Estimators", 
% Computational Statistics, 1998, Vol. 13, pp. 397-418.
%----------------------------------------------------------------

% written by:
% James P. LeSage, 1/2000
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

if (nargin < 4 | nargin > 5)
    error('far_g: Wrong # of input arguments');
end;

 [n n2] = size(W);  
 if n ~= n2 % a non-square spatial weight matrix
 error('far_g: Wrong size 1st-order contiguity matrix');
 end; 
 [tst junk] = size(y);
 if tst ~= n % y-vector length doesn't match W-matrix
 error('far_g: Wrong size y vector on input');
 end;
 if junk ~= 1 % user didn't enter a column vector
 error('far_g: Wrong size y vector on input');
 end;

 if nargin == 4
     prior.lflag = 1;
 end;
 
[nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter] = far_parse(prior);

results.y = y;      
results.nobs = n;
results.nvar = 1;   
results.meth = 'far_g';
results.order = order;
results.iter = iter;

timet = clock; % start the timer


V = ones(n,1); in = ones(n,1); % initial value for V   
ys = y.*sqrt(V);
vi = in;
          
bsave = zeros(ndraw-nomit,1);    % allocate storage for results
ssave = zeros(ndraw-nomit,1);
vmean = zeros(n,1);
yhat = zeros(n,1);

if mm~= 0                        % storage for draws on rvalue
rsave = zeros(ndraw-nomit,1);
end;

[rmin,rmax,time2] = far_eigs(eflag,W,rmin,rmax,n);

[detval,time1] = far_lndet(ldetflag,W,rmin,rmax,detval,order,iter);

iter = 1; 
time3 = clock; % start timing the sampler

hwait = waitbar(0,'Gibbs sampling ...');              

% =====================================================
% The sampler starts here
% =====================================================

Wy = sparse(W)*y;

while (iter <= ndraw);        % start sampling;

% update sige;
nu1 = n + nu; 
Wys = Wy.*sqrt(V);
e = ys - rho*Wys;
d1 = d0 + e'*e;
chi = chis_rnd(1,nu1);
t2 = chi/d1;
sige = 1/t2;

% update vi
e = y - rho*Wy;
chiv = chis_rnd(n,rval+1);  
vi = ((e.*e./sige) + in*rval)./chiv;
V = in./vi;   
ys = y.*sqrt(V);

% update rval
if mm ~= 0           
rval = gamm_rnd(1,1,mm,kk);  
end;

% update rho using numerical integration
          e0 = ys;
          ed = Wys;
          epe0 = e0'*e0;
          eped = ed'*ed;
          epe0d = ed'*e0;
          
          rho = draw_rho(detval,epe0,eped,epe0d,n,1,rho);


    if (iter > nomit)
    ssave(iter-nomit,1) = sige;
    bsave(iter-nomit,1) = rho;
    vmean = vmean + vi;
     if mm~= 0
     rsave(iter-nomit,1) = rval;
     end; 
    end; % end of if > nomit
    
iter = iter+1;
waitbar(iter/ndraw);

end; % end of sampling loop
close(hwait);
% =====================================================
% The sampler ends here
% =====================================================

time3 = etime(clock,time3);

vmean = vmean/(ndraw-nomit);
yhat = mean(bsave)*Wy;

results.meth  = 'far_g';
results.pdraw = bsave;
results.sdraw = ssave;
results.vmean = vmean;
results.yhat = yhat;
results.tflag = 'plevel';
results.lflag = ldetflag;
results.nobs  = n;
results.ndraw = ndraw;
results.nomit = nomit;
results.y = y;
results.nvar = 1;
if mm~= 0
results.rdraw = rsave;
results.m     = mm;
results.k     = kk;
else
results.r     = rval;
results.rdraw = 0;
end;

results.time = etime(clock,timet);
results.time1 = time1;
results.time2 = time2;
results.time3 = time3;
results.lndet = detval;
results.rmax = rmax; 
results.rmin = rmin;



function rho = draw_rho(detval,epe0,eped,epe0d,n,k,rho)
% update rho via univariate numerical integration

nmk = (n-k)/2;
nrho = length(detval(:,1));
iota = ones(nrho,1);

z = epe0*iota - 2*detval(:,1)*epe0d + detval(:,1).*detval(:,1)*eped;
den = detval(:,2) - nmk*log(z);
n = length(den);
y = detval(:,1);
adj = max(den);
den = den - adj;
x = exp(den);

% trapezoid rule
isum = sum((y(2:n,1) + y(1:n-1,1)).*(x(2:n,1) - x(1:n-1,1))/2);
z = abs(x/isum);
den = cumsum(z);

rnd = unif_rnd(1,0,1)*sum(z);
ind = find(den <= rnd);
idraw = max(ind);
if (idraw > 0 & idraw < nrho)
rho = detval(idraw,1);
end;


function [nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter] = far_parse(prior)
% PURPOSE: parses input arguments for far, far_g models
% ---------------------------------------------------
%  USAGE: [rmin,rmax,convg,maxit,detval,ldetflag,eflag,order,iter] = far_parse(info)
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
        error('far_g: unrecognizable lflag value on input');
        end;
    elseif strcmp(fields{i},'order')
        order = prior.order;  
    elseif strcmp(fields{i},'iter')
        iter = prior.iter; 
    end;
 end;
 
else, % the user has input a blank info structure
      % so we use the defaults
end; 

function [rmin,rmax,time2] = far_eigs(eflag,W,rmin,rmax,n);
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


function [detval,time1] = far_lndet(ldetflag,W,rmin,rmax,detval,order,iter);
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
            error('far_g: wrgon lndet input argument');
        end;
        [n1,n2] = size(detval);
        if n2 ~= 2
            error('far_g: wrong sized lndet input argument');
        elseif n1 == 1
            error('far_g: wrong sized lndet input argument');
        end;          
end;

