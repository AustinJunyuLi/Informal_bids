function results = sem(y,x,W,info)
% PURPOSE: computes spatial error model estimates
%           y = XB + u,  u = p*W*u + e, using sparse algorithms
% ---------------------------------------------------
%  USAGE: results = sem(y,x,W,info)
%  where: y = dependent variable vector
%         x = independent variables matrix
%         W = contiguity matrix (standardized)
%       info = an (optional) structure variable with input options:
%       info.rmin  = (optional) minimum value of rho to use in search  
%       info.rmax  = (optional) maximum value of rho to use in search    
%       info.convg = (optional) convergence criterion (default = 1e-4)
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
%         results.meth  = 'sem'
%         results.beta  = bhat
%         results.rho   = rho (p above)
%         results.tstat = asymp t-stats (last entry is rho)
%         results.yhat  = yhat
%         results.resid = residuals
%         results.sige  = sige = e'(I-p*W)'*(I-p*W)*e/n
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
%         results.liter = info.iter option from input
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
%  NOTES: if you use lflag = 1 or 2, info.rmin will be set = -1 
%                                    info.rmax will be set = 1
%         For n < 1000 you should use lflag = 0 to get exact results
% --------------------------------------------------    
%  SEE ALSO: prt(results), sar, sdm, sac, far
% ---------------------------------------------------
% REFERENCES: Luc Anselin Spatial Econometrics (1988) pages 182-183.
% For lndet information see: Ronald Barry and R. Kelley Pace, 
% "A Monte Carlo Estimator of the Log Determinant of Large Sparse Matrices", 
% Linear Algebra and its Applications", Volume 289, Number 1-3, 1999, pp. 41-54.
% and: R. Kelley Pace and Ronald P. Barry 
% "Simulating Mixed Regressive Spatially autoregressive Estimators", 
% Computational Statistics, 1998, Vol. 13, pp. 397-418.
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

timet = clock; % start the clock for overall timing


% check size of user inputs for comformability
[n nvar] = size(x);
results.meth = 'sem';

[n1 n2] = size(W);
if n1 ~= n2
error('sem: wrong size weight matrix W');
elseif n1 ~= n
error('sem: wrong size weight matrix W');
end;

% return the easy stuff
results.y = y;
results.nobs = n;
results.nvar = nvar; 

% if we have no options, invoke defaults
if nargin == 3
    info.lflag = 1;
end;

% parse input options
[rmin,rmax,convg,maxit,detval,ldetflag,eflag,order,miter,options] = sem_parse(info);


% compute eigenvalues or limits
[rmin,rmax,time2] = sem_eigs(eflag,W,rmin,rmax,n);

results.rmin = rmin;
results.rmax = rmax;
results.lflag = ldetflag;
results.miter = miter;
results.order = order;

% do log-det calculations
[detval,time1] = sem_lndet(ldetflag,W,rmin,rmax,detval,order,miter);



t0 = clock;
Wx = sparse(W)*x;
Wy = sparse(W)*y;
% step 1) ols of y on x -> b0
b0 = (x'*x)\(x'*y); 
% step 2) e = y - x*b0
eD = y - x*b0;
% step 3) find rho that maximizes Lc
econverge = eD;
criteria = 0.001;
converge = 1.0;
iter = 1;
itermax = 100;

while (converge > criteria & iter < itermax)
    
	 options = optimset('fminbnd');
    [rho,like,exitflag,output] = fminbnd('f_sem',rmin,rmax,options,eD,W,detval);

% step 4) find Begls
xs = x - rho*Wx;
ys = y - rho*Wy;
begls = (xs'*xs)\(xs'*ys);

% step 5) find Eegls
eD = y - x*begls;

% step 6) check convergence
converge = max(abs(eD - econverge));
econverge = eD;

iter = iter + 1;

end;
% end of while loop

liktmp = like;
time4 = etime(clock,t0);

if iter == itermax
fprintf(1,'sem: convergence not obtained in %4d iterations \n',itermax);
end;
results.iter = iter;


% compute results 
results.beta= begls;
results.rho = rho;
results.yhat = x*results.beta;
results.resid = y - results.yhat;

Be = (speye(n) - rho*W)*eD;
epe = Be'*Be;
results.sige = (1/n)*epe;

sigu = epe;
sige = results.sige;
parm = [results.beta
        results.rho
        results.sige];

results.lik = f2_sem(parm,y,x,W,detval);
    
if n <= 500, % t-stats using information matrix (Anselin, 1982 pages 
t0 = clock;
B = (speye(n) - rho*sparse(W));
BI = inv(B); WB = W*BI;
pterm = trace(WB'*WB);
xpx = zeros(nvar+2,nvar+2);
xpx(1:nvar,1:nvar) = (1/sige)*x'*B'*B*x;
% rho, rho
xpx(nvar+1,nvar+1) = trace(WB'*WB) + pterm;
% sige, sige
xpx(nvar+2,nvar+2) = n/(2*sige*sige);
% rho, sige
xpx(nvar+1,nvar+2) = -(1/sige)*(rho*trace(WB'*WB) - trace(BI'*WB));
xpx(nvar+2,nvar+1) = xpx(nvar+1,nvar+2);
tmp = diag(inv(xpx));
bvec = [results.beta
        results.rho];
results.tstat = bvec./(sqrt(tmp(1:nvar+1,1)));
time3 = etime(clock,t0);

elseif n > 500 

t0 = clock;
hessn = hessian('f2_sem',parm,y,x,W,detval);

if hessn(nvar+2,nvar+2) == 0
 hessn(nvar+2,nvar+2) = 1/sige;  % this is a hack for very large models that 
end;                             % should not affect inference in these cases


xpxi = inv(-hessn); 
xpxi = diag(xpxi(1:nvar+1,1:nvar+1));
zip = find(xpxi <= 0);
 if length(zip) > 0
 xpxi(zip,1) = 1;
 fprintf(1,'sem: negative or zero variance from numerical hessian \n');
 fprintf(1,'sem: replacing t-stat with 0 \n');
 end;
 tmp = [results.beta
        results.rho];
 results.tstat = tmp./sqrt(xpxi);
 if length(zip) ~= 0
 results.tstat(zip,1) = 0;
 end;
time3 = etime(clock,t0);

end; % end of t-stat calculations


ym = y - mean(y);
rsqr1 = sigu;
rsqr2 = ym'*ym;
results.rsqr = 1.0 - rsqr1/rsqr2; % r-squared
rsqr1 = rsqr1/(n-nvar);
rsqr2 = rsqr2/(n-1.0);
results.rbar = 1 - (rsqr1/rsqr2); % rbar-squared
results.lndet = detval;

results.time = etime(clock,timet);
results.time1 = time1;
results.time2 = time2;
results.time3 = time3;
results.time4 = time4;

function llike = f2_sem(parm,y,x,W,detval)
% PURPOSE: evaluates log-likelihood -- given ML parameters
%  spatial error model using sparse matrix algorithms
% ---------------------------------------------------
%  USAGE:llike = f2_sem(parm,y,X,W,detm)
%  where: parm = vector of maximum likelihood parameters
%                parm(1:k-2,1) = b, parm(k-1,1) = rho, parm(k,1) = sige
%         y    = dependent variable vector (n x 1)
%         X    = explanatory variables matrix (n x k)
%         W    = spatial weight matrix
%         ldet = matrix with [rho log determinant] values
%                computed in sem.m using one of Kelley Pace's routines
% ---------------------------------------------------                                           
%  NOTE: this is really two functions depending
%        on nargin = 3 or nargin = 4 (see the function)
% ---------------------------------------------------
%  RETURNS: a  scalar equal to minus the log-likelihood
%           function value at the ML parameters
%  --------------------------------------------------
%  SEE ALSO: sem, f2_sem2, f_sem
% ---------------------------------------------------

% written by: James P. LeSage 4/2002
% University of Toledo
% Department of Economics
% Toledo, OH 43606
% jlesage@spatial.econometrics.com

n = length(y);
k = length(parm);
b = parm(1:k-2,1);
rho = parm(k-1,1);
sige = parm(k,1);


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
ez = (speye(n) - rho*sparse(W))*(y - x*b);
epe = ez'*ez;
llike = -(n/2)*(1+log(2*pi)) - (n/2)*log(epe/n) + detm;

function lik = f_sem(rho,eD,W,detval)
% PURPOSE: evaluates concentrated log-likelihood for the 
%  spatial error model using sparse matrix algorithms
% ---------------------------------------------------
%  USAGE:llike = f_sem(lam,eD,W,detm)
%  where: rho  = spatial error parameter
%         eD   = begls residuals
%         W    = spatial weight matrix
%         detm =  matrix with [rho log determinant] values
%                computed in sem.m using one of 
%                Kelley Pace's routines           
% ---------------------------------------------------
%  RETURNS: a  scalar equal to minus the log-likelihood
%           function value at the parameter rho
%  --------------------------------------------------
%  NOTE: this is really two functions depending
%        on nargin = 3 or nargin = 4 (see the function)
% ---------------------------------------------------        
%  SEE ALSO: sem, f_far, f_sac, f_sar
% ---------------------------------------------------

% written by: James P. LeSage 1/2000
% University of Toledo
% Department of Economics
% Toledo, OH 43606
% jlesage@spatial-econometrics.com


n = length(eD); 
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
tmp = speye(n) - rho*sparse(W);
epe = eD'*tmp'*tmp*eD;
lik = (n/2)*log(pi) + (n/2)*log(epe) - detm;



function [rmin,rmax,convg,maxit,detval,ldetflag,eflag,order,iter,options] = sem_parse(info)
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
        error('sar: unrecognizable lflag value on input');
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

function [rmin,rmax,time2] = sem_eigs(eflag,W,rmin,rmax,n);
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


function [detval,time1] = sem_lndet(ldetflag,W,rmin,rmax,detval,order,iter);
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
            error('sem: wrong lndet input argument');
        end;
        [n1,n2] = size(detval);
        if n2 ~= 2
            error('sem: wrong sized lndet input argument');
        elseif n1 == 1
            error('sem: wrong sized lndet input argument');
        end;          
end;


