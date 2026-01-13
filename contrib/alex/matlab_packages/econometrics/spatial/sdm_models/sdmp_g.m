function results = sdmp_g(y,x,W,ndraw,nomit,prior)
% PURPOSE: Bayesian estimates of the heteroscedastic spatial durbin probit model
%         (I-rho*W)y = a + X*B1 + W*X*B2 + e, e = N(0,sige*V), V = diag(v1,v2,...vn)
%          r/vi = ID chi(r)/r, r = Gamma(m,k)
%          a, B1, B2 = diffuse
%          1/sige = Gamma(nu,d0), 
%          rho = Uniform(rmin,rmax) 
%          y = a vector of 0,1 binary values
%-------------------------------------------------------------
% USAGE: results = sdmp_g(y,x,W,ndraw,nomit,prior)
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
%            prior.lndet = a matrix returned by sar, sar_g, sarp_g, etc.
%                          containing log-determinant information to save time
%-------------------------------------------------------------
% RETURNS:  a structure:
%          results.meth   = 'sdmp_g'
%          results.bdraw  = bhat draws (ndraw-nomit x nvar)
%          results.pdraw  = rho  draws (ndraw-nomit x 1)
%          results.sdraw  = sige draws (ndraw-nomit x 1)
%          results.vmean  = mean of vi draws (nobs x 1) 
%          results.rdraw  = r draws (ndraw-nomit x 1) (if m,k input)
%          results.r      = value of hyperparameter r (if input)
%          results.nobs   = # of observations
%          results.nvar   = # of variables in x-matrix
%          results.ndraw  = # of draws
%          results.nomit  = # of initial draws omitted
%          results.y      = y-vector from input (nobs x 1)
%          results.zip    = # of zero y-values
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
%          results.novi  = novi from input (or default)
% --------------------------------------------------------------
% NOTES: constant term should be in 1st column of the x-matrix
%        constant is excluded from B2 estimates
% - use either improper prior.rval 
%          or informative Gamma prior.m, prior.k, not both of them
% - if you use lflag = 1 or 2, prior.rmin will be set = 0 
%                              prior.rmax will be set = 1
% - for n < 1000 you should use lflag = 0 to get exact results  
% --------------------------------------------------------------
% SEE ALSO: (sdmp_gd, sdmp_gd2 demos), prt
% --------------------------------------------------------------
% REFERENCES: James P. LeSage, "Bayesian Estimation of Limited Dependent
%             variable Spatial Autoregressive Models", 
%             Geographical Analysis, 2000, Vol. 32, pp. 19-35.
%             James P. LeSage, `Bayesian Estimation of Spatial Autoregressive
%             Models',  International Regional Science Review, 1997 
%             Volume 20, number 1\&2, pp. 113-129.
% also, R. Kelley Pace and Ronald P. Barry 
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
yin = y;
results.y = y;
[n1 k] = size(x);
[n3 n4] = size(W);
time1 = 0;
time2 = 0;
time3 = 0;

if n1 ~= n
error('sdmp_g: x-matrix contains wrong # of observations');
elseif n3 ~= n4
error('sdmp_g: W matrix is not square');
elseif n3~= n
error('sdmp_g: W matrix is not the same size at y,x');
end;

if nargin == 5
    prior.lflag = 1;
end;

[nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag] = sdm_parse(prior);

results.order = order;
results.iter = iter;

[rmin,rmax,time2] = sdm_eigs(eflag,W,rmin,rmax,n);

results.rmax = rmax; 
results.rmin = rmin;

results.lflag = ldetflag;

[detval,time1] = sdm_lndet(ldetflag,W,rmin,rmax,detval,order,iter);

% ====== initializations
% compute this stuff once to save time
iter = 1;
in = ones(n,1);
V = in;
vi = in;
Wy = sparse(W)*y;
Wx = sparse(W)*x(:,2:k);
xdx = [ x(:,2:k) Wx ones(n,1)];
[j nk] = size(xdx);
zipv = find(y == 0);
nzip = length(zipv);
sige = 1;


% storage for draws
          bsave = zeros(ndraw-nomit,nk);
          if mm~= 0
          rsave = zeros(ndraw-nomit,1);
          end;
          psave = zeros(ndraw-nomit,1);
          ssave = zeros(ndraw-nomit,1);
          vmean = zeros(n,1);
          ymean = zeros(n,1);
          yhat = zeros(n,1);

switch novi_flag
    
case{0} % do heteroscedastic model

tmp = sum(W.^2);
W2diag = tmp';
clear tmp;

hwait = waitbar(0,'sdmp\_g: MCMC sampling ...');
t0 = clock;                  
iter = 1;
          while (iter <= ndraw); % start sampling;
                  
          % update beta   
          xs = matmul(xdx,sqrt(V));
          ys = sqrt(V).*y;
          Wys = sqrt(V).*Wy;
          AI = inv(xs'*xs);
          bmat = AI*[xs'*ys xs'*Wys];
          beta = bmat(:,1) - rho*bmat(:,2);
          bhat = norm_rnd(sige*AI) + beta;
          
          % update sige
          % nu1 = n + 2*nu; 
          % e = (ys -rho*Wys - xs*bhat);
          % d1 = 2*d0 + e'*e;
          % chi = chis_rnd(1,nu1);
          % sige = d1/chi;
          sige = 1;

          % update vi
          ev = y - rho*Wy - xdx*bhat; 
          chiv = chis_rnd(n,rval+1);   
          vi = ((ev.*ev/sige) + in*rval)./chiv;
          V = in./vi; 
              
          % update rval
          if mm ~= 0           
          rval = gamm_rnd(1,1,mm,kk);  
          end;

          % update rho using numerical integration
          b0 = bmat(:,1);
          bd = bmat(:,2);
          e0 = ys - xs*b0;
          ed = Wys - xs*bd;
          epe0 = e0'*e0;
          eped = ed'*ed;
          epe0d = ed'*e0;
          
          rho = draw_rho(detval,epe0,eped,epe0d,n,nk,rho);
 
          % update z-values
          Wx1 = sparse(W)*xs;
          Wx2 = sparse(W)*Wx1;
          Wx3 = sparse(W)*Wx2;
          Wx4 = sparse(W)*Wx3;
          Wx5 = sparse(W)*Wx4;
          Wx6 = sparse(W)*Wx5;

          mu = xs*bhat + rho*Wx1*bhat + (rho.^2)*Wx2*bhat + (rho.^3)*Wx3*bhat + ...
               (rho.^4)*Wx4*bhat + (rho.^5)*Wx5*bhat + (rho.^6)*Wx6*bhat;

          ymu = ys - mu;

          dsig = ones(n,1) + rho*rho*W2diag;
          yvar = ones(n,1)./(vi.*dsig*sige);

          A = (1/sige)*(speye(n) - rho*W)*ymu;
          B = (speye(n) - rho*W')*A;
          C = ymu - yvar.*B;
          ym = mu + C;
                    
             ind = find(yin == 0);
	         y(ind,1) = normrt_rnd(ym(ind,1),yvar(ind,1),0);
	         ind = find(yin == 1);
	         y(ind,1) = normlt_rnd(ym(ind,1),yvar(ind,1),0);
         
          % reformulate Wy
          Wy = sparse(W)*y;
                    
               
    if iter > nomit % if we are past burn-in, save the draws
    bsave(iter-nomit,:) = bhat';
    ssave(iter-nomit,1) = sige;
    psave(iter-nomit,1) = rho;
    yhat = yhat + rho*Wys + xs*bhat;
    vmean = vmean + vi;
    ymean = ymean + y;
    if mm~= 0
        rsave(iter-nomit,1) = rval;
    end;         
    end;
                    

iter = iter + 1; 
waitbar(iter/ndraw);         
end; % end of sampling loop
close(hwait);

time3 = etime(clock,t0);

vmean = vmean/(ndraw-nomit);
yhat  = yhat/(ndraw-nomit);
yprob = stdn_cdf(yhat);
ymean = ymean /(ndraw-nomit);
results.vmean = vmean;

case{1} % do homoscedastic model

        
tmp = sum(W.^2);
W2diag = tmp';
clear tmp;
Wx1 = sparse(W)*xdx;
Wx2 = sparse(W)*Wx1;
Wx3 = sparse(W)*Wx2;
Wx4 = sparse(W)*Wx3;
Wx5 = sparse(W)*Wx4;
Wx6 = sparse(W)*Wx5;


hwait = waitbar(0,'sdmp\_g: MCMC sampling ...');
t0 = clock;                  
iter = 1;
          while (iter <= ndraw); % start sampling;
                  
          % update beta   
          AI = inv(xdx'*xdx);
          bmat = AI*[xdx'*y xdx'*Wy];
          beta = bmat(:,1) - rho*bmat(:,2);
          bhat = norm_rnd(sige*AI) + beta;
          
          % update sige
          % nu1 = n + 2*nu; 
          % e = (y - rho*Wy - xdx*bhat);
          % d1 = 2*d0 + e'*e;
          % chi = chis_rnd(1,nu1);
          % sige = d1/chi;
          sige = 1;

          % update rho using numerical integration
          b0 = bmat(:,1);
          bd = bmat(:,2);
          e0 = y - xdx*b0;
          ed = Wy - xdx*bd;
          epe0 = e0'*e0;
          eped = ed'*ed;
          epe0d = ed'*e0;
          
          rho = draw_rho(detval,epe0,eped,epe0d,n,nk,rho);
                       
          % update z-values
          mu = xdx*bhat + rho*Wx1*bhat + (rho.^2)*Wx2*bhat + (rho.^3)*Wx3*bhat + ...
               (rho.^4)*Wx4*bhat + (rho.^5)*Wx5*bhat + (rho.^6)*Wx6*bhat;

          ymu = y - mu;

          dsig = ones(n,1) + rho*rho*W2diag;
          yvar = ones(n,1)./(dsig*sige);

          A = (1/sige)*(speye(n) - rho*W)*ymu;

          B = (speye(n) - rho*W')*A;
          C = ymu - yvar.*B;
          ym = mu + C;
                    
             ind = find(yin == 0);
	         y(ind,1) = normrt_rnd(ym(ind,1),yvar(ind,1),0);
	         ind = find(yin == 1);
	         y(ind,1) = normlt_rnd(ym(ind,1),yvar(ind,1),0);
         
          % reformulate Wy
          Wy = sparse(W)*y;
                    
               
    if iter > nomit % if we are past burn-in, save the draws
    bsave(iter-nomit,:) = bhat';
    ssave(iter-nomit,1) = sige;
    psave(iter-nomit,1) = rho;
    yhat = yhat + rho*Wy + xdx*bhat;
    ymean = ymean + y;
    end;
                    
iter = iter + 1; 
waitbar(iter/ndraw);         
end; % end of sampling loop
close(hwait);

yhat = yhat/(ndraw-nomit);
rval = 0;
yprob = stdn_cdf(yhat);
ymean = ymean /(ndraw-nomit);
results.vmean = zeros(n,1);

otherwise
error('sdmp_g: unrecognized prior.novi value on input');
% we should never get here

end; % end of homo vs hetero models



% rearrange bdraws with constant term first, then x, then Wx
[na nb] = size(bsave);
btmp = zeros(na,nb);
btmp(:,1) = bsave(:,nb);
btmp(:,2:nb) = bsave(:,1:nb-1);

time = etime(clock,timet);

results.meth  = 'sdmp_g';
results.bdraw = btmp;
results.pdraw = psave;
results.sdraw = ssave;
results.yhat  = yhat;
results.ymean = ymean;
results.yprob = yprob;
results.nobs  = n;
results.nvar  = k;
results.ndraw = ndraw;
results.nomit = nomit;
results.time  = time;
results.time1 = time1;
results.time2 = time2;
results.time3 = time3;
results.nu = nu;
results.d0 = d0;
results.tflag = 'plevel';
results.lndet = detval;
results.novi = novi_flag;
results.zip = nzip;
if mm~= 0
results.rdraw = rsave;
results.m     = mm;
results.k     = kk;
else
results.r     = rval;
results.rdraw = 0;
end;


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




function [nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag] = sdm_parse(prior)
% PURPOSE: parses input arguments for far, far_g models
% ---------------------------------------------------
%  USAGE: [nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag] = 
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
        error('sdmp_g: unrecognizable lflag value on input');
        end;
    elseif strcmp(fields{i},'order')
        order = prior.order;  
    elseif strcmp(fields{i},'iter')
        iter = prior.iter; 
    elseif strcmp(fields{i},'novi')
        novi_flag = prior.novi;
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
            error('sdmp_g: wrgon lndet input argument');
        end;
        [n1,n2] = size(detval);
        if n2 ~= 2
            error('sdmp_g: wrong sized lndet input argument');
        elseif n1 == 1
            error('sdmp_g: wrong sized lndet input argument');
        end;          
end;



