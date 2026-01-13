function results = sem_g(y,x,W,ndraw,nomit,prior)
% PURPOSE: Bayesian estimates of the spatial error model
%          y = XB + u, u = rho*W + e
%          e = N(0,sige*V), V = diag(v1,v2,...vn) 
%          r/vi = ID chi(r)/r, r = Gamma(m,k)
%          B = N(c,T), 
%          1/sige = Gamma(nu,d0), 
%          rho = Uniform(rmin,rmax) 
%-------------------------------------------------------------
% USAGE: results = sem_g(y,x,W,ndraw,nomit,prior)
% where: y = dependent variable vector (nobs x 1)
%        x = independent variables matrix (nobs x nvar)
%        W = 1st order contiguity matrix (standardized, row-sums = 1)
%    ndraw = # of draws
%    nomit = # of initial draws omitted for burn-in            
%    prior = a structure variable with:
%            prior.beta  = prior means for beta,   c above (default 0)
%            priov.bcov  = prior beta covariance , T above (default 1e+12)
%            prior.novi  = 1 turns off sampling for vi, producing homoscedastic model            
%            prior.rval  = r prior hyperparameter, default=4
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
%            prior.dflag = 0 for numerical integration, 1 for Metropolis-Hastings
%                          (default = 0)
%            prior.lndet = a matrix returned by sar, sar_g, sarp_g, etc.
%                          containing log-determinant information to save time
%-------------------------------------------------------------
% RETURNS:  a structure:
%          results.meth   = 'sem_g'
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
%          results.dflag  = dflag value from input (or default value used)
%          results.lndet = a matrix containing log-determinant information
%                          (for use in later function calls to save time)
% --------------------------------------------------------------
% NOTES: - use either improper prior.rval 
%          or informative Gamma prior.m, prior.k, not both of them
% - if you use lflag = 1 or 2, prior.rmin will be set = -1 
%                              prior.rmax will be set = +1
% - for n < 1000 you should use lflag = 0 to get exact results  
% --------------------------------------------------------------
% SEE ALSO: (sem_gd, sem_gd2 demos) sem_gc.m, prt
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
error('sem_g: x-matrix contains wrong # of observations');
elseif n3 ~= n4
error('sem_g: W matrix is not square');
elseif n3~= n
error('sem_g: W matrix is not the same size at y,x');
end;

if nargin == 5
    prior.lflag = 1;
end;

[nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,c,T,prior_beta,cc,metflag] = sem_parse(prior,k);

results.order = order;
results.iter = iter;

% error checking on prior information inputs
[checkk,junk] = size(c);
if checkk ~= k
error('sem_g: prior means are wrong');
elseif junk ~= 1
error('sem_g: prior means are wrong');
end;

[checkk junk] = size(T);
if checkk ~= k
error('sem_g: prior bcov is wrong');
elseif junk ~= k
error('sem_g: prior bcov is wrong');
end;

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

[rmin,rmax,time1] = sem_eigs(eflag,W,rmin,rmax,n);

results.rmin = rmin;
results.rmax = rmax;
results.lflag = ldetflag;

[detval,time2] = sem_lndet(ldetflag,W,rmin,rmax,detval,order,iter);


% storage for draws
          bsave = zeros(ndraw-nomit,k);
          if mm~= 0
          rsave = zeros(ndraw-nomit,1);
          end;
          psave = zeros(ndraw-nomit,1);
          ssave = zeros(ndraw-nomit,1);
          vmean = zeros(n,1);
          ymean = zeros(n,1);
          yhat = zeros(n,1);

% ====== initializations
% compute this stuff once to save time
TI = inv(T);
TIc = TI*c;
iter = 1;
in = ones(n,1);
V = in;
Wy = sparse(W)*y;
Wx = sparse(W)*x;
vi = in;

switch novi_flag
    
case{0} % we do heteroscedastic model
    
hwait = waitbar(0,'MCMC sampling ...');
t0 = clock;                  
iter = 1;
          while (iter <= ndraw); % start sampling;
                  
          % update beta   
          xs = x - rho*Wx;
          xss = matmul(xs,sqrt(V));
          AI = inv(xss'*xss + sige*TI);
          ys = y - rho*Wy;
          yss = matmul(ys,sqrt(V));
          b = xss'*yss + sige*TIc;
          b0 = AI*b;
          bhat = norm_rnd(sige*AI) + b0; 
            
          % update sige
          nu1 = n + 2*nu; 
          e = (speye(n) - rho*W)*(y - x*bhat);
          d1 = 2*d0 + e'*e;
          chi = chis_rnd(1,nu1);
          sige = d1/chi;

          % update vi
          ev = ys - xs*bhat; 
          chiv = chis_rnd(n,rval+1);   
          vi = ((ev.*ev/sige) + in*rval)./chiv;
          V = in./vi; 
              
          % update rval
          if mm ~= 0           
          rval = gamm_rnd(1,1,mm,kk);  
          end;

       if metflag == 0
          % update rho using numerical integration          
          rho = draw_rho(detval,y,x,Wy,Wx,V,n,k,rmin,rmax,rho);
       else
          % update rho using metropolis-hastings
          rhox = c_sem(rho,ev,W,sige,detval); 
          accept = 0;
          rho2 = rho + cc*randn(1,1);
          while accept == 0
           if ((rho2 > rmin) & (rho2 < rmax)); 
           accept = 1;  
           else
           rho2 = rho + cc*randn(1,1);
           end; 
          end; 
           rhoy = c_sem(rho2,ev,W,sige,detval);
          ru = unif_rnd(1,0,1);
          if ((rhoy - rhox) > exp(1)),
          p = 1;
          else,          
          ratio = exp(rhoy-rhox);
          p = min(1,ratio);
          end;
              if (ru < p)
              rho = rho2;
              end;
       end % end of rho update
                           
               
    if iter > nomit % if we are past burn-in, save the draws
    bsave(iter-nomit,1:k) = bhat';
    ssave(iter-nomit,1) = sige;
    psave(iter-nomit,1) = rho;
    yhat = yhat + x*bhat;
    vmean = vmean + vi;

    if mm~= 0
        rsave(iter-nomit,1) = rval;
    end;         
    end;
                    

iter = iter + 1; 
waitbar(iter/ndraw);         
end; % end of sampling loop
close(hwait);

vmean = vmean/(ndraw-nomit);


case{1} % we do homoscedastic model
    
% do integration up front
% if we have a diffuse prior on beta
% and if we are not using metropolis-hastings
if (metflag == 0 & prior_beta == 0)
nmk = (n-k)/2;
nrho = length(detval(:,1));
iota = ones(nrho,1);
rvec = detval(:,1);
epe = zeros(nrho,1);

for i=1:nrho;
xs = x - rvec(i,1)*Wx;
ys = y - rvec(i,1)*Wy;
bs = (xs'*xs + TI)\(xs'*ys + TIc);
e = ys - xs*bs;
epe(i,1) = e'*e;
end;

den = detval(:,2) - nmk*log(epe);
adj = max(den);
den = den - adj;
den = exp(den);

nn = length(den);
yy = detval(:,1);
xx = den;

% trapezoid rule
isum = sum((yy(2:nn,1) + yy(1:nn-1,1)).*(xx(2:nn,1) - xx(1:nn-1,1))/2);

zz = abs(xx/isum);
den = cumsum(zz);

end; % end of integration

hwait = waitbar(0,'MCMC sampling ...');
t0 = clock;                  
iter = 1;
          while (iter <= ndraw); % start sampling;
                  
          % update beta   
          xs = x - rho*Wx;
          AI = inv(xs'*xs + sige*TI);
          ys = y - rho*Wy;
          b = xs'*ys + sige*TIc;
          b0 = AI*b;
          bhat = norm_rnd(sige*AI) + b0; 
            
          % update sige
          nu1 = n + 2*nu; 
          e = (y - x*bhat);
          e = (speye(n) - rho*W)*e;
          d1 = 2*d0 + e'*e;
          chi = chis_rnd(1,nu1);
          sige = d1/chi;
              
          % update rval
          if mm ~= 0           
          rval = gamm_rnd(1,1,mm,kk);  
          end;

     if (metflag == 0 & prior_beta == 0) %  update rho using numerical integration          
          rho = draw_rhoh(detval,den,zz,nn,k,rho);
     end;
     if (metflag == 0 & prior_beta == 1)
          rho = draw_rho(detval,y,x,Wy,Wx,V,nn,k,rmin,rmax,rho);
     end;
          
     if (metflag == 1)
         % update rho using metropolis-hastings
          rhox = c_sem(rho,e,W,sige,detval); 

          accept = 0;
          rho2 = rho + cc*randn(1,1);
          while accept == 0
           if ((rho2 > rmin) & (rho2 < rmax)); 
           accept = 1;  
           else
           rho2 = rho + cc*randn(1,1);
           end; 
          end; 
           rhoy = c_sem(rho2,e,W,sige,detval);
          ru = unif_rnd(1,0,1);
          if ((rhoy - rhox) > exp(1)),
          p = 1;
          else,          
          ratio = exp(rhoy-rhox);
          p = min(1,ratio);
          end;
              if (ru < p)
              rho = rho2;
              end;
      end; % end of rho update
                           
               
    if iter > nomit % if we are past burn-in, save the draws
    bsave(iter-nomit,1:k) = bhat';
    ssave(iter-nomit,1) = sige;
    psave(iter-nomit,1) = rho;
    yhat = yhat + x*bhat;

    if mm~= 0
        rsave(iter-nomit,1) = rval;
    end;         
    end;
                    

iter = iter + 1; 
waitbar(iter/ndraw);         
end; % end of sampling loop
close(hwait);

otherwise
error('sem_g: unrecognized novi_flag value on input');
% we should never get here

end; % end of homoscedastic vs. heteroscedastic options


time3 = etime(clock,t0);

yhat  = yhat/(ndraw-nomit);

rho = mean(psave);
y = results.y;
n = length(y);
e = y - yhat;
e = (speye(n) - rho*W)*e;
sigu = (e'*e);
ym = y - mean(y);
rsqr1 = sigu;
rsqr2 = ym'*ym;
rsqr = 1.0 - rsqr1/rsqr2; % conventional r-squared


time = etime(clock,timet);

results.meth  = 'sem_g';
results.bdraw = bsave;
results.pdraw = psave;
results.sdraw = ssave;
results.vmean = vmean;
results.yhat  = yhat;
results.bmean = c;
results.bstd  = sqrt(diag(T));
results.nobs  = n;
results.nvar  = k;
results.rsqr  = rsqr;
results.sige  = sigu/n;
results.ndraw = ndraw;
results.nomit = nomit;
results.time  = time;
results.time1 = time1;
results.time2 = time2;
results.time3 = time3;
results.nu = nu;
results.d0 = d0;
results.tflag = 'plevel';
results.novi = novi_flag;
results.lndet = detval;
results.dflag = metflag;
results.priorb = prior_beta;
if mm~= 0
results.rdraw = rsave;
results.m     = mm;
results.k     = kk;
else
results.r     = rval;
results.rdraw = 0;
end;



function rho = draw_rhoh(detval,den,z,n,k,rho)
% update rho via univariate numerical integration
% for the homoscedastic model case

nrho = length(detval);
rnd = unif_rnd(1,0,1)*sum(z);
ind = find(den <= rnd);
idraw = max(ind);
if (idraw > 0 & idraw < n)
rho = detval(idraw,1);
end;

function rho = draw_rho(detval,y,x,Wy,Wx,V,n,k,rmin,rmax,rho)
% update rho via univariate numerical integration
% for the heteroscedastic model case

nmk = (n-k)/2;
nrho = length(detval(:,1));
iota = ones(nrho,1);
rvec = detval(:,1);
epe = zeros(nrho,1);

for i=1:nrho;
xs = x - rvec(i,1)*Wx;
xs = matmul(xs,sqrt(V));
ys = y - rvec(i,1)*Wy;
ys = ys.*sqrt(V);
bs = (xs'*xs)\(xs'*ys);
e = ys - xs*bs;
epe(i,1) = e'*e;
end;

den = detval(:,2) - nmk*log(epe);
adj = max(den);
den = den - adj;
den = exp(den);


n = length(den);
y = detval(:,1);
x = den;

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

function cout = c_sem(rho,e,W,sige,detval,c,T)
% PURPOSE: evaluates conditional distribution of rho
%  for the Bayesian spatial error model 
%          y = XB + e, e = N(0,sige*V), V = diag(v1,v2,...vn) 
%          r/vi = ID chi(r)/r, r = Gamma(m,k)
%          B = N(c,T), 
%          rho = N(p,R) or: 
%          rho = Uniform(1/lmin,1/lmax)  
%          1/sige = Gamma(nu,d0), 
% ---------------------------------------------------
%  USAGE: cout = c_sem(rho,e,W,sige,V,detval,p,R)
%  where: rho  = spatial autoregressive parameter
%           e  = z - x*b 
%           W  = spatial weight matrix     
%         sige = sige scalar shown above
%       detval = an (ngrid,2) matrix of values for det(I-rho*W) 
%                 over a grid of rho values 
%                 detval(:,1) = determinant values
%                 detval(:,2) = associated rho values
%            p = (optional) prior mean for rho (default = uniform)
%            R = (optional) prior variance for rho     
% ---------------------------------------------------
%  RETURNS: a  scalar evaluation of the conditional distribution
%  p(rho | other parameters in the model)
%  --------------------------------------------------
%  SEE ALSO: lndetfull, lndetint, lndetmc functions that
%            provide detval for use in this function
% ---------------------------------------------------

% written by: James P. LeSage 1/2000
% University of Toledo
% Department of Economics
% Toledo, OH 43606
% jlesage@spatial-econometrics.com

gsize = detval(2,1) - detval(1,1);
% Note these are actually log detvalues
i1 = find(detval(:,1) <= rho + gsize);
i2 = find(detval(:,1) <= rho - gsize);
i1 = max(i1);
i2 = max(i2);
index = round((i1+i2)/2);
if isempty(index)
index = 1;
end;
detm = detval(index,2); 

n = length(e);

if nargin == 5     % case of diffuse prior
z = (speye(n) - rho*W)*e;
epe = z'*z;
epe = epe/(2*sige);
elseif nargin == 7 % case of informative prior
T = T*sige;
z = (speye(n) - rho*W)*e;
epe = ((z'*z)/2*sige) + 0.5*(((rho-c)^2)/T);

else
error('c_sem: Wrong # of inputs arguments');
end;

cout =  detm - epe;



function [nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,c,T,prior_beta,cc,metflag] = sem_parse(prior,k)
% PURPOSE: parses input arguments for far, far_g models
% ---------------------------------------------------
%  USAGE: [nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,c,T,prior_beta,cc,metflag] = 
%                           sar_parse(prior,k)
% where info contains the structure variable with inputs 
% and the outputs are either user-inputs or default values
% ---------------------------------------------------

% set defaults

eflag = 1;     % default to not computing eigenvalues
ldetflag = 1;  % default to 1999 Pace and Barry MC determinant approx
order = 50;    % there are parameters used by the MC det approx
iter = 30;     % defaults based on Pace and Barry recommendation
rmin = -0.99;     % use -1,1 rho interval as default
rmax = 0.99;
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
cc=0.1;
metflag = 0;


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
        error('sem_g: unrecognizable lflag value on input');
        end;
    elseif strcmp(fields{i},'order')
        order = prior.order;  
    elseif strcmp(fields{i},'iter')
        iter = prior.iter; 
    elseif strcmp(fields{i},'novi')
        novi_flag = prior.novi;
    elseif strcmp(fields{i},'dflag')
        metflag = prior.dflag;
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
            error('sem_g: wrgon lndet input argument');
        end;
        [n1,n2] = size(detval);
        if n2 ~= 2
            error('sem_g: wrong sized lndet input argument');
        elseif n1 == 1
            error('sem_g: wrong sized lndet input argument');
        end;          
end;


