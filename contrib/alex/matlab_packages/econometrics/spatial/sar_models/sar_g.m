function results = sar_g(y,x,W,ndraw,nomit,prior)
% PURPOSE: Bayesian estimates of the spatial autoregressive model
%          y = rho*W*y + XB + e, e = N(0,sige*V), V = diag(v1,v2,...vn) 
%          r/vi = ID chi(r)/r, r = Gamma(m,k)
%          B = N(c,T), 
%          1/sige = Gamma(nu,d0), 
%          rho = Uniform(rmin,rmax) 
%-------------------------------------------------------------
% USAGE: results = sar_g(y,x,W,ndraw,nomit,prior)
% where: y = dependent variable vector (nobs x 1)
%        x = independent variables matrix (nobs x nvar)
%        W = 1st order contiguity matrix (standardized, row-sums = 1)
%    ndraw = # of draws
%    nomit = # of initial draws omitted for burn-in            
%    prior = a structure variable with:
%            prior.beta  = prior means for beta,   c above (default 0)
%            priov.bcov  = prior beta covariance , T above (default 1e+12)
%            prior.rval  = r prior hyperparameter, default = 4
%            prior.novi  = 1 turns off sampling for vi, producing homoscedastic model            
%            prior.m     = informative Gamma(m,k) prior on r
%            prior.k     = (default: not used)
%            prior.nu    = informative Gamma(nu,d0) prior on sige
%            prior.d0    = default: nu=0,d0=0 (diffuse prior)
%            prior.rmin  = (optional) min rho used in sampling (default = -1)
%            prior.rmax  = (optional) max rho used in sampling (default = 1)  
%            prior.lflag = 0 for full lndet computation (default = 1, fastest)
%                        = 1 for MC approx (fast for large problems)
%                        = 2 for Spline approx (medium speed)
%            prior.order = order to use with prior.lflag = 1 option (default = 50)
%            prior.iter  = iters to use with prior.lflag = 1 option (default = 30) 
%            prior.dflag = 0 for numerical integration, 1 for Metropolis-Hastings
%                          (default = 1 for nobs <= 1,000, =0 for nobs > 1,000)
%            prior.lndet = a matrix returned by sar, sar_g, sarp_g, etc.
%                          containing log-determinant information to save time
%-------------------------------------------------------------
% RETURNS:  a structure:
%          results.meth   = 'sar_g'
%          results.bdraw  = bhat draws (ndraw-nomit x nvar)
%          results.pdraw  = rho  draws (ndraw-nomit x 1)
%          results.sdraw  = sige draws (ndraw-nomit x 1)
%          results.vmean  = mean of vi draws (nobs x 1) 
%          results.rdraw  = r draws (ndraw-nomit x 1) (if m,k input)
%          results.bmean  = b prior means, prior.beta from input
%          results.bstd   = b prior std deviations sqrt(diag(prior.bcov))
%          results.r      = value of hyperparameter r (if input)
%          results.novi   = 1 for prior.novi = 1, 0 for prior.rval input
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
%                              prior.rmax will be set = 1
% - for n < 1000 you should use lflag = 0 to get exact results  
% --------------------------------------------------------------
% SEE ALSO: (sar_gd, sar_gd2 demos) sar_gc.m, prt
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
[n1 k] = size(x);
[n3 n4] = size(W);
time1 = 0;
time2 = 0;
time3 = 0;

results.nobs  = n;
results.nvar  = k;
results.y = y;      

if n1 ~= n
error('sar_g: x-matrix contains wrong # of observations');
elseif n3 ~= n4
error('sar_g: W matrix is not square');
elseif n3~= n
error('sar_g: W matrix is not the same size at y,x');
end;

if nargin == 5
    prior.lflag = 1;
end;

[nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,c,T,prior_beta,cc,metflag] = sar_parse(prior,k);

% error checking on prior information inputs
[checkk,junk] = size(c);
if checkk ~= k
error('sar_g: prior means are wrong');
elseif junk ~= 1
error('sar_g: prior means are wrong');
end;

[checkk junk] = size(T);
if checkk ~= k
error('sar_g: prior bcov is wrong');
elseif junk ~= k
error('sar_g: prior bcov is wrong');
end;

results.order = order;
results.iter = iter;

timet = clock; % start the timer

[rmin,rmax,time1] = sar_eigs(eflag,W,rmin,rmax,n);

[detval,time2] = sar_lndet(ldetflag,W,rmin,rmax,detval,order,iter);

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
vi = in;
Wy = sparse(W)*y;

switch novi_flag
    
case{0} % we do heteroscedastic model    

hwait = waitbar(0,'sar\_g: MCMC sampling ...');
t0 = clock;                  
iter = 1;
          while (iter <= ndraw); % start sampling;
                  
          % update beta   
          xs = matmul(x,sqrt(V));
          ys = sqrt(V).*y;
          Wys = sqrt(V).*Wy;
          AI = inv(xs'*xs + sige*TI);		  
          yss = ys - rho*Wys;          
          b = xs'*yss + sige*TIc;
          b0 = AI*b;
          bhat = norm_rnd(sige*AI) + b0;  
          xb = xs*bhat;
          
          % update sige
          nu1 = n + 2*nu; 
          e = (yss - xb);
          d1 = 2*d0 + e'*e;
          chi = chis_rnd(1,nu1);
          sige = d1/chi;

          % update vi
          ev = y - rho*Wy - x*bhat; 
          chiv = chis_rnd(n,rval+1);   
          vi = ((ev.*ev/sige) + in*rval)./chiv;
          V = in./vi; 
              
          % update rval
          if mm ~= 0           
          rval = gamm_rnd(1,1,mm,kk);  
          end;
          
     if metflag == 1
         % metropolis step to get rho update
          rhox = c_sar(rho,ys,xb,sige,W,detval);
          accept = 0; 
          rho2 = rho + cc*randn(1,1); 
          while accept == 0
           if ((rho2 > rmin) & (rho2 < rmax)); 
           accept = 1;  
           else
           rho2 = rho + cc*randn(1,1);
           end; 
          end;
          rhoy = c_sar(rho2,ys,xb,sige,W,detval);
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
          rtmp(iter,1) = rho;
      else % we use numerical integration to perform rho-draw
          b0 = AI*xs'*ys;
          bd = AI*xs'*Wys;
          e0 = ys - xs*b0;
          ed = Wys - xs*bd;
          epe0 = e0'*e0;
          eped = ed'*ed;
          epe0d = ed'*e0;
          rho = draw_rho(detval,epe0,eped,epe0d,n,k,rho);
      end;

               
    if iter > nomit % if we are past burn-in, save the draws
    bsave(iter-nomit,1:k) = bhat';
    ssave(iter-nomit,1) = sige;
    psave(iter-nomit,1) = rho;
    yhat = yhat + rho*Wy + x*bhat;
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
yhat  = yhat/(ndraw-nomit);

case{1} % we do homoscedastic model    
    
hwait = waitbar(0,'sar\_g: MCMC sampling ...');
t0 = clock;                  
iter = 1;
xpx = (x'*x + TI);
xpy = (x'*y + TIc);
% integrate the conditional distribution for rho
% if we are not doing metropolis-hastings for rho
% and if we have a diffuse prior on beta

if (metflag == 0 & prior_beta == 0)
          b0 = xpx\xpy;
          bd = xpx\(x'*Wy);
          e0 = y - x*b0;
          ed = Wy - x*bd;
          epe0 = e0'*e0;
          eped = ed'*ed;
          epe0d = ed'*e0;
nmk = (n-k)/2;
nrho = length(detval(:,1));
iota = ones(nrho,1);
z = epe0*iota - 2*detval(:,1)*epe0d + detval(:,1).*detval(:,1)*eped;
den = detval(:,2) - nmk*log(z);
n = length(den);
yy = detval(:,1);
adj = max(den);
den = den - adj;
xx = exp(den);
% trapezoid rule
isum = sum((yy(2:n,1) + yy(1:n-1,1)).*(xx(2:n,1) - xx(1:n-1,1))/2);
zz = abs(xx/isum);
den = cumsum(zz);
end;
% end of integrating rho

          while (iter <= ndraw); % start sampling;
                  
          % update beta   
          AI = inv(xpx + sige*TI);		  
          ys = y - rho*Wy;          
          b = x'*ys + sige*TIc;
          b0 = AI*b;
          bhat = norm_rnd(sige*AI) + b0;  
          xb = x*bhat;
          
          % update sige
          nu1 = n + 2*nu; 
          e = (ys - xb);
          d1 = 2*d0 + e'*e;
          chi = chis_rnd(1,nu1);
          sige = d1/chi;
          
     if metflag == 1
         % metropolis step to get rho update
          rhox = c_sar(rho,y,xb,sige,W,detval);
          accept = 0; 
          rho2 = rho + cc*randn(1,1); 
          while accept == 0
           if ((rho2 > rmin) & (rho2 < rmax)); 
           accept = 1;  
           else
           rho2 = rho + cc*randn(1,1);
           end; 
          end;
          rhoy = c_sar(rho2,y,xb,sige,W,detval);
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
          rtmp(iter,1) = rho;
      end;
      if (metflag == 0 & prior_beta == 1) % we use numerical integration to perform rho-draw
          b0 = AI*xpy;
          bd = AI*x'*Wy;
          e0 = y - x*b0;
          ed = Wy - x*bd;
          epe0 = e0'*e0;
          eped = ed'*ed;
          epe0d = ed'*e0;
          rho = draw_rho(detval,epe0,eped,epe0d,n,k,rho);
      end;
      if (metflag == 0 & prior_beta == 0) % we can work real fast here
          rho = rdraw_rho(detval,den,zz,rho);
      end;

               
    if iter > nomit % if we are past burn-in, save the draws
    bsave(iter-nomit,1:k) = bhat';
    ssave(iter-nomit,1) = sige;
    psave(iter-nomit,1) = rho;
    yhat = yhat + rho*Wy + xb;
    end;
                    
iter = iter + 1; 
waitbar(iter/ndraw);         
end; % end of sampling loop
close(hwait);

vmean = vi;
rval = 0;
yhat  = yhat/(ndraw-nomit);

otherwise
error('sar_g: unrecognized novi_flag value on input');
% we should never get here

end; % end of homoscedastic vs. heteroscedastic options


time3 = etime(clock,t0);

time = etime(clock,timet);

results.meth  = 'sar_g';
results.bdraw = bsave;
results.pdraw = psave;
results.sdraw = ssave;
results.vmean = vmean;
results.yhat  = yhat;
results.bmean = c;
results.bstd  = sqrt(diag(T));
results.ndraw = ndraw;
results.nomit = nomit;
results.time  = time;
results.time1 = time1;
results.time2 = time2;
results.time3 = time3;
results.nu = nu;
results.d0 = d0;
results.tflag = 'plevel';
results.dflag = metflag;
results.order = order;
results.rmax = rmax; 
results.rmin = rmin;
results.lflag = ldetflag;
results.lndet = detval;
results.novi  = novi_flag;
results.priorb = prior_beta;

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


function rho = rdraw_rho(detval,den,z,rho)
% update rho via univariate numerical integration

nrho = length(detval);
rnd = unif_rnd(1,0,1)*sum(z);
ind = find(den <= rnd);
idraw = max(ind);
if (idraw > 0 & idraw < nrho)
rho = detval(idraw,1);
end;


function cout = c_sar(rho,y,xb,sige,W,detval,c,T);
% PURPOSE: evaluate the conditional distribution of rho given sige
%  spatial autoregressive model using sparse matrix algorithms
% ---------------------------------------------------
%  USAGE:cout = c_sar(rho,y,x,b,sige,W,detval,p,R)
%  where:  rho  = spatial autoregressive parameter
%          y    = dependent variable vector
%          W    = spatial weight matrix
%        detval = an (ngrid,2) matrix of values for det(I-rho*W) 
%                 over a grid of rho values 
%                 detval(:,1) = determinant values
%                 detval(:,2) = associated rho values
%          sige = sige value
%          p    = (optional) prior mean for rho
%          R    = (optional) prior variance for rho
% ---------------------------------------------------
%  RETURNS: a conditional used in Metropolis-Hastings sampling
%  NOTE: called only by sar_g
%  --------------------------------------------------
%  SEE ALSO: sar_g, c_far, c_sac, c_sem
% ---------------------------------------------------

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

if nargin == 6      % case of diffuse prior
n = length(y);
z = speye(n) - rho*sparse(W);
e = z*y - xb; 
epe = (e'*e)/(2*sige);

elseif nargin == 8  % case of informative prior
T = T*sige;
z = (speye(n) - rho*W)*e;
epe = ((z'*z)/2*sige) + 0.5*(((rho-c)^2)/T);

else
error('c_sar: Wrong # of inputs arguments');

end;

cout =   detm - epe;



function [nu,d0,rval,mm,kk,rho,sige,rmin,rmax,detval,ldetflag,eflag,order,iter,novi_flag,c,T,prior_beta,cc,metflag] = sar_parse(prior,k)
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
        error('sar_g: unrecognizable lflag value on input');
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


