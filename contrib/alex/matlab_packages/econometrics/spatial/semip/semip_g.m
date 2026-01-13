function results = semip_g(y,x,W,m,mobs,ndraw,nomit,prior)
% PURPOSE: Bayesian Probit model with individual effects exhibiting spatial dependence:
%	      Y = (Yi, i=1,..,m) with each vector, Yi = (yij:j=1..Ni) consisting of individual 
%          dichotomous observations in regions i=1..m, as defined by yij = Indicator(zij>0), 
%          where latent vector Z = (zij) is given by the linear model:
%
%          Z = x*b + del*a + e   with:
%
%          x = n x k matrix of explanatory variables [n = sum(Ni: i=1..m)]; 
%			  del = n x m indicator matrix with del(j,i) = 1 iff indiv j is in reg i;
%          a = (ai: i=1..m) a vector of random regional effects modeled by
%
%          a = rho*W*a + U,     U ~ N[0,sige*I_m] ; (I_m = m-square Identity matrix)
%
%          and with e ~ N(0,V), V = diag(del*v) where v = (vi:i=1..m). 
%
%          The priors for the above parameters are of the form:
%          r/vi ~ ID chi(r), r ~ Gamma(m,k)
%          b ~ N(c,T),  
%          1/sige ~ Gamma(nu,d0), 
%          rho ~ Uniform(1/lmin,1/lmax)  
%-----------------------------------------------------------------
% USAGE: results = semip_g(y,x,W,m,mobs,ndraw,nomit,prior)
% where: y = dependent variable vector (nobs x 1) [must be zero-one]
%        x = independent variables matrix (nobs x nvar)
%        W = 1st order contiguity matrix (standardized, row-sums = 1)
%        m = # of regions 
%     mobs = an m x 1 vector containing the # of observations in each
%            region [= (Ni:i=1..m) above]
%    ndraw = # of draws
%    nomit = # of initial draws omitted for burn-in            
%    prior = a structure variable with:
%            prior.beta  = prior means for beta,  (= c above) 
%                          (default = 0)
%            prior.bcov  = prior beta covariance , (= T above)  
%                          [default = 1e+12*I_k ]
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
%---------------------------------------------------
% RETURNS:  a structure:
%          results.meth   = 'semip_g'
%          results.bdraw  = bhat draws (ndraw-nomit x nvar)
%          results.pdraw  = rho  draws (ndraw-nomit x 1)
%          results.adraw  = a draws (ndraw-nomit x m)
%          results.amean  = mean of a draws (m x 1)
%          results.sdraw  = sige draws (ndraw-nomit x 1)
%          results.vmean  = mean of vi draws (m x 1) 
%          results.rdraw  = r draws (ndraw-nomit x 1) (if m,k input)
%          results.bmean  = b prior means, prior.beta from input
%          results.bstd   = b prior std deviations sqrt(diag(prior.bcov))
%          results.r      = value of hyperparameter r (if input)
%          results.rsqr   = R-squared
%          results.nobs   = # of observations
%          results.mobs   = mobs vector from input
%          results.nreg   = # of regions
%          results.nvar   = # of variables in x-matrix
%          results.ndraw  = # of draws
%          results.nomit  = # of initial draws omitted
%          results.y      = actual (0,1) observations (nobs x 1)
%          results.zmean  = mean of latent z-draws (nobs x 1)
%          results.yhat   = mean of posterior y-predicted (nobs x 1)
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
%          results.rflag  = 1, if a normal(p,S) prior for rho, 0 otherwise
%          results.lflag  = lflag from input
%          results.iter   = prior.iter  option from input
%          results.order  = prior.order option from input
%          results.limit  = matrix of [rho lower95,logdet approx, upper95] 
%                           intervals for the case of lflag = 1 
% ----------------------------------------------------
% SEE ALSO: sem_gd, prt, semp_g, coda
% ----------------------------------------------------
% REFERENCES: Tony E. Smith "A Bayesian Probit Model with Spatial Dependencies" unpublished manuscript
% For lndet information see: Ronald Barry and R. Kelley Pace, "A Monte Carlo Estimator
% of the Log Determinant of Large Sparse Matrices", Linear Algebra and
% its Applications", Volume 289, Number 1-3, 1999, pp. 41-54.
% and: R. Kelley Pace and Ronald P. Barry "Simulating Mixed Regressive
% Spatially autoregressive Estimators", Computational Statistics, 1998,
% Vol. 13, pp. 397-418.
%----------------------------------------------------------------

% written by:
% James P. LeSage, Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jpl@jpl.econ.utoledo.edu

% NOTE: much of the speed for large problems comes from:
% the use of methods pioneered by Pace and Barry.
% R. Kelley Pace was kind enough to provide functions
% lndetmc, and lndetint from his spatial statistics toolbox
% for which I'm very grateful.

timet = clock;

time1 = 0;
time2 = 0;
time3 = 0;

% error checking on inputs
[n junk] = size(y);
results.y = y;
[n1 k] = size(x);
[n3 n4] = size(W);

if n1 ~= n
error('semip_g: x-matrix contains wrong # of observations');
elseif n3 ~= n4
error('semip_g: W matrix is not square');
elseif n3~= m
error('semip_g: W matrix is not the same size as # of regions');
end;

% check that mobs vector is correct
obs_chk = sum(mobs);
if obs_chk ~= n
error('semip_g: wrong # of observations in mobs vector');
end;
if length(mobs) ~= m
error('semip_g: wrong size mobs vector -- should be m x 1');
end;

% set defaults
mm = 0;    % default for m
rval = 4;  % default for r
nu = 0;    % default diffuse prior for sige
d0 = 0;
sig0 = 1;  % default starting values for sige
c = zeros(k,1);   % diffuse prior for beta
T = eye(k)*1e+12;
p0 = 0.5;         % default starting value for rho
inV0 = ones(n,1);   % default starting value for inV [= inv(V)]
a0 = ones(m,1);
z0 = y; %default starting value for latent vector z
rflag = 1; % don't compute max and min eigenvalues of W
lflag = 1; % use Pace's fast MC approximation to lndet(I-rho*W)
pflag = 0; % use diffuse prior on rho
rmin = 0;  % use 0,1 rho interval
rmax = 1;
order = 50; iter = 30; % defaults


if nargin == 8   % parse input values
 fields = fieldnames(prior);
 nf = length(fields);
 for i=1:nf
    if strcmp(fields{i},'rval')
        rval = prior.rval; 
    elseif strcmp(fields{i},'m')
        mm = prior.m;
        kk = prior.k;
        rval = gamm_rnd(1,1,mm,kk);    % initial value for rval
    elseif strcmp(fields{i},'beta')
        c = prior.beta;
    elseif strcmp(fields{i},'bcov')
        T = prior.bcov;
    elseif strcmp(fields{i},'nu')
        nu = prior.nu;
    elseif strcmp(fields{i},'d0')
        d0 = prior.d0;
    elseif strcmp(fields{i},'rmin')
    rmin = prior.rmin;
    rmax = prior.rmax;
    rflag = 1;
    elseif strcmp(fields{i},'lflag')
        tst = prior.lflag;
        if tst == 0,
        lflag = 0; rflag = 0; % compute min and max eigenvalues
        elseif tst == 1,
        lflag = 1; rmin = 0; rmax = 1; rflag = 1; % reset this from default
        elseif tst == 2,
        lflag = 2; rmin = 0; rmax = 1; rflag = 1; % reset this from default
        else
        error('semip_g: unrecognizable lflag value on input');
        end;
    elseif strcmp(fields{i},'order')
        order = prior.order;  results.order = order;
    elseif strcmp(fields{i},'iter')
    iter = prior.iter; results.iter = iter;
    end;
 end;

elseif nargin == 7   % we supply all defaults

else
error('Wrong # of arguments to semip_g');
end;

results.order = order;
results.iter = iter;

% error checking on prior information inputs
[checkk,junk] = size(c);
if checkk ~= k
error('semip_g: prior means are wrong');
elseif junk ~= 1
error('semip_g: prior means are wrong');
end;

[checkk junk] = size(T);
if checkk ~= k
error('semip_g: prior bcov is wrong');
elseif junk ~= k
error('semip_g: prior bcov is wrong');
end;


if rflag == 0 % we compute eigenvalues
t0 = clock;  
opt.tol = 1e-3; opt.disp = 0;
lambda = eigs(sparse(W),speye(m),1,'SR',opt);  
rmin = 1/lambda;   
rmax = 1;
time1 = etime(clock,t0);
end;
results.rmax = rmax; results.rmin = rmin;

% do lndet calculations using 1 of 3 methods
switch lflag

case {0} % use full method, no approximations

t0 = clock;
out = lndetfull(W,rmin,rmax);
time2 = etime(clock,t0);
tt=.001:.001:1; % interpolate a finer grid
outi = interp1(out.rho,out.lndet,tt','spline');
detval = [tt' outi];

rv = tt';
nr = length(rv);

case{1} % use Pace and Barry, 1999 MC approximation

t0 = clock;
out = lndetmc(order,iter,W);
time2 = etime(clock,t0);
tt=.001:.001:1; % interpolate a finer grid
outi = interp1(out.rho,out.lndet,tt','spline');
detval = [tt' outi];

results.limit = [out.rho out.lo95 out.lndet out.up95];

rv = tt';
nr = length(rv);

case{2} % use Pace and Barry, 1998 spline interpolation

t0 = clock;
out = lndetint(W);
time2 = etime(clock,t0);
tt=.001:.001:1; % interpolate a finer grid
outi = interp1(out.rho,out.lndet,tt','spline');
detval = [tt' outi];

rv = tt';
nr = length(rv);

otherwise
error('semip_g: unrecognized lflag value on input');
% we should never get here

end; % end of different det calculation options

% storage for draws
          bsave = zeros(ndraw-nomit,k);
          asave = zeros(ndraw-nomit,m);
          if mm~= 0
          rsave = zeros(ndraw-nomit,1);
          end;
          ssave = zeros(ndraw-nomit,1);
          psave = zeros(ndraw-nomit,1);
          vmean = zeros(m,1);
          amean = zeros(m,1);
          zmean = zeros(n,1);
          yhat = zeros(n,1);

% ====== initializations
% compute this once to save time
TI = inv(T);
TIc = TI*c;
iter = 1;
rho = p0;
a = a0;
z = z0;
in = ones(n,1);
inV = inV0;
vi = ones(m,1);
sige = sig0;
tvec = ones(n,1); % initial values
evec = ones(m,1);
b1 = ones(m,1);
Bp = speye(m) - rho*sparse(W);

%Computations for updating vector a

if(m > 100)
W1 = zeros(m,m-1);
W2 = zeros(m-1,m);
W3 = zeros(m,m-1);

for i = 1:m
   w1(i) = W(:,i)'*W(:,i);
   if i == 1
      W1(1,:) = W(1,[2:m]);            %W-rows minus entry i
      W2(:,1) = W([2:m],1);            %W-columns minus entry i
      W3(1,:) = W(:,1)'*W(:,[2:m]);
   elseif i == m
      W1(m,:) = W(m,[1:m-1]);
      W2(:,m) = W([1:m-1],m);
      W3(m,:) = W(:,m)'*W(:,[1:m-1]);
   else
      W1(i,:) = W(i,[1:i-1,i+1:m]);
      W2(:,i) = W([1:i-1,i+1:m],i);
      W3(i,:) = W(:,i)'*W(:,[1:i-1,i+1:m]);
   end
end

end %end if(m > 10)
    
%*********************************
% START SAMPLING
%*********************************

dmean = zeros(length(detval),1);

hwait = waitbar(0,'MCMC sampling ...');
t0 = clock;                  
iter = 1;
          while (iter <= ndraw); % start sampling;

          
          % UPDATE: beta   
          xs = matmul(sqrt(inV),x);
          zs = sqrt(inV).*z; 
          A0i = inv(xs'*xs + TI);
          zmt = sqrt(inV).*(z-tvec);
                   
          b = xs'*zmt + TIc;
          b0 = A0i*b;
          bhat = norm_rnd(A0i) + b0; 
          %Update b1
          e0 = z - x*bhat;
          cobs = 0;
          for i=1:m;
            obs = mobs(i,1);
            b1(i,1) = sum(e0(cobs+1:cobs+obs,1)/vi(i,1));
            cobs = cobs + obs;
           end;

                             
          % UPDATE: a 
          
          if m <= 100   %Procedure for small m 
             vii = ones(m,1)./vi;
             A1i = inv((1/sige)*Bp'*Bp + diag(vii.*mobs));
             a = norm_rnd(A1i) + A1i*b1;             
             
           else   %Procedure for large m
             
             cobs = 0;
             
             for i = 1:m
                
                obs = mobs(i,1); 
                
                if i == 1            %Form complementary vector
                   ai = a(2:m);
                elseif i == m
                   ai = a(1:m-1);
                else
                   ai = a([1:i-1,i+1:m]);
                end                
                
                                
                di = (1/sige) + ((rho^2)/sige)*w1(i) + (obs/vi(i));                
                               
                zi = z(cobs+1:cobs+obs,1);                
                xbi = x([cobs+1:cobs+obs],:)* bhat;                
                phi = (1/vi(i))*(ones(1,obs)*(zi - xbi));  
                awi = ai'*(W1(i,:)' + W2(:,i));                
                bi = phi + (rho/sige)*awi - ((rho^2)/sige)*(W3(i,:)*ai) ;                
                a(i) = (bi/di) + sqrt(1/di)*randn(1,1);                                              
                cobs = cobs + obs;
                
             end %end for i = 1:m 
             
          end %end if on m
          
          % Update tvec = del*a   
             
          cobs = 0;              
          for i=1:m;
             obs = mobs(i,1);
             tvec(cobs+1:cobs+obs,1) = a(i,1)*ones(obs,1);
             cobs = cobs + obs;
          end;
          
                  
			 % UPDATE: sige

          term1 = a'*Bp'*Bp*a + 2*d0;
          chi = chis_rnd(1,m + 2*nu);
          sige = term1/chi; 
          
          
			 % UPDATE: vi (and form inV, b1)

           e = z - x*bhat - tvec;         
                                
           cobs = 0;
           for i=1:m;
            obs = mobs(i,1);
            ee = e(cobs+1:cobs+obs,1)'*e(cobs+1:cobs+obs,1);
            chi = chis_rnd(1,rval+obs);
            vi(i,1) = (ee + rval)/chi; 
            inV(cobs+1:cobs+obs,1) = ones(obs,1)/vi(i,1);
            b1(i,1) = sum(e0(cobs+1:cobs+obs,1)/vi(i,1));
            cobs = cobs + obs;
           end;
    
          % UPDATE: rval (if necessary)
           
          if mm ~= 0           
          rval = gamm_rnd(1,1,mm,kk);  
          end;
                 
          % UPDATE: rho (using univariate integration)
          C0 = a'*a;
          Wa = W*a;
          C1 = a'*Wa;
          C2 = Wa'*Wa;
          
		  rho = draw_rho(detval,C0,C1,C2,m,k,rho);
 
	  	  Bp = speye(m) - rho*sparse(W); 

   % UPDATE: z

      lp = x*bhat + tvec;
	       ind = find(y == 0);
		    tmp = ones(n,1)./inV;
          z(ind,1) = normrt_rnd(lp(ind,1),tmp(ind,1),0);
          %z(ind,1) = rtanorm_combo(lp(ind,1),tmp(ind,1),zeros(length(ind),1));
	       ind = find(y == 1);
          z(ind,1) = normlt_rnd(lp(ind,1),tmp(ind,1),0);
          %z(ind,1) = rtbnorm_combo(lp(ind,1),tmp(ind,1),zeros(length(ind),1));

    if iter > nomit % if we are past burn-in, save the draws
    	bsave(iter-nomit,1:k) = bhat';
    	asave(iter-nomit,1:m) = a'; % should just return the mean
    	ssave(iter-nomit,1) = sige;
    	psave(iter-nomit,1) = rho;
    	amean = amean + a;
    	yhat = yhat + lp;
    	zmean = zmean + z;
    	vmean = vmean + vi;

    	if mm~= 0
        rsave(iter-nomit,1) = rval;
     	end;
    end;
    
           
    iter = iter + 1;
       
%[iter rho sige]

waitbar(iter/ndraw);     

end; % end of sampling loop

close(hwait);

time3 = etime(clock,t0);

vmean = vmean/(ndraw-nomit);
yhat = yhat/(ndraw-nomit);
amean = amean/(ndraw-nomit);
zmean = zmean/(ndraw-nomit);


time = etime(clock,timet);

results.meth  = 'semip_g';
results.bdraw = bsave;
results.adraw = asave;
results.pdraw = psave;
results.sdraw = ssave;
results.vmean = vmean;
results.amean = amean;
results.yhat  = yhat;
results.zmean = zmean;
results.bmean = c;
results.bstd  = sqrt(diag(T));
results.nobs  = n;
results.nvar  = k;
results.ndraw = ndraw;
results.nomit = nomit;
results.time = time;
results.time1 = time1;
results.time2 = time2;
results.time3 = time3;
results.nu = nu;
results.d0 = d0;
results.tflag = 'plevel';
results.lflag = lflag;
results.nreg = m;
results.mobs = mobs;
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

z = epe0*iota - 2*detval(:,1)*eped + detval(:,1).*detval(:,1)*epe0d;
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

