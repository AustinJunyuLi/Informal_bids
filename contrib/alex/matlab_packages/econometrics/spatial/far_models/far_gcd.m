% PURPOSE: An example of using far_g() Gibbs sampling
%          1st-order spatial autoregressive model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: far_gd (see also far_gd2 for a large data set)
%---------------------------------------------------

clear all;
% W-matrix from Anselin's neigbhorhood crime data set
load anselin.dat; % standardized 1st-order spatial weight matrix
xc = anselin(:,4);
yc = anselin(:,5);
[j1 W j2] = xy2cont(xc,yc);
[n junk] = size(W);
IN = eye(n); 
rho = 0.7;  % true value of rho
sige = 1;
y = inv(IN-rho*W)*randn(n,1)*sqrt(sige); 
ydev = y - mean(y);
vnames = strvcat('y-simulated','y-spatial lag');

% do maximum likelihood for comparison    
info.rmin = 0; 
info.rmax = 1; % constrain 0 < rho < 1     
res = far(ydev,W,info);
disp('True value of rho = 0.7');
prt(res,vnames);

ndraw = 2200;
nomit = 200;

% Gibbs sampling function homoscedastic prior
prior.rmin = 0; 
prior.rmax = 1; % constrain 0 < rho < 1
prior.rval = 100; % homoscedastic prior for comparison
                 % to maximum likelihood estimates
% this is the c-mex version
result = far_gc(ydev,W,ndraw,nomit,prior);
disp('True value of rho = 0.7');
result.tflag = 'tstat';
prt(result,vnames);

% Gibbs sampling function heteroscedastic prior
prior.rmin = 0; 
prior.rmax = 1; % constrain 0 < rho < 1
prior.rval = 4; % heteroscedastic prior (recommended approach)
result2 = far_gc(ydev,W,ndraw,nomit,prior);
disp('True value of rho = 0.7');
result2.tflag = 'tstat';
prt(result2,vnames);


subplot(211),
pltdens(result2.pdraw,0.1);
title('posterior density of rho heteroscedastic prior');
subplot(212),
pltdens(result.pdraw,0.1);
title('posterior density of rho homoscedastic prior');
pause;

subplot(211),
hist(result2.pdraw);
title('histogram of posterior for rho heteroscedastic prior');
subplot(212),
hist(result.pdraw);
title('histogram of posterior for rho homoscedastic prior');
pause;

subplot(111);

% test seed 
prior.seed = 102010;
result3 = far_gc(ydev,W,ndraw,nomit,prior);
prt(result3,vnames);
prior.seed = 201020;
result4 = far_gc(ydev,W,ndraw,nomit,prior);
prt(result4,vnames);
prior.seed = 102010;
result5 = far_gc(ydev,W,ndraw,nomit,prior);
prt(result5,vnames);
