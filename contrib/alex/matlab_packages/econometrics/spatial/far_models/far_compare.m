% PURPOSE: An example of using far_g(), far_gc() Gibbs sampling
%          1st-order spatial autoregressive model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: far_compare
%---------------------------------------------------

clear all;

% W-matrix from Anselin's neigbhorhood crime data set
load anselin.dat; % standardized 1st-order spatial weight matrix
xc = anselin(:,4);
yc = anselin(:,5);
[j1 W j2] = xy2cont(xc,yc);
[n junk] = size(W);
In = speye(n);
rho = 0.6;  % true value of rho
sige = 1;
y = (In - rho*W)\(randn(n,1)*sqrt(sige)); 
ydev = y - mean(y);
vnames = strvcat('y-simulated','y-spatial lag');

% do maximum likelihood for comparison    
info.lflag = 0;
res = far(ydev,W,info);
disp(['True value of rho = ' num2str(rho)]);
prt(res,vnames);

ndraw = 2500;
nomit = 500;

% Gibbs sampling function homoscedastic prior
prior.lflag = 0;
prior.rval = 200; % homoscedastic prior for comparison
                 % to maximum likelihood estimates
                 
% call matlab function                 
result = far_g(ydev,W,ndraw,nomit,prior); % matlab function
disp(['True value of rho = ' num2str(rho)]);
result.tflag = 'tstat';
prt(result,vnames);

% call c-mex function
result2 = far_gc(ydev,W,ndraw,nomit,prior); % cmex function
disp('True value of rho = 0.7');
result2.tflag = 'tstat';
prt(result2,vnames);

% plot posterior densities for comparison
[y1,f1,h1] = pltdens(result2.pdraw);
[y2,f2,h2] = pltdens(result.pdraw);
plot(h1,f1,'.r',h2,f2,'.g');
title('posterior density of rho matlab vs. c-mex');
legend('c-mex','matlab');
