% PURPOSE: An example of using sar_gc() and sar_g Gibbs sampling
%          spatial autoregressive model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: sar_compare (see also sar_gcd2 for a large data set)
%---------------------------------------------------

clear all;

% W-matrix from Anselin's neigbhorhood crime data set
load anselin.dat; % standardized 1st-order spatial weight matrix
latt = anselin(:,4);
long = anselin(:,5);
[junk W junk] = xy2cont(latt,long);
[n junk] = size(W);
IN = eye(n); 
rho = 0.7;  % true value of rho
sige = 2;
k = 3;
x = randn(n,k);
beta(1,1) = -1.0;
beta(2,1) = 0.0;
beta(3,1) = 1.0;

y = inv(IN-rho*W)*x*beta + inv(IN-rho*W)*randn(n,1)*sqrt(sige); 
ysave = y;

info.lflag = 0; % no log-determinant approximation
res = sar(ysave,x,W,info);
prt(res);


% Gibbs sampling function homoscedastic prior
prior.novi = 1;  % homoscedastic prior for comparison
prior.lflag = 0; % to maximum likelihood estimates

ndraw = 2500;
nomit = 500;

% this is the c-mex version
results = sar_g(y,x,W,ndraw,nomit,prior);
results.tflag = 'tstat'; % print t-statistics
prt(results);            % for comparison with maximum likelihood

results2 = sar_gc(y,x,W,ndraw,nomit,prior);
results2.tflag = 'tstat';
prt(results2);

tt=1:n;

plot(tt,ysave,tt,results.yhat,'--',tt,results2.yhat,'-.');
title('actual y vs mean of predicted y-draws');
legend('actual','matlab predicted','c-mex predicted');
pause;

[h1,f1,y1] = pltdens(results.pdraw);
[h2,f2,y2] = pltdens(results2.pdraw);
plot(y1,f1,'.g',y2,f2,'.r');
title('matlab-posterior density for rho');
legend('matlab','cmex');