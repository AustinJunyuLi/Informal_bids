% PURPOSE: An example of using sdm_gc() and sdm_g Gibbs sampling
%          spatial durbin model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: sdm_compare (see also sdm_gcd2 for a large data set)
%---------------------------------------------------

clear all;

% W-matrix from Anselin's neigbhorhood crime data set
load anselin.dat; % standardized 1st-order spatial weight matrix
y = anselin(:,1);
n = length(y);
x = [ones(n,1) anselin(:,2:3)];
vnames = strvcat('crime','constant','income','hvalue');

latt = anselin(:,4);
long = anselin(:,5);
[junk W junk] = xy2cont(latt,long);
[n junk] = size(W);

info.rmin = 0;
info.rmax = 1;
res = sdm(y,x,W,info);
prt(res,vnames);


% Gibbs sampling function homoscedastic prior
prior.rval = 100; % homoscedastic prior for comparison
prior.rmin = 0;
prior.rmax = 1;
% to maximum likelihood estimates
ndraw = 1200;
nomit = 200;

% this is the matlab version
results = sdm_g(y,x,W,ndraw,nomit,prior);
results.tflag = 'tstat';
prt(results,vnames);

% this is the c-mex version
results2 = sdm_gc(y,x,W,ndraw,nomit,prior);
results2.tflag = 'tstat';
prt(results2,vnames);

tt=1:n;

plot(tt,y,tt,results.yhat,'--',tt,results2.yhat,'-.');
title('actual y vs mean of predicted y-draws');
legend('actual','matlab predicted','c-mex predicted');

