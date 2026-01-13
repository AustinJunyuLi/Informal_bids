% PURPOSE: An example of using sdm_g() Gibbs sampling
%          spatial durbin model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: sdm_gd (see also sdm_gd2 for a large data set)
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
k = 2;
x = [ones(n,1) randn(n,k)];
xmat = [x W*x(:,2:end)];
[junk nk] = size(xmat);
beta = ones(nk,1);
y = inv(IN - rho*W)*xmat*beta + inv(IN - rho*W)*randn(n,1)*sqrt(sige);

vnames = strvcat('crime','constant','income','hvalue');
res = sdm(y,x,W);
prt(res);

% Gibbs sampling function homoscedastic prior
prior.rval = 200; % homoscedastic prior for comparison
% to maximum likelihood estimates
ndraw = 2500;
nomit = 500;

% this is the matlab version, use sdm_gc for c-mex version
results = sdm_g(y,x,W,ndraw,nomit,prior);
prt(results,vnames);

prior.novi = 1;
results = sdm_g(y,x,W,ndraw,nomit,prior);
prt(results,vnames);

