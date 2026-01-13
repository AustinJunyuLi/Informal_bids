% PURPOSE: An example of using sdm_gc() Gibbs sampling
%          spatial durbin model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: sdm_gcd (see also sdm_gcd2 for a large data set)
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
x = [ones(n,1) anselin(:,2:3)];
y = anselin(:,1);
vnames = strvcat('crime','constant','income','hvalue');
info.rmin = 0;
info.rmax = 1;
res = sdm(y,x,W,info);
prt_sdm(res,vnames);

% Gibbs sampling function homoscedastic prior
prior.novi = 1; % homoscedastic prior for comparison
prior.rmin = 0;
prior.rmax = 1;

% to maximum likelihood estimates
ndraw = 2500;
nomit = 500;

% this is the matlab version
results = sdm_g(y,x,W,ndraw,nomit,prior);
prt_sdm(results,vnames);


prior2.novi = 1; % homoscedastic model with no vi-draws
prior2.rmin = 0;
prior2.rmax = 1;

results = sdm_gc(y,x,W,ndraw,nomit,prior2);
prt_sdm(results,vnames);


