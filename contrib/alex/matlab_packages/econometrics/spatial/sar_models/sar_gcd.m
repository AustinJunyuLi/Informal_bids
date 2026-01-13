% PURPOSE: An example of using sar_gc() Gibbs sampling
%          spatial autoregressive model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: sar_gcd (see also sar_gcd2 for a large data set)
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

info.lflag = 0;
res0 = sar(y,x,W);
prt(res0);

% Gibbs sampling function homoscedastic prior
prior.novi = 1; % homoscedastic prior 
ndraw = 2500;
nomit = 500;

% this is the c-mex version
results = sar_gc(y,x,W,ndraw,nomit,prior);
results.tflag = 'tstat';
prt(results);



