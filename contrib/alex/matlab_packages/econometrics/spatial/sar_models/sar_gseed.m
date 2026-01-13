% PURPOSE: An example of using sar_gc() with a
%          random number seed
%---------------------------------------------------
% USAGE: sar_gseed 
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

% Gibbs sampling function homoscedastic prior
% to maximum likelihood estimates
ndraw = 2500;
nomit = 500;

prior.seed = 11223344;
prior.novi = 1;
prior.lflag = 0; % must set this to avoid lndet approximation
                 % if you want identical results
results = sar_gc(y,x,W,ndraw,nomit,prior);
prt(results);

prior2.seed = 11223344;
prior2.novi = 1;
prior2.lflag = 0; % must set this to avoid lndet approximation
                  % if you want identical results
results2 = sar_gc(y,x,W,ndraw,nomit,prior2);
prt(results2);



