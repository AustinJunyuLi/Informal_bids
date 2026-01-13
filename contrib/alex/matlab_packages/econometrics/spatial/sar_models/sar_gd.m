% PURPOSE: An example of using sar_g() Gibbs sampling
%          spatial autoregressive model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: sar_gd (see also sar_gd2 for a large data set)
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

result0 = sar(y,x,W);
prt(result0);

ndraw = 1200;
nomit = 200;
prior.novi = 1; % homoscedastic prior
results = sar_g(y,x,W,ndraw,nomit,prior);
prt(results);

prior.rval = 4; % heteroscedastic prior
results2 = sar_g(y,x,W,ndraw,nomit,prior);
prt(results2);


[h1,f1,y1] = pltdens(results.pdraw);
[h2,f2,y2] = pltdens(results2.pdraw);
plot(y1,f1,'.r',y2,f2,'.g');
legend('homoscedatic','heteroscedastic');
title('posterior distribution for rho');
pause;

% demonstrate plotting function
plt(results);


