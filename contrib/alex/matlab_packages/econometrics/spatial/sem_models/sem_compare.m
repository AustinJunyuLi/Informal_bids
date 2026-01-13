% PURPOSE: An example of using sem_g() Gibbs sampling
%          spatial error model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: sem_compare (see also sem_gcd2 for a large data set)
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
sige = 1;
k = 3;
x = randn(n,k);
beta(1,1) = -1.0;
beta(2,1) = 0.0;
beta(3,1) = 1.0;

u = inv(IN-rho*W)*randn(n,1)*sqrt(sige); 
y = x*beta + u;

res = sem(y,x,W);
prt(res);


% Gibbs sampling function homoscedastic prior
prior.novi = 1;
% to maximum likelihood estimates
ndraw = 5500;
nomit = 500;

results = sem_g(y,x,W,ndraw,nomit,prior);
prt(results);

% this is the c-mex version
results2 = sem_gc(y,x,W,ndraw,nomit,prior);
prt(results2);

tt=1:n;

plot(tt,y,tt,results.yhat,'--',tt,results2.yhat,'-.');
title('actual y vs yhat');
legend('actual','matlab','c-mex');
pause;

[h1,f1,y1] = pltdens(results.pdraw);
[h2,f2,y2] = pltdens(results2.pdraw);
plot(y1,f1,'.g',y2,f2,'.r');
title('matlab-posterior density for rho');
legend('matlab','cmex');