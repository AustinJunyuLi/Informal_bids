% PURPOSE: An example of using sdmp_g() Gibbs sampling
%          spatial durbin probit model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: sdmp_gd (see also sdmp_gd2 for a large data set)
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
k=2;
x = randn(n,k);
beta(1,1) = 1.0;
beta(2,1) = -1.0;

y = inv(IN-rho*W)*x*beta + inv(IN-rho*W)*randn(n,1)*sqrt(sige); 

z = (y > mean(y))*1.0; % avoid a logical vector

% Gibbs sampling function homoscedastic prior
prior.novi = 1; % homoscedastic prior for comparison
% to maximum likelihood estimates
ndraw = 2500;
nomit = 500;

% this is the matlab version, use sdm_gc for c-mex version
results = sdm_g(y,x,W,ndraw,nomit,prior);
prt(results);

prior.novi = 1;
results2 = sdmp_g(z,x,W,ndraw,nomit,prior);
prt(results2);


tt=1:n;
plot(tt,y,'-b',tt,results2.yhat,'-r',tt,results.yhat,'-g');
legend('actual','sdmp predicted','sdm predicted');
pause;

[ysort yind] = sort(z);

tt=1:n;
plot(tt,ysort,'or',tt,results2.yprob(yind,1),'.g');
