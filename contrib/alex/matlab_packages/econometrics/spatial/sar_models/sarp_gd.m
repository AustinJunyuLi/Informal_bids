% PURPOSE: An example of using sarp_g() Gibbs sampling
%          spatial autoregressive probit model
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: sarp_gd (see also sarp_gd2 for a large data set)
%---------------------------------------------------

clear all;

% W-matrix from Anselin's neigbhorhood crime data set
load anselin.dat; % standardized 1st-order spatial weight matrix
latt = anselin(:,4);
long = anselin(:,5);
[junk W junk] = xy2cont(latt,long);
[n junk] = size(W);
IN = eye(n); 
rho = 0.6;  % true value of rho
sige = 1;
k = 4;

randn('seed',221020);
x = randn(n,k);
beta(1,1) = -1.0;
beta(2,1) = -1.0;
beta(3,1) = 1.0;
beta(4,1) = 1.0;

y = inv(IN-rho*W)*x*beta + inv(IN-rho*W)*randn(n,1)*sqrt(sige); 

res = sar(y,x,W);
prt(res);

z = (y > 0);
z = ones(n,1).*z; % eliminate a logical vector

% Gibbs sampling function homoscedastic prior
% to maximum likelihood estimates
ndraw = 1200;
nomit = 200;
prior.novi = 1; % homoscedastic prior
%prior.rval = 4;% heteroscedastic prior
results = sarp_g(z,x,W,ndraw,nomit,prior);
results.tflag = 'tstat';
prt(results);

% sort by 0,1 values
[ys yind] = sort(y);
zprob = results.yprob(yind,1);
zact = z(yind,1);

tt=1:n;
plot(tt,zact,'+',tt,zprob,'o');
title('predicted probabilities vs 0,1 values');
pause;

plot(tt,y,tt,results.yhat,'--');
title('actual y vs predicted values');
legend('actual','predicted');
pause;

plot(tt,y,tt,results.ymean,'--');
title('actual y vs mean of latent y-draws');
legend('actual','mean of y-draws');




