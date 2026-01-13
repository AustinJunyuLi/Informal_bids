% PURPOSE: An example of using sem_g() 
% Gibbs sampling spatial error model(on a small data set)  
%                                   
%---------------------------------------------------
% USAGE: sem_gd (see also sem_gd2 for a large data set)
%---------------------------------------------------

clear all;

% W-matrix from Anselin's neigbhorhood crime data set
load anselin.dat; 
xc = anselin(:,4);
yc = anselin(:,5);
% crate standardized 1st-order spatial weight matrix
[j1 W j2] = xy2cont(xc,yc);
[n junk] = size(W);
IN = eye(n); 
rho = 0.7;  % true value of rho
sige = 3;
k = 3;
x = randn(n,k);
beta(1,1) = 1.0;
beta(2,1) = -1.0;
beta(3,1) = 1.0;

u = inv(IN - rho*W)*randn(n,1)*sqrt(sige);
y = x*beta + u;


ndraw = 2500;
nomit = 500;

% maximum likelihood estimates
results0 = sem(y,x,W);
prt_sem(results0);

% this is the matlab version of the function, see sem_gc for a c-mex version
prior.novi = 1; % homoscedastic model for comparison with max lik
results2 = sem_g(y,x,W,ndraw,nomit,prior);
results2.tflag = 'tstat';
prt(results2);


tt=1:n;
plot(tt,results2.y,'o',tt,results2.yhat,'.r',tt,results0.yhat,'.g');
legend('actual','mcmc','maxlik');

