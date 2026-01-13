% PURPOSE: An example of using sem_gc) 
% Gibbs sampling spatial error model(on a small data set)  
%                                   
%---------------------------------------------------
% USAGE: sem_gcd (see also sem_gcd2 for a large data set)
%---------------------------------------------------

% W-matrix from Anselin's neigbhorhood crime data set
load anselin.dat; 
xc = anselin(:,4);
yc = anselin(:,5);
% crate standardized 1st-order spatial weight matrix
[j1 W j2] = xy2cont(xc,yc);
[n junk] = size(W);
IN = eye(n); 
rho = 0.7;  % true value of rho
sige = 2.0;
k = 3;
x = randn(n,k);
beta(1,1) = 1.0;
beta(2,1) = -1.0;
beta(3,1) = 1.0;

u = inv(IN - rho*W)*randn(n,1)*sqrt(sige);
y = x*beta + u;

results0 = sem(y,x,W);
prt(results0);

ndraw = 2500;
nomit = 500;
% Gibbs sampling function homoscedastic prior
prior.novi = 1; % homoscedastic model for comparison
                % to maximum likelihood estimates
% this is the c-mex version
results2 = sem_gc(y,x,W,ndraw,nomit,prior);
results2.tflag = 'tstat';
prt(results2);

