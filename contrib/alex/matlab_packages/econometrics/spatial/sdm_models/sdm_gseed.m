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
info.lflag = 0;
res = sdm(y,x,W,info);
prt_spat(res,vnames);

% Gibbs sampling function homoscedastic prior
prior.rval = 4; % heteroscedastic prior for comparison
%prior.novi = 1; % homoscedastic prior
prior.lflag = 0;  % must turn off lndet approximation
                  % to get identical results
ndraw = 2500;
nomit = 500;

prior.seed = 101020;
results = sdm_gc(y,x,W,ndraw,nomit,prior);
prt(results,vnames);

prior.seed = 101020;
results2 = sdm_gc(y,x,W,ndraw,nomit,prior);
prt(results2,vnames);


[h1,f1,y1] = pltdens(results.pdraw);
[h2,f2,y2] = pltdens(results2.pdraw);
plot(y1,f1,'.r',y2,f2,'.g');



