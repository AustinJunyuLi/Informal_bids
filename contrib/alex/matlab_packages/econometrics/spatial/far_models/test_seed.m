% PURPOSE: An example of using far_gc() Gibbs sampling
%          with random number seed option
%          (on a small data set)                  
%---------------------------------------------------
% USAGE: test_seed 
%---------------------------------------------------

clear all;
% W-matrix from Anselin's neigbhorhood crime data set
load anselin.dat; % standardized 1st-order spatial weight matrix
xc = anselin(:,4);
yc = anselin(:,5);
[j1 W j2] = xy2cont(xc,yc);
[n junk] = size(W);
IN = eye(n); 
rho = 0.7;  % true value of rho
sige = 5;
y = inv(IN-rho*W)*randn(n,1)*sqrt(sige); 
ydev = y - mean(y);
vnames = strvcat('y-simulated','y-spatial lag');

% do maximum likelihood for comparison 
info.lflag = 0; % no lndet approximation
res = far(ydev,W,info);
disp('True value of rho = 0.7');
prt(res,vnames);

ndraw = 3000;
nomit = 1000;

% Gibbs sampling function homoscedastic prior
prior.rval = 200; % homoscedastic prior for comparison
                  % to maximum likelihood estimates
prior.lflag = 0;  % must set this to avoid lndet mc approximation
                  % if you want identical results

% this is the c-mex version
prior.seed = 101020;
res3 = far_gc(ydev,W,ndraw,nomit,prior);
fprintf(1,'seed set to %10d \n',prior.seed);
prt(res3);
res4 = far_gc(ydev,W,ndraw,nomit,prior);
fprintf(1,'seed set to %10d \n',prior.seed);
prt(res4);
prior.seed = 2222111;
res5 = far_gc(ydev,W,ndraw,nomit,prior);
fprintf(1,'seed set to %10d \n',prior.seed);
prt(res5);


[h1,f1,y1] = pltdens(res3.pdraw);
[h2,f2,y2] = pltdens(res4.pdraw);
[h3,f3,y3] = pltdens(res5.pdraw);

plot(y1,f1,'.r',y2,f2,'.b',y3,f3,'.g');
title('posterior density for rho from 2 runs');
xlabel('rho values');
ylabel('kernel density estimates');
legend('seed=101020','seed=101020','seed=2222111');
