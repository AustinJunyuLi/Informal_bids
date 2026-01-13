% PURPOSE: An example of using sdm_gc() on a large data set
%          Bayesian heteroscedastic spatial Durbin model                              
%---------------------------------------------------
% USAGE: sdm_gcd2 (see sdm_gcd for a small data set)
%---------------------------------------------------

clear all;
% NOTE a large data set with 3107 observations from Pace and Barry
load elect.dat;             % load data on votes
y =  (elect(:,7)./elect(:,8));
x1 = log(elect(:,9)./elect(:,8));
x2 = log(elect(:,10)./elect(:,8));
x3 = log(elect(:,11)./elect(:,8));
n = length(y); x = [ones(n,1) x1 x2 x3];
xc = elect(:,5);
yc = elect(:,6);
clear x1; clear x2; clear x3;
clear elect;                % conserve on RAM memory
[j1 W j2] = xy2cont(xc,yc);

n = 3107;
vnames = strvcat('voters','const','educ','homeowners','income');

% Gibbs sampling function homoscedastic prior
prior.rval = 100; % homoscedastic prior 
prior.rmin = 0;
prior.rmax = 1;
ndraw = 1200;
nomit = 200;

% this is the c-mex version
results = sdm_gc(y,x,W,ndraw,nomit,prior);
results.tflag = 'tstat';
prt(results,vnames);

