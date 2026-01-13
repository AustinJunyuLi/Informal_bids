% PURPOSE: An example of using far_gc()
%          1st order spatial autoregressive model
%                              
%---------------------------------------------------
% USAGE: far_gcd2
%---------------------------------------------------

% NOTE a large data set with 3107 observations
% from Pace and Barry, takes around 150 seconds

clear all;
load elect.dat;             % load data on votes
y = log(elect(:,7)./elect(:,8)); % proportion of voters casting votes
ydev = y - mean(y);         % deviations from the means form 
latt = elect(:,5);
long = elect(:,6);
clear y;     % conserve on RAM memory
clear elect; % conserve on RAM memory
n = 3107;
[junk W junk] = xy2cont(latt,long);
% do maximum likelihood for comparison
info.lflag = 1; % use Pace and Barry MC approximation to ln det
result1 = far(ydev,W,info);
prt(result1);
% do Gibbs sampling
% with diffuse prior for rho, sige
ndraw = 1200;
nomit = 200;
% Gibbs sampling function
prior.rval = 200;  % homoscedastic prior
prior.lflag = 1; % use Pace and Barry MC approximation to ln det
%prior.m = 4;
%prior.k = 2;
result2 = far_gc(ydev,W,ndraw,nomit,prior);
prt(result2);

prior.rval = 4;  % heteroscedastic prior
% demo inappropriate prior on rho
% prior.prho = 0.2;
% prior.pvar = 0.01;

result3 = far_gc(ydev,W,ndraw,nomit,prior);
prt(result3);
plt(result3);
pause;

% compare posterior densities from homoscedastic and heteroscedastic
[h1,f1,y1] = pltdens(result2.pdraw);
[h2,f2,y2] = pltdens(result3.pdraw);

plot(y1,f1,'--r',y2,f2,'-g');
legend('homo','hetro');
