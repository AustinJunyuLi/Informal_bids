% PURPOSE: An example of using sarp_g() on a large data set   
%          Gibbs sampling spatial autoregressive probit model                         
%---------------------------------------------------
% USAGE: sarp_gd2 (see sarp_gd for a small data set)
%---------------------------------------------------

clear all;
% NOTE a large data set with 3107 observations
% from Pace and Barry, 
load elect.dat;             % load data on votes
latt = elect(:,5);
long = elect(:,6);
n = length(latt);
k = 4;
x = randn(n,k);
clear elect;                % conserve on RAM memory
n = 3107;
[junk W junk] = xy2cont(latt,long);
vnames = strvcat('voters','const','educ','homeowners','income');

rho = 0.7;
beta = ones(k,1);
beta(1:2,1) = -1;

y = (speye(n) - rho*W)\(x*beta) + (speye(n) - rho*W)\randn(n,1);
ysave = y;

ndraw = 2500; 
nomit = 500;

prior.novi = 1;
res = sar_gc(ysave,x,W,ndraw,nomit,prior);
prt(res,vnames);

%prior2.rval = 400;
prior2.novi = 1;
y = (y > 0); % convert to 0,1 y-values
resg = sarp_g(y,x,W,ndraw,nomit,prior2);
prt(resg,vnames);


plot(resg.pdraw);
title('rho draws');
pause;
plot(resg.bdraw);
title('beta draws');