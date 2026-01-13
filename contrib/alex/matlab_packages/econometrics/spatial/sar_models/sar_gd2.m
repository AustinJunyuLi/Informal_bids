% PURPOSE: An example of using sar_g() on a large data set   
%          Gibbs sampling spatial autoregressive model                         
%---------------------------------------------------
% USAGE: sar_gd2 (see sar_gd for a small data set)
%---------------------------------------------------

clear all;
% NOTE a large data set with 3107 observations
% from Pace and Barry, takes around 150-250 seconds
load elect.dat;             % load data on votes
y =  log(elect(:,7)./elect(:,8));
x1 = log(elect(:,9)./elect(:,8));
x2 = log(elect(:,10)./elect(:,8));
x3 = log(elect(:,11)./elect(:,8));
latt = elect(:,5);
long = elect(:,6);
n = length(y); x = [ones(n,1) x1 x2 x3];
clear x1; clear x2; clear x3;
clear elect;                % conserve on RAM memory
n = 3107;
[junk W junk] = xy2cont(latt,long);
vnames = strvcat('voters','const','educ','homeowners','income');

result = sar(y,x,W); % maximum likelihood estimates
prt(result,vnames);

% do Gibbs sampling estimation
ndraw = 1200; 
nomit = 200;
prior.novi = 1; % homoscedastic model
result2 = sar_g(y,x,W,ndraw,nomit,prior);
result2.tflag = 'tstat';
prt(result2,vnames);

prior2.rval = 4; % heteroscedastic model
result3 = sar_g(y,x,W,ndraw,nomit,prior2);
result3.tflag = 'tstat';
prt(result3,vnames);
plt(result3);
pause;

[h1,f1,y1] = pltdens(result2.pdraw);
[h2,f2,y2] = pltdens(result3.pdraw);
plot(y1,f1,'.r',y2,f2,'.g');
legend('homoscedastic','heteroscedastic');
title('posterior distribution for rho');
xlabel('rho values');

