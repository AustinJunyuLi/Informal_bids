% PURPOSE: An example of using sar_gc() on a large data set   
%          Gibbs sampling spatial autoregressive model                         
%---------------------------------------------------
% USAGE: sar_gcd2 (see sar_gcd for a small data set)
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
result = sar(y,x,W);
prt(result,vnames);


% do Gibbs sampling estimation
ndraw = 2500; 
nomit = 500;
prior.rval = 4; % heteroscedastic model

result2 = sar_g(y,x,W,ndraw,nomit,prior);
prt(result2,vnames);

prior.rval = 4; % heteroscedastic model
result3 = sar_gc(y,x,W,ndraw,nomit,prior);
prt(result3,vnames);

% compare posterior densities from homoscedastic and heteroscedastic
[h1,f1,y1] = pltdens(result2.pdraw);
[h2,f2,y2] = pltdens(result3.pdraw);

plot(y1,f1,'--r',y2,f2,'-g');
legend('homo','hetro');

