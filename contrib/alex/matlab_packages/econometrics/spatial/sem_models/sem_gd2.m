% PURPOSE: An example of using sem_g()
%          Gibbs sampling spatial autoregressive model
%          on a large data set                    
%---------------------------------------------------
% USAGE: sem_gd2 (see sem_gd for a small data set)
%---------------------------------------------------

clear all;
% NOTE a large data set with 3107 observations
% from Pace and Barry, takes around 150-250 seconds
load elect.dat;             % load data on votes
y =  log(elect(:,7)./elect(:,8));
x1 = log(elect(:,9)./elect(:,8));
x2 = log(elect(:,10)./elect(:,8));
x3 = log(elect(:,11)./elect(:,8));
n = length(y); 
x = [ones(n,1) x1 x2 x3];
clear x1; clear x2; clear x3;
xc = elect(:,5);
yc = elect(:,6);
[j1 W j2] = xy2cont(xc,yc);
clear elect;                % conserve on RAM memory
n = 3107;
vnames = strvcat('voters','const','educ','homeowners','income');
info.lmin = 0;
info.lmax = 1;
res = sem(y,x,W,info);
prt(res,vnames);

% do Gibbs sampling estimation
ndraw = 1500; 
nomit = 500;
prior.novi = 1;
resg = sem_g(y,x,W,ndraw,nomit,prior);
resg.tflag = 'tstat';
prt(resg,vnames);

% these homoscedastic results should match max lik results
prior.rval = 4;
resg2 = sem_g(y,x,W,ndraw,nomit,prior);
resg2.tflag = 'tstat';
prt(resg2,vnames);

