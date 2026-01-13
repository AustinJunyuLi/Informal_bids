% PURPOSE: An example of using sem_gc()
%          Gibbs sampling spatial autoregressive model
%          on a large data set                    
%---------------------------------------------------
% USAGE: sem_gcd2 (see sem_gcd for a small data set)
%---------------------------------------------------

clear all;
% NOTE a large data set with 3107 observations
% from Pace and Barry, takes around 150-250 seconds
load elect.dat;             % load data on votes
y =  log(elect(:,7)./elect(:,8));
x1 = log(elect(:,9)./elect(:,8));
x2 = log(elect(:,10)./elect(:,8));
x3 = log(elect(:,11)./elect(:,8));
n = length(y); x = [ones(n,1) x1 x2 x3];
clear x1; clear x2; clear x3;
xc = elect(:,5);
yc = elect(:,6);
[j1 W j2] = xy2cont(xc,yc);
clear elect;                % conserve on RAM memory
n = 3107;
vnames = strvcat('voters','const','educ','homeowners','income');
info.rmin = 0; 
info.rmax = 1;
info.lflag = 1; % use MC lndet approximation
res = sem(y,x,W,info);
prt(res,vnames);

% do Gibbs sampling estimation
ndraw = 5500; 
nomit = 500;
prior.novi = 1;
% these homoscedastic results should match max lik results
resg = sem_g(y,x,W,ndraw,nomit,prior);
resg.tflag = 'tstat';
prt(resg,vnames);

resg2 = sem_gc(y,x,W,ndraw,nomit,prior);
resg2.tflag = 'tstat';
prt(resg2,vnames);


[h1,f1,y1] = pltdens(resg.pdraw,0.015);
[h2,f2,y2] = pltdens(resg2.pdraw,0.015);
plot(y1,f1,'.r',y2,f2,'.g');





