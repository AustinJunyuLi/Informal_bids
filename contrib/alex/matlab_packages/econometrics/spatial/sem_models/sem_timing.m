% PURPOSE: A timing comparison of matlab vs. c-mex functions
%          spatial error models
%                              
%---------------------------------------------------
% USAGE: sem_timing
%---------------------------------------------------

clear all;
load anselin.dat; % small 49-observation dataset
xc = anselin(:,4);
yc = anselin(:,5);
y = anselin(:,1);
n = length(y);
x = [ones(n,1) anselin(:,2:3)];
[j1 W j2] = xy2cont(xc,yc);
vnames = strvcat('crime','constant','income','hvalue');

times = zeros(6,2);

ndraw = 2000;
nomit = 500;
prior.rval = 4;
result = sem_g(y,x,W,ndraw,nomit,prior); 
times(1,1) = result.time; % matlab version
result = sem_gc(y,x,W,ndraw,nomit,prior);
times(1,2) = result.time; % c-mex version

z = (y > mean(y)); % create 0,1 y-values
result = semp_g(z,x,W,ndraw,nomit,prior); 
times(2,1) = result.time; % matlab version
result = semp_gc(z,x,W,ndraw,nomit,prior);
times(2,2) = result.time; % c-mex version

y = studentize(y);
limit = mean(y);
yc = y;
ind = find(y < limit);
yc(ind,1) = 0; % censore values < the mean
result = semt_g(yc,x,W,ndraw,nomit,prior); 
times(3,1) = result.time; % matlab version
result = semt_gc(yc,x,W,ndraw,nomit,prior);
times(3,2) = result.time; % c-mex version

% NOTE a large data set with 3107 observations
% from Pace and Barry
load elect.dat;             % load data on votes
y =  log(elect(:,7)./elect(:,8));
x1 = log(elect(:,9)./elect(:,8));
x2 = log(elect(:,10)./elect(:,8));
x3 = log(elect(:,11)./elect(:,8));
latt = elect(:,5);
long = elect(:,6);
n = length(y); 
x = [ones(n,1) x1 x2 x3];
clear x1; clear x2; clear x3;
clear elect;                % conserve on RAM memory
n = 3107;
[junk W junk] = xy2cont(latt,long);
vnames = strvcat('voters','const','educ','homeowners','income');

result = sem_g(y,x,W,ndraw,nomit,prior);
times(4,1) = result.time;

result = sem_gc(y,x,W,ndraw,nomit,prior);
times(4,2) = result.time;

z = (y > mean(y)); % create 0,1 y-values
result = semp_g(z,x,W,ndraw,nomit,prior); 
times(5,1) = result.time; % matlab version
result = semp_gc(z,x,W,ndraw,nomit,prior);
times(5,2) = result.time; % c-mex version

y = studentize(y);
limit = mean(y);
yc = y;
ind = find(y < limit);
yc(ind,1) = 0; % censor values < the mean
result = semt_g(yc,x,W,ndraw,nomit,prior); 
times(6,1) = result.time; % matlab version
result = semt_gc(yc,x,W,ndraw,nomit,prior);
times(6,2) = result.time; % c-mex version


in.cnames = strvcat('matlab','c-mex');
in.rnames = strvcat('time in seconds','49 obs sem model','49 obs sem probit model', ...
    '49 obs sem tobit model','3,107 obs sem model','3,107 obs sem probit model', ...
    '3,107 obs sem tobit model');
fprintf('time for 2,000 draws \n');
mprint(times,in);

