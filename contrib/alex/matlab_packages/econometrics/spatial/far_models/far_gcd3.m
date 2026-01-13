% PURPOSE: An example of using far_gc()
%          1st order spatial autoregressive model
%          on a Monte Carlo generated data set                   
%---------------------------------------------------
% USAGE: far_gcd3, (see also far_gcd for a small data set)
%---------------------------------------------------

clear all;
% load Anselin (1988) 1st order contiguity matrix
load anselin.dat;
xc = anselin(:,4);
yc = anselin(:,5);
[j1 W j2] = xy2cont(xc,yc);

[n junk] = size(W);

% generate FAR models

rho = -0.4:0.1:0.9;
sige = 1;
ndraw = 1500;
nomit = 500;
In = speye(n);
infot.lflag = 0;
ytmp = anselin(:,1);
res = far(ytmp,W,infot); % compute lndet only once
info.lndet = res.lndet;
prior.lndet = res.lndet;

save = zeros(length(rho),4);
for i=1:length(rho);
y = (In - rho(i)*W)\(randn(n,1)*sqrt(sige));
ydev = y - mean(y);
fprintf(1,'true value of rho = %16.8f \n',rho(i));
result0 = far(y,W,info);
prior.rval = 200;
result2 = far_gc(y,W,ndraw,nomit,prior);
save(i,1) = result0.rho;
save(i,2) = result0.tstat;
save(i,3) = mean(result2.pdraw);
save(i,4) = mean(result2.pdraw)./std(result2.pdraw);
end;

rnames = 'True rho';
for i=1:length(rho);
 rnames = strvcat(rnames,['rho =' num2str(rho(i))]);
end;

cnames = strvcat('max lik rho','max lik tstat','Bayes rho','Bayes tstat');


in.rnames = rnames;
in.cnames = cnames;
mprint(save,in);
