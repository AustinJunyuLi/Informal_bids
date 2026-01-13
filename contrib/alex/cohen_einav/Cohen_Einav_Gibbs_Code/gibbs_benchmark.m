% This is called by a separate function 'loaddata' that prepares the data

lmax = 6; % truncation point of l
rmax = .5; % truncation point of r
eps = 10^(-10); % tollerance
llmax = log(lmax);
lrmax = log(rmax);

myX = myX(SampleInd,:);
choice = choice(SampleInd);
ratio  = ratio(SampleInd);
dbar   = dbar(SampleInd);
claims = claims(SampleInd);
exposure = exposure(SampleInd);

I = size(myX,1)

% start program - initialization
% ------------------------------

X = (myX - ones(I,1)*mean(myX)) ./ (ones(I,1)*std(myX)); % rescaling
X = [X ones(I,1)]; % adding a constant

Xr = X(:,Xr_ind==1);
Xl = X(:,Xl_ind==1);

kr = size(Xr,2);
kl = size(Xl,2);

Xblock = [Xl zeros(I,kr); zeros(I,kl) Xr];
invXXblock  = inv(Xblock'*Xblock);
invXXXblock = invXXblock*Xblock';
invXX = inv(X'*X);

beta_l = zeros(kl,1);
beta_r = zeros(kr,1); 
SIG    = [100 -90; -90 100];

l = 0.15*ones(I,1);
r = 0.0001*ones(I,1);

ll = log(l);
lr = log(r);

J = 100000;

betasave = zeros(J,kl+kr);
SIGsave  = zeros(J,3);
descsave = zeros(J,7);

% start loop
% ----------

for j = 1:J,
    
   j 
    
   SIG_l = SIG(1,1)^.5;
   SIG_r = SIG(2,2)^.5;
   rho   = SIG(1,2)/(SIG_l*SIG_r);

   betasave(j,:) = [beta_l' beta_r'];
   SIGsave(j,:)  = [SIG_l SIG_r rho];
   tempp = corrcoef(r,l);
   descsave(j,:) = [mean(l) median(l) std(l) mean(r) median(r) std(r) tempp(1,2)];
   
   % conditional for r

   mur   = Xr*beta_r + rho*(SIG_r/SIG_l)*(ll - Xl*beta_l);
   sigr  = SIG_r*(1-rho^2)^.5;

   rcutoff = ((ratio./l)-1)./dbar;   % not to get confused between 1 (one) and l (L)
   lrcut   = log(rcutoff);

   lower = -inf*ones(I,1);
   upper = inf*ones(I,1);   
   
   upper(choice~=1) = lrcut(choice~=1);
   lower(choice==1 & rcutoff>0) = lrcut(choice==1 & rcutoff>0);   

   a_prime = normcdf(lower-mur,0,sigr);
   b_prime = normcdf(upper-mur,0,sigr);
   u       = a_prime + rand(I,1).*(b_prime - a_prime);
   lr      = norminv(u,0,sigr) + mur;
   lr      = min([lr lrmax*ones(length(lr),1)],[],2);   
   r       = exp(lr);

   % conditional for u1, u2 (hirarchical in l)

   u1 = rand(I,1).*(l.^claims);
   u2 = rand(I,1).*exp(-l.*exposure);

   % conditional for l

   mul   = Xl*beta_l + rho*(SIG_l/SIG_r)*(lr - Xr*beta_r);
   sigl  = SIG_l*(1-rho^2)^.5;

   lcutoff = ratio./(r.*dbar+1);
   llcut   = log(lcutoff);

   warning off MATLAB:divideByZero;
   u1_bound = log(u1)./claims;
   u2_bound = log(-log(u2)./exposure);

   lower = -inf*ones(I,1);
   upper = inf*ones(I,1);   

   upper(choice==1) = u2_bound(choice==1);
   upper(choice~=1) = min([u2_bound(choice~=1) llcut(choice~=1)],[],2);   
   lower(choice==1 & claims>0)  = max([u1_bound(choice==1 & claims>0) llcut(choice==1 & claims>0)],[],2);   
   lower(choice==1 & claims==0) = llcut(choice==1 & claims==0);   
   lower(choice~=1 & claims>0)  = u1_bound(choice~=1 & claims>0);   

   a_prime = normcdf(lower-mul,0,sigl);
   b_prime = normcdf(upper-mul,0,sigl);
   u       = a_prime + rand(I,1).*(b_prime - a_prime);
   ll      = norminv(u,0,sigl) + mul;   
   ll      = min([ll llmax*ones(length(ll),1)],[],2);   
   l       = exp(ll);

   % conditional for SIG

   nu  = ll - Xl*beta_l; 
   eps = lr - Xr*beta_r;
   S   = [sum(nu.^2) sum(eps.*nu); sum(eps.*nu) sum(eps.^2)];
   D   = wishrnd(myinv(S),I-kr-kl);
   SIG = myinv(D);

   % conditional for beta

   sigpost = kron(SIG,invXX);
   temp2   = mvnrnd(zeros(1,2*k),sigpost);
   temp1   = invXXXblock*[ll; lr];

   for m = 1:k,
      if Xl_ind(m) == 1, 
         beta_l(sum(Xl_ind(1:m))) = temp1(sum(Xl_ind(1:m))) + temp2(m);
      end;
      if Xr_ind(m) == 1, 
         beta_r(sum(Xr_ind(1:m))) = temp1(kl+sum(Xr_ind(1:m))) + temp2(k+m);
      end;
   end;

end; % end of big loop

betasave2 = betasave;

betasave2(:,1:kl-1)       = betasave(:,1:kl-1)      ./(ones(J,1)*std(myX(:,[(Xl_ind(1:kl-1)==1)'])));
betasave2(:,kl+1:kl+kr-1) = betasave(:,kl+1:kl+kr-1)./(ones(J,1)*std(myX(:,[(Xr_ind(1:kl-1)==1)'])));

