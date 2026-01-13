simGP_Aug2025;

rand('state',10); randn('state',10); % reset the random number generators

T_bi = 5000; T_gen = 50000; % number of burn-in and main drawing simulations

gpdata = sortrows(gpdata,[1 -4]);

Id = gpdata(:,1); % auction id. typically, this is SDC id
Nid = gpdata(:,2); % number of bidders in an auction
B = gpdata(:,3); % bid/market value of the target
W = gpdata(:,4); % winner 0/1 indicator
X = gpdata(:,5:end); % explanatory variables and/or additional deal terms

X_len = length(Id); % number of bids across all auctions
X_dim = size(X,2); % size of X
N = length(unique(Id)); % number of auctions

sigma = 0.15; rho = 0.0; % prior on var-cov matrix

invXX = inv(X'*X); % useful for posterior of beta formed via OLS
invXXX = invXX*X'; % useful for posterior of beta formed via OLS

% --- GIBBS SAMPLER ---
% define the truncated (from above, below, interval) normal density as a function of mu, sigma, and truncation bounds

randntruncd = @(mu_t,si_t,db_t) mu_t+si_t.*norminv(rand(length(mu_t),1).*(1-normcdf((db_t-mu_t)./si_t))+normcdf((db_t-mu_t)./si_t)); % transformation between Uniform and Normal, see e.g. Greene (2003). truncation from below
randntruncu = @(mu_t,si_t,ub_t) mu_t+si_t.*norminv(rand(length(mu_t),1).*normcdf((ub_t-mu_t)./si_t)); % truncation from above
randntrunci = @(mu_t,si_t,db_t,ub_t) mu_t+si_t.*norminv(rand(length(mu_t),1).*(normcdf((ub_t-mu_t)./si_t)-normcdf((db_t-mu_t)./si_t))+normcdf((db_t-mu_t)./si_t)); % truncation from both ends

tic

Z_tmp = zeros(X_len,1); % vector of valuations for each bidder across all auctions for the current simulation
Z = zeros(T_gen,X_len); % vectors of valuations across all main simulations
beta = zeros(T_gen,X_dim); % vectors of betas the size of X across all main simulations
Si = zeros(T_gen,1); % sigmas across all main simulations. the size has to change to (T_gen,3) -- two variances and covariance -- when bidder's valuations are incorporated
beta_tmp = zeros(X_dim,1); % vector of betas the size of X for the current simulation
Si_tmp = sigma; %for now, errors in GP valuations are uncorrelated 

for i = -T_bi : T_gen % for each burn-in drawing (-T_bi to zero), then for each main drawing (1 to T_gen):
  
  if mod(i,10000)==0, disp(num2str(i)); end % display progress each 10000 iterations

  N_cum = 0; % starting data row

   for k = 1 : N % for each auction:
  
      N_k = Nid(N_cum+1); % number of bidders in auction k
      N_cumnext = N_cum+N_k; % keep track of the last data row corresponding to auction k
   
            %b_gibbs = zeros(N_k-1,N_k); % is there a more elegant way to do this? % (SIG12/sqrt(SIG1*SIG2))*sqrt(SIG1)/sqrt(SIG2), when bidder's valuations are incorporated % calculate projection coefficients: (1) beta of the conditional mean
      % use the current conditional var-cov matrix
      se_gibbs = sqrt(Si_tmp); 
      % se_gibbs = sqrt(SIG1)*(1-(SIG12/sqrt(SIG1*SIG2))^2)^0.5, when bidder's valuations are incorporated
      
      indwin = N_cum+1;
      indlose = N_cum+[2:N_k]';
            %Zc_win = Z_tmp(indlose)-B(indlose)-X(indlose,:)*beta_tmp;
            %Z_tmp(indwin) = randntruncd(B(indwin)+X(indwin,:)*beta_tmp+b_gibbs(:,indwin)'*Zc_win,se_gibbs,max([1; Z_tmp(indlose)])); % draw the winning valuation with beta'*(centered Z_tmp(others)) as conditional mean and se_gibbs as conditional s.e.
      Z_tmp(indwin) = randntruncd(B(indwin)+X(indwin,:)*beta_tmp,se_gibbs,max([1; Z_tmp(indlose)])); % draw the winning valuation with beta'*(centered Z_tmp(others)) as conditional mean and se_gibbs as conditional s.e.
            %Zc_lose = ((Z_tmp([indwin;indlose])-B([indwin;indlose])-X([indwin;indlose],:)*beta_tmp)*ones(1,N_k)); 
            %Zc_lose([1:1:N_k, 1:N_k+1:N_k^2]) = []; 
            %Zc_lose = reshape(Zc_lose,N_k-1,N_k-1); %prepare centered Z_tmp(others) for losing valuations
      Z_tmp(indlose) = randntruncu(B(indlose)+X(indlose,:)*beta_tmp,se_gibbs,ones(N_k-1,1)*Z_tmp(indwin)); % draw the losing valuations. Q: is there a more efficient way to compute this without deleting elements and taking diag?
            %Z_tmp(indlose) = randntrunci(B(indlose)+X(indlose,:)*beta_tmp+diag(b_gibbs(:,indlose)'*Zc_lose),se_gibbs,0.5*ones(N_k-1,1),ones(N_k-1,1)*Z_tmp(indwin)); % draw the losing valuations
            %Z_tmp(indlose) = randntrunci(B(indlose)+X(indlose,:)*beta_tmp,se_gibbs,0.5*ones(N_k-1,1),ones(N_k-1,1)*Z_tmp(indwin)); % draw the losing valuations
      
      N_cum = N_cumnext;
      
   end
   
   if (i >= 1), % save main drawings
      Z(i,:) = Z_tmp'; 
      beta(i,:) = beta_tmp';
      Si(i)  = Si_tmp; % [SIG1_tmp SIG2_tmp rho];
   end

   eps = Z_tmp-B-X*beta_tmp; % conditional for Sigma
   SE_tmp = sum(eps.^2);
   D_tmp = wishrnd(inv(SE_tmp),X_len-1); % when bidder's valuations are incorporated: wishrnd(inv(Si_tmp),mu_n-dimX-dimY);
   Si_tmp = inv(D_tmp);

   si_post = kron(Si_tmp,invXX);
   temp1 = invXXX*(Z_tmp-B);
   temp2 = mvnrnd(zeros(1,X_dim),si_post);
   for j = 1 : X_dim,
     beta_tmp(j) = temp1(j)+temp2(j); %OLS beta
   end
   
end

%mean(Z), cov(Z), mean(beta), 
disp('structural model');
[mean(beta)' quantile(beta,0.025)' quantile(beta,0.975)'], mean(sqrt(Si)),

toc

% plot some comparative distribution (just for 1st auction) and estimates graphs
leg_arr = cell(Nid(1),1); for i = 1 : Nid(1), leg_arr{i} = num2str(i); end
figure(1); subplot(3,1,1); hist(Z(:,1:Nid(1)),100); legend(leg_arr);
subplot(3,1,2); for i = 1 : X_dim, hist(beta,100); end;
subplot(3,1,3); hist(sqrt(Si),100);

%OLS regressions
ols_noconst = ols(W-B,X);
ols_const = ols(W-B,[ones(size(W)) X]);

disp('OLS without constant');
[ols_noconst.beta ols_noconst.beta+norminv(0.025)*ols_noconst.bstd ols_noconst.beta+norminv(0.975)*ols_noconst.bstd],
disp('OLS with constant');
[ols_const.beta ols_const.beta+norminv(0.025)*ols_const.bstd ols_const.beta+norminv(0.975)*ols_const.bstd],