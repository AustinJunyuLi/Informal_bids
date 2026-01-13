rand('state',1234); randn('state',1234);

N_len = 250;
N = 1+randi(4,N_len,1);
X_len = sum(N);
%X = [randi(2,X_len,1)-1 normrnd(1,0.5,X_len,1)];
X = [randi(2,X_len,1)-1];
B = 1.3+normrnd(0,0.4,X_len,1);
%Z = B+0.1*X(:,1)+0.0*X(:,2)+normrnd(0,0.45,X_len,1);
Z = B+0.1*X(:,1)+normrnd(0,0.25,X_len,1);
Id = zeros(X_len,1);
Nid = zeros(X_len,1);
W = zeros(X_len,1);
N_k = 0;
for i = 1 : N_len
  Id(N_k+1:N_k+N(i)) = i;
  Nid(N_k+1:N_k+N(i)) = N(i);
  W(N_k+1:N_k+N(i)) = (Z(N_k+1:N_k+N(i))==max(Z(N_k+1:N_k+N(i))));
  N_k = N_k+N(i);
end
gpdata = [Id Nid B W X];