function res = myinv(X);

denom = X(1,1)*X(2,2) - X(1,2)*X(2,1);
res   = [X(2,2)/denom  -X(1,2)/denom; -X(2,1)/denom X(1,1)/denom];