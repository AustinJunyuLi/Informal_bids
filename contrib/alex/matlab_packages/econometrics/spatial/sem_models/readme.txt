% directory contents

% sem_g.m  - matlab program to compute estimates for
%            Bayesian heteroscedastic spatial autoregressive model
% sem_gc.m - matlab program that calls sem_gcc.c mex file to do estimation
% compile with: mex sem_gcc.c matrixjpl.c randomlib.c
% sem_gcc.c - c-program source
% matrixjpl.c, matrixjpl.h support files
% randomlib.c, randomlib.h support files
% sem_gcd, demo file
% sem_compare, demo file comparing c-language and matlab results


% semp_g.m    - matlab program to compute estimates
%              for Bayesian heteroscedastic spatial autoregressive probit model
% semp_gc.m   - matlab program that calls semp_gcc.c mex file to do estimation
% compile with: mex semp_gcc.c matrixjpl.c randomlib.c
% semp_gcc.c  - c-program source
% matrixjpl.c, matrixjpl.h are support files
% randomlib.c, randomlib.h are support files
% semp_gd,      demo file using a small dataset
% semp_gcd      demo file using cmex file
% semp_compare, demo file comparing c-language and matlab results


% semt_g.m    - matlab program to compute estimates
%              for Bayesian heteroscedastic spatial autoregressive tobit model
% semt_gc.m   - matlab program that calls semt_gcc.c mex file to do estimation
% compile with: mex semt_gcc.c matrixjpl.c randomlib.c
% semt_gcc.c  - c-program source
% matrixjpl.c, matrixjpl.h are support files
% randomlib.c, randomlib.h are support files
% semt_gd,      demo file using a small dataset
% semt_gcd      demo file using cmex file
% semt_compare, demo file comparing c-language and matlab results
