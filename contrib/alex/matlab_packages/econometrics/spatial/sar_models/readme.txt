% directory contents

% sar_g.m  - matlab program to compute estimates for
%            Bayesian heteroscedastic spatial autoregressive model
% sar_gc.m - matlab program that calls sar_gcc.c mex file to do estimation
% compile with: mex sar_gcc.c matrixjpl.c randomlib.c
% sar_gcc.c - c-program source
% matrixjpl.c, matrixjpl.h support files
% randomlib.c, randomlib.h support files
% sar_gcd, demo file
% sar_compare, demo file comparing c-language and matlab results
% sar_gseed.m, demo of the seed function for sar_gc.m
% sar_timing,  a comparison of matlab vs. c-mex function times

% sarp_g.m    - matlab program to compute estimates
%              for Bayesian heteroscedastic spatial autoregressive probit model
% sarp_gd,      demo file using a small dataset
% sarp_gd2      demo file using a large dataset

% sart_g.m    - matlab program to compute estimates
%              for Bayesian heteroscedastic spatial autoregressive tobit model
% sart_gd,      demo file using a small dataset
% sart_gd2      demo file using a large dataset
