% directory contents

% sdm_g.m  - matlab program to compute estimates for
%            Bayesian heteroscedastic spatial durbin model
% sdm_gc.m - matlab program that calls sdm_gcc.c mex file to do estimation
% compile with: mex sdm_gcc.c matrixjpl.c randomlib.c
% sdm_gcc.c - c-program source
% matrixjpl.c, matrixjpl.h support files
% randomlib.c, randomlib.h support files
%
% sdm_gd,  demo file using small dataset
% sdm_gd2, demo file using large dataset
% sdm_gcd, demo file using small dataset
% sdm_gcd2, demo file using large dataset
% sdm_gseed, demo file showing seed control for sdm_gc
% sdm_compare, demo file comparing c-language and matlab results

% sdmp_g.m    - matlab program to compute estimates
%              for Bayesian heteroscedastic spatial durbin probit model
% sdmp_gd,      demo file using a small dataset
% sdmp_gd2,     demo file using a large dataset


% sdmt_g.m    - matlab program to compute estimates
%              for Bayesian heteroscedastic spatial durbin tobit model
% sdmt_gd,      demo file using a small dataset
% sdmt_gd2      demo file using large dataset
