% directory contents

% far_g.m  - matlab program to compute estimates for 1st-order spatial
%            autoregressive model
% far_gc.m - matlab program that calls far_gcc.c mex file to do estimation
% compile with: mex far_gcc.c matrixjpl.c randomlib.c
% far_gcc.c - c-program source for a mex file
% matrixjpl.c, matrixjpl.h support files
% randomlib.c, randomlib.h support files
% far_d,   far demo files
% far_d2,
%
% far_gd,
% far_gd2, far_g demo files
%
% far_gcd, far_gc demo files
% far_gcd2,
% far_gcd3
%
% test_seed.m, demonstrates using the seed
%              NOTE that due to rounding errors
%              from the single-precision c-mex routines
%              we don't get perfect control, about 5-decimal digits
% far_compare.m, compares c-mex output to matlab function output
%
% far_timing.m, compares c-mex and matlab run times
%               on a small and large problem