function [wswdel,wwsdel,wmatdel]=xy2cont(xcoord,ycoord)
% PURPOSE: uses x,y coordinates to produce spatial contiguity weight matrices
%          with delaunay routine from MATLAB version 5.2
% ------------------------------------------------------
% USAGE: [w1 w2 w3] = xy2cont(xcoord,ycoord)
% where:     xcoord = x-direction coordinate
%            ycoord = y-direction coordinate
% ------------------------------------------------------
% RETURNS: 
%          w1 = W*S*W, a symmetric spatial weight matrix (max(eig)=1)
%          w2 = W*W*S, a row-stochastic spatial weight matrix
%          w3 = diagonal matrix with i,i equal to 1/sqrt(sum of ith row)
% ------------------------------------------------------
% References: Kelley Pace, Spatial Statistics Toolbox 1.0
% ------------------------------------------------------

% Written by Kelley Pace, 6/23/97 (named fdelw3)
% Documentation modified by J. LeSage 11/2002

n = length(xcoord);
tri=delaunay(xcoord,ycoord);
%finds triangulariztion
clear xcoord ycoord;

sa=(1:n)';
o=[ones(length(tri(:,1)),1);0];
bigma=spconvert([[tri(:,1:2);[n n]] o]);
s=bigma+bigma';
clear bigma;
bigmb=spconvert([[tri(:,2:3);[n n]] o]);
s=s+bigmb+bigmb';
clear bigmb;
bigmc=spconvert([[tri(:,[1 3]);[n n]] o]);
s=s+bigmc+bigmc';
clear bigmc;



%takes three of the six possible relations implied by the triangle
clear tri;
%[nbig,bigk]=size(bigm);
%o=ones(nbig,1);
%shalf=spconvert([bigm o]) ;
%converts to a sparse matrix
%clear bigm;

%s=shalf+shalf';
%uses symmetry to create other 3 possible relations
s=(s>0);
%converts to 0,1 matrix

srowsum=sum(s');
whalf=sqrt((1./srowsum)');
%computes normalization based on row sums

n=length(whalf);

wmatdel=spdiags(whalf,0,n,n);
%creates diagonal scaling matrix
wwsdel=wmatdel*wmatdel*s;
%creates row-stochastic weight matrix
wswdel=wmatdel*s*wmatdel;
%creates symmetric matrix with max eigenvalue of 1
%see Ord (JASA, 1975) for more on this scaling

