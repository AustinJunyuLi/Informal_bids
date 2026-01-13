function W = make_neighborsw(xc,yc,m)
% PURPOSE: finds the nth nearest neighbor and constructs
%          a spatial weight matrix based on this neighbor
% --------------------------------------------------------
% USAGE: W = make_neighborsw(xc,yc,nn)
%       where: 
%             xc = x-coordinate for each obs (nobs x 1)
%             yc = y-coordinate for each obs (nobs x 1)
%             nn = nth nearest neighbor to be used
% --------------------------------------------------------
% RETURNS: W an (nobs x nobs) spatial weight matrix based on the nth
%          nearest neighbor (a sparse matrix)
% --------------------------------------------------------
% NOTES: W takes a form such that: W*y would produce a vector
%        consisting of the values of y for the nth nearest neighbor
%        for each observation i in the (nobs x 1) vector y
% To construct a weight matrix based on neighbors 1,3,4
% (where 1 is the nearest neighbor, 3 is the 3rd nearest and so on)
% W1 = make_neighborsw(xc,yc,1);
% W3 = make_neighborsw(xc,yc,3);
% W4 = make_neighborsw(xc,yc,4);
% W = W1 + W3 + W4;
% --------------------------------------------------------
% SEE ALSO: find_neighbors() which finds an index to the observations
%           that are nearest neighbors, and find_nn()
% --------------------------------------------------------

% written by:
% James P. LeSage, 5/2002
% Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com


if nargin == 3
[n junk] = size(xc);    
else,
error('make_neighborsw: Wrong # of input arguments');
end;


nnlist = find_neighbors(xc,yc,m);

% convert the list into a row-standardized spatial weight matrix
rowseqs=(1:n)';
vals1=ones(n,1);
vals0=zeros(n,1);

for i=1:m;

colseqs=nnlist(:,i);
ind_to_keep=logical(colseqs>0);

z1=[rowseqs colseqs vals1];
z1=z1(ind_to_keep,:);

z2=[rowseqs rowseqs vals0];
%this last statement makes sure the dimensions are right
z=[z1
   z2];

end;

W = spconvert(z);

