function W = make_nnw(xc,yc,m,flg)
% PURPOSE: finds the nth nearest neighbor and constructs
%          a spatial weight matrix based on this neighbor
% --------------------------------------------------------
% USAGE: W = make_nnw(xc,yc,nn,flag)
%       where: 
%             xc = x-coordinate for each obs (nobs x 1)
%             yc = y-coordinate for each obs (nobs x 1)
%             nn = nth nearest neighbor to be used
%           flag = 1 to use find_nn function to determine neighbors
%                = 2 to use find_nn2 function to determine neighbors
%           defaults: flag = 1, which will produce fewer neighbors 
%                than flag = 2 (for large matrices or cases where you
%                     are looking for 30 neighbors say, use flag = 2)
% --------------------------------------------------------
% RETURNS: W an (nobs x nobs) spatial weight matrix based on the nth
%          nearest neighbor (a sparse matrix)
% --------------------------------------------------------
% NOTES: W takes a form such that: W*y would produce a vector
%        consisting of the values of y for the nth nearest neighbor
%        for each observation i in the (nobs x 1) vector y
% To construct a weight matrix based on neighbors 1,3,4
% (where 1 is the nearest neighbor, 3 is the 3rd nearest and so on)
% W1 = make_nnw(xc,yc,1);
% W3 = make_nnw(xc,yc,3);
% W4 = make_nnw(xc,yc,4);
% W = W1 + W3 + W4;
% --------------------------------------------------------
% SEE ALSO: find_nn() which finds an index to the observations
%           that are nearest neighbors, and find_nn2()
% --------------------------------------------------------

% written by:
% James P. LeSage, 1/2000
% Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com

% NOTE: this function draws on ideas from the Spatial Statistics Toolbox
%       by R. Kelley Pace (that I stole to construct this function)

if nargin == 3
flag = 1;
elseif nargin == 4
flag = flg;
else,
error('make_nnw: Wrong # of input arguments');
end;

[n junk] = size(xc);

if flag == 1
nnlist = find_nn(xc,yc,m);
elseif flag == 2
nnlist = find_nn2(xc,yc,m);
else
error('make_nnw: flag must = 1, or 2');
end;

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

