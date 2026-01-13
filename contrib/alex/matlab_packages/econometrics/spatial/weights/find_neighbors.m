function nnlist = find_neighbors(xc,yc,m)
% PURPOSE: finds observations containing m nearest neighbors, slow but low memory version
%          to each observation and returns an index to these
%          neighboring observations
% --------------------------------------------------------
% USAGE: nnindex = find_neighbors(xc,yc,m)
%       where: 
%             xc = x-coordinate for each obs (nobs x 1)
%             yc = y-coordinate for each obs (nobs x 1)
%             m  = # of nearest neighbors to be found
% --------------------------------------------------------
% RETURNS: an (nobs x m) matrix of indices to the m neighbors
% --------------------------------------------------------
% NOTES: nnindex takes a form such that: ind = nnindex(i,:)';
%        y(ind,1)
%        would pull out the nn nearest neighbor observations to
%        y(i,1), and y(ind,1)/nn would represent an avg of these
%   ---> This function will is the same as find_nn, but uses less
%        memory and takes more time. If you run out of memory using
%        find_nn, try this function
% --------------------------------------------------------
% SEE ALSO: find_nn.m, and make_nnw which constructs a standardized spatial 
%           weight matrix based on nearest neighbors
% --------------------------------------------------------

% written by:
% James P. LeSage, 12/2001
% Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com

% NOTE: this is a truly slow hack, but doesn't require much RAM memory

if nargin ~= 3
error('find_neighbors: Wrong # of input arguments');
end;

n = length(xc);

nnlist = zeros(n,m);

for i=1:n;
    xi = xc(i,1);
    yi = yc(i,1);
dist = sqrt((xc - xi*ones(n,1)).^2 + (yc - yi*ones(n,1)).^2);
[xds xind] = sort(dist);
nnlist(i,1:m) = xind(2:m+1,1)';
end;


