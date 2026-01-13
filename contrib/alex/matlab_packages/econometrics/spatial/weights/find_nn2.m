function nnlist = find_nn2(xc,yc,m,W)
% PURPOSE: finds observations containing nn nearest neighbors, high-order search
%          to each observation and returns an index to these
%          neighboring observations
%      --->This function to third and fourth order delauney triangles<---
%          (see find_nn for a function that picks nearest neighbors
%           based on delauney triangles or triangles of those triangles)
% --------------------------------------------------------
% USAGE: nnindex = find_nn2(xc,yc,nn,W)
%       where: 
%             xc = x-coordinate for each obs (nobs x 1)
%             yc = y-coordinate for each obs (nobs x 1)
%             nn = # of nearest neighbors to be found
%              W = (optional) contiguity matrix from xy2cont()
%                  if omitted this will be constructed by the function
% --------------------------------------------------------
% RETURNS: an (nobs x nn) matrix of indices to the nn neighbors
% --------------------------------------------------------
% NOTES: nnindex takes a form such that: ind = nnindex(i,:)';
%        y(ind,1)
%        would pull out the nn nearest neighbor observations to
%        y(i,1), and y(ind,1)/nn would represent an avg of these
%   ---> This function will find more higher order neighbors than 
%        the function find_nn, which doesn't use as high an order
%        of search as this function.
% --------------------------------------------------------
% SEE ALSO: find_nn, find_neighbors, and make_nnw which constructs a standardized spatial 
%           weight matrix based on nearest neighbors
% --------------------------------------------------------

% written by:
% James P. LeSage, Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com

% NOTE: this function draws on ideas from the Spatial Statistics Toolbox
%       by R. Kelley Pace (that I stole to construct this function)

if nargin < 3 | nargin > 4
error('find_nn2: Wrong # of input arguments');
end;

if nargin == 3
[junk W junk] = xy2cont(xc,yc);
end;

n = length(xc);
W2 = W*W;
poslist=((W+W2+W2*W+W2*W2)>0)';

m1=m+1;
nnlist=zeros(n,m1);
nnseq=(1:n)';
for i=1:n
   plist=logical(poslist(:,i));
   nns=nnseq(plist);
   n_del_neighbors=length(nns);
   d=(xc(plist)-xc(i)).^2+(yc(plist)-yc(i)).^2;
   [ds,dind]=sort(d);
   if n_del_neighbors<m1;
      nnlist(i,1:n_del_neighbors)=nns(dind)';
    else;
      nnlist(i,:)=nns(dind(1:m1))';
   end;
end;

nnlist=nnlist(:,2:m1);

