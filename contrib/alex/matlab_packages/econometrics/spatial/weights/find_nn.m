function nnlist = find_nn(xc,yc,m,W)
% PURPOSE: finds observations containing m nearest neighbors, fast but high memory version
%          to each observation and returns an index to these
%          neighboring observations
% --------------------------------------------------------
% USAGE: nnindex = find_nn(xc,yc,m,W)
%       where: 
%             xc = x-coordinate for each obs (nobs x 1)
%             yc = y-coordinate for each obs (nobs x 1)
%             m  = # of nearest neighbors to be found
%             W  = (optional) contiguity matrix from xy2cont()
%                  if omitted this will be constructed by the function
% --------------------------------------------------------
% RETURNS: an (nobs x m) matrix of indices to the m neighbors
% --------------------------------------------------------
% NOTES: nnindex takes a form such that: ind = nnindex(i,:)';
%        y(ind,1)
%        would pull out the nn nearest neighbor observations to
%        y(i,1), and y(ind,1)/m would represent an avg of these
% --------------------------------------------------------
% SEE ALSO: find_neighbors, find_nn2, and make_nnw which constructs a standardized spatial 
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
error('find_nn: Wrong # of input arguments');
end;

if nargin == 3
[junk W junk] = xy2cont(xc,yc);
end;

n = length(xc);

poslist=((W+W*W)>0)';

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

