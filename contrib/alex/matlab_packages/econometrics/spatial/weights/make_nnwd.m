% PURPOSE: An example of using make_nnw()
%          a nearest neighbor spatial weight matrix
%          on a small data set                   
%---------------------------------------------------
% USAGE: make_nnwd
%---------------------------------------------------

% load Anselin (1988) Columbus neighborhood crime data
load anselin.dat; 
xc = anselin(:,4);
yc = anselin(:,5);
% To construct a weight matrix based on neighbors 1,3,4
% (where 1 is the nearest neighbor, 3 is the 3rd nearest and so on)
% W1 = make_nnw(xc,yc,1);
% W3 = make_nnw(xc,yc,3);
% W4 = make_nnw(xc,yc,4);
% W = W1 + W3 + W4;

W1 = make_nnw(xc,yc,1);

spy(W1,'.g');
title('nearest neighbor matrix');
pause;
hold on;

W2 = make_nnw(xc,yc,2);

spy(W2,'.r');
title('second nearest neighbor matrix');
pause;

W = normw(W1+W2);

spy(W,'oc');
title('1st plus 2nd nearest neighbors matrix');

hold off;

