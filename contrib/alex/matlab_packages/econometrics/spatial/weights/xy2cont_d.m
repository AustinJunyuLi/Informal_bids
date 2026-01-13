% PURPOSE: An example of using xy2cont()
%          creates a contiguity matrix from
%          latittude-longitude coordinates                  
%---------------------------------------------------
% USAGE: xy2cont_d 
%---------------------------------------------------

load anselin.dat;  % Columbus neighborhood crime
xc = anselin(:,5);  % longitude coordinate
yc = anselin(:,4);  % latittude coordinate
% create contiguity matrix from x-y coordinates
[W1 W2 W3] = xy2cont(xc,yc);
spy(W2,'ok'); 
