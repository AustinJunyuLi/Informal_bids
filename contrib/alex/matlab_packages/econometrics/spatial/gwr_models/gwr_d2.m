% PURPOSE: An example of using gwr()
%          Geographically weighted regression model
%          (on a fairly large data set)                  
%---------------------------------------------------
% USAGE: gwr_d2 
%---------------------------------------------------

load boston.dat; % Harrison-Rubinfeld data
[n k] = size(boston);
y = boston(:,k-2);     % median house values
latit = boston(:,k-1);  % lattitude coordinates
longi = boston(:,k);    % longitude coordinates

x = [ones(n,1) boston(:,1:k-3)];       % other variables
vnames = strvcat('hprice','crime','zoning','industry','charlesr', ...
         'noxsq','rooms2','houseage','distance','access','taxrate', ...
         'pupil/teacher','blackpop','lowclass');
ys = studentize(log(y)); xs = studentize(x(:,2:end));
clear boston; 
clear y;
clear x;
info.dtype = 'exponential';
result = gwr(ys,xs,latit,longi,info);
prt(result,vnames);
plt(result,vnames);