function [XY] = connectPoints(XY1,XY2)

% Created by Mrugank:
% The data points are connected using linear regression. Three conditions
% are considered wherein the two points have same x-coordinates, first
% point has a smaller co-ordinate than the second, and vice versa.

x1 = XY1(1); x2 = XY2(1);
y1 = XY1(2); y2 = XY2(2);

if x1 == x2
    Y = y1:y2;
    X = x1 * ones(1, length(Y));
elseif x1 < x2
    slope = (y2-y1)/(x2-x1);
    X = x1:x2;
    Y = round(slope*(X - x1) + y1);
elseif x1 > x2
    slope = (y2-y1)/(x2-x1);
    X = x1:-1:x2;
    Y = round(slope*(X - x1) + y1);
end
XY = [X; Y]';
end
