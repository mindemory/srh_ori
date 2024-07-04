function [va, stimSize_px, targX, targY, nontargX, nontargY] = ...
                            get_visang(X_mean, Y_mean, parameters)
dx = X_mean - parameters.xCenter;
dy = -(Y_mean - parameters.yCenter);

% convert pixel distance to cm
dx_cm = dx.*parameters.pixSize;
dy_cm = dy.*parameters.pixSize;

% compute euclidean distance of XY from center in cm
r = abs(sqrt(dx_cm.^2 + dy_cm.^2));

% compute visual angle subtended by XY
va = atan2d(r, parameters.viewingDistance);
if va > parameters.maxRadius
    va = parameters.maxRadius;
    disp('Visual angle for mean of phosphene overlap is too big. Updating it...');
elseif va < parameters.minRadius
    va = parameters.minRadius;
    disp('Visual angle for mean of phosphene overlap is too small. Updating it...');
end
    
disp(['Visual angle for mean of phosphene overlap = ' num2str(va, "%02f")]);

% Compute stimSize in pixels
simSize_cm = tand(parameters.stimSize) * parameters.viewingDistance; % stimulus size in cm
stimSize_px = round(simSize_cm / parameters.pixSize);

targX = X_mean;
targY = Y_mean;

if X_mean > parameters.xCenter
    nontargX = parameters.xCenter - abs(dx);
else
    nontargX = parameters.xCenter + abs(dx);
nontargY = Y_mean;

end