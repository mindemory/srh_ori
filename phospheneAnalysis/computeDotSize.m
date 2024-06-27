function dotSize = computeDotSize(parameters, XY)
% compute euclidean distance of point from center
dx = XY(:, 1) - parameters.xCenter; %in pix
dy = -(XY(:, 2) - parameters.yCenter); %in pix
dx_cm = dx.*parameters.pixSize; %in cm
dy_cm = dy.*parameters.pixSize; %in cm
r = sqrt(dx_cm.^2 + dy_cm.^2); % in cm

% compute visual angle
va = atan2d(r, parameters.viewingDistance);

% compute visual angle bounds based on dotSize
va_lower = va - parameters.dotSize/2;
va_upper = va + parameters.dotSize/2;

% compute radial bounds for dot
r_lower_cm = parameters.viewingDistance .* tand(va_lower); %in cm
r_upper_cm = parameters.viewingDistance .* tand(va_upper); %in cm

% compute dotSize
r_lower = round((abs(r_lower_cm-r))./parameters.pixSize); %in pix
r_upper = round((abs(r_upper_cm-r))./parameters.pixSize); %in pix
dotSize = max([r_lower, r_upper], [], 2);
end