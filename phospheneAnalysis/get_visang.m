function [va, stimSize_px, targX, targY, nontargX, nontargY] = ...
                            get_visang(X_mean, Y_mean, va_quantiles, parameters)


desired_va = (parameters.maxRadius + parameters.minRadius) / 2;
if (desired_va >= min(va_quantiles)) && (desired_va <= max(va_quantiles))
    va = desired_va;
else
    disp('This phosphene is undesired ...');
    va = desired_va;
end

% Find distance and angle of centroid from screen center
dx_mean = X_mean - parameters.xCenter;
dy_mean = -(Y_mean - parameters.yCenter);
dtheta = atan2d(dy_mean, dx_mean); % Get the angle of the centroid of the phosphene

% Compute radius of stimulus in pixels
rho_cm = tand(va) * parameters.viewingDistance; % Radius of stimulus in cm
rho_px = round(rho_cm) / parameters.pixSize; % Radius of stimulus in pixels

% Find distance at which stimulus should be presented
dx_stimulus = rho_px * cosd(dtheta);
dy_stimulus = rho_px * sind(dtheta);

% Compute target location
targX = dx_stimulus + parameters.xCenter;
targY = -(dy_stimulus - parameters.yCenter);

% Compute non-target location (mirror along x-dimension
nontargX = -dx_stimulus + parameters.xCenter;
nontargY = -(dy_stimulus - parameters.yCenter);

% Compute stimSize in pixels
simSize_cm = tand(parameters.stimSize) * parameters.viewingDistance; % Stimulus size in cm
stimSize_px = round(simSize_cm / parameters.pixSize); % Stimulus size in pixels


end