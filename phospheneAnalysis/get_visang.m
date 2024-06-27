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

% % compute r given a buffer of visual angle
% r_outer_cm = parameters.viewingDistance * tand(va+0.5*parameters.rbuffer); % in cm
% r_inner_cm = parameters.viewingDistance * tand(va-0.5*parameters.rbuffer); % in cm
% r_outer_max_cm = parameters.viewingDistance * tand(parameters.maxRadius); % in cm
% r_inner_min_cm = parameters.viewingDistance * tand(parameters.minRadius); % in cm
% 
% % compute r_outer and r_inner in pixel
% r_outer = r_outer_cm./parameters.pixSize;
% r_inner = r_inner_cm./parameters.pixSize;
% r_outer_max = r_outer_max_cm./parameters.pixSize;
% r_inner_min = r_inner_min_cm./parameters.pixSize;
% 
% % r_outer and r_inner within permissible bounds
% r_outer = min([r_outer, r_outer_max]);
% r_inner = max([r_inner, r_inner_min]);
% 
% polar angle of XY
% theta = atan2d(dy_cm, dx_cm);

% Compute stimSize in pixels
simSize_cm = tand(parameters.stimSize) * parameters.viewingDistance; % stimulus size in cm
stimSize_px = round(simSize_cm / parameters.pixSize);

targX = X_mean;
targY = Y_mean;

nontargX = parameters.xCenter + dx;
nontargY = Y_mean;

% boundary of polar angles possible
% theta_range = theta-60:1:theta+60; % assuming phosphenes cannot span more than 120 degrees
% theta_range(theta_range<0) = 360 + theta_range(theta_range<0); % accounting for angles between 0 to 360
% 
% % compute border of outer arc
% x_outer = r_outer.*cosd(theta_range) + parameters.xCenter; % in pixel
% y_outer = (-r_outer.*sind(theta_range) + parameters.yCenter); % in pixel
% coords_outer = [x_outer; y_outer]';
% 
% % compute border of inner arc
% x_inner = r_inner.*cosd(theta_range) + parameters.xCenter; % in pixel
% y_inner = (-r_inner.*sind(theta_range) + parameters.yCenter); % in pixel
% coords_inner = [x_inner; y_inner]';
% 
% % compute edges that connect outer and inner arcs
% edge1 = connectPoints(coords_outer(1, :), coords_inner(1, :));
% %edge1 = connect_missing_points(edge1);
% edge2 = connectPoints(coords_outer(end, :), coords_inner(end, :));
% %edge2 = connect_missing_points(edge2);
% 
% % create a giant matrix of all coordinates of probably span of
% % phosphene map within which target locations will be drawn
% coords_all = [edge1; coords_outer; flip(edge2); flip(coords_inner)];
% coords_all = round(coords_all);
end