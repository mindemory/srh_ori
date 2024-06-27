function plotGrating(parameters, centerX, centerY, size, sf, ori)
% Parameters:
% centerX: X coordiante of the grating center
% centerY: Y coordiante of the grating center
% size: Size of the grating in pixels
% sf: Spatial frequency of the grating
% ori: Orientation of the grating in degrees

if nargin < 5
    sf = 1;
end

if nargin < 6
    ori =  45; % in degrees
end
size_cm = tand(size) * parameters.viewingDistance;
size_pixx = size_cm / parameters.pixSize;

% Create a meshgrid
[x, y] = meshgrid(-size_pixx/2:size_pixx/2, -size_pixx/2:size_pixx/2);

% Rotate the grid 
xRot = x * cos(deg2rad(ori)) - y * sin(deg2rad(ori));

% Create a gray screen
grayScreen = 0.5 * ones(parameters.screenYpixels, parameters.screenXpixels);

% Determine the position of the grating on the screen
xStart = max(1, centerX - size_pixx/2);
xEnd = min(parameters.screenXpixels, centerX + size_pixx/2);
yStart = max(1, centerY - size_pixx/2);
yEnd = min(parameters.screenYpixels, centerY + size_pixx/2);

% Make the sinusoidal grating
grating = sin(2 * pi * sf * xRot);

grayScreen(yStart:yEnd, xStart:xEnd) = grating(1:(yEnd-yStart+1), 1:(xEnd-xStart+1));
figure();
imagesc(grayScreen, [0 1]);
% imagesc(centerX + [-size_pixx/2, size_pixx/2], centerY + [-size_pixx/2, size_pixx/2], grating);
colormap gray;


end