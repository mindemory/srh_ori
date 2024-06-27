% Using the dimensions of the screen, this script simulates phosphene
% drawings. Play with the values of a, b and alpha to generate different
% shaped phosphenes. The value of power can be modulated for manipulating
% coarseness of phosphenes.

%% Initialization
% Screen size
screen.screenXpixels = 1920;
screen.screenYpixels = 1080;
screen.xCenter = screen.screenXpixels/2;
screen.yCenter = screen.screenYpixels/2;
% Parameters
power = 1;
alpha = 4.4;
a = 15;
b = 1.5;

%% Phosphene location
% Creating a noise mask
noisee = wgn(screen.screenYpixels, screen.screenXpixels, power);
%noiseMask = imread(noisee);
%figure();
%imshow(noisee)

% Defining the location of the noise mask
mask_location = zeros(screen.screenYpixels, screen.screenXpixels);
xpixs = 1:screen.screenXpixels;  % plotting range from -5 to 5
ypixs = 1:screen.screenXpixels;
[x, y] = meshgrid(xpixs, ypixs);  % Get 2-D mesh for x and y based on r
delx = screen.xCenter + 100;
dely = screen.yCenter + 800;
x = x - delx; y = y-dely;
ellipse_phosph = (((x.*cos(alpha) + y.*sin(alpha)).^2)./a) + ...
    (((x.*sin(alpha) - y.*cos(alpha)).^2)./b) - 150.*x < 0;


%% Phosphene drawing
output_phosph = ones(length(xpixs), length(ypixs)); % Initialize to 1

output_phosph(~ellipse_phosph) = 0;

col_diff = screen.screenXpixels - screen.screenYpixels;
output_phosph(:,1:col_diff/2-1) = [];
output_phosph(:,end-col_diff/2:end) = [];

ellipse_bound = [];
[outputx, outputy] = size(output_phosph);
for xx = 1:outputx
    if sum(output_phosph(xx, :) > 0)
        yy_f = find(output_phosph(xx, :), 1, 'first');
        ellipse_bound = [ellipse_bound; [xx, yy_f]];
        yy_l = find(output_phosph(xx, :), 1, 'last');
        ellipse_bound = [ellipse_bound; [xx, yy_l]];
    end
end

% Phosphene boundary
for yy = 1:outputy
    if sum(output_phosph(:, yy) > 0)
        yy;
        xx_f = find(output_phosph(:, yy), 1, 'first');
        ellipse_bound = [ellipse_bound; [xx_f, yy]];
        xx_l = find(output_phosph(:, yy), 1, 'last');
        ellipse_bound = [ellipse_bound; [xx_l, yy]];
    end
end
noiseeNorm = (noisee - min(noisee, [], 'all'))./(max(noisee, [], 'all') - min(noisee, [], 'all'));
noiseMask = noiseeNorm .* output_phosph';

figure();
imagesc(noiseMask.*0.9 )
colormap('gray'); 
caxis([0, 1])
hold on;
% Draw fixation cross
line(screen.xCenter.*[0.95, 1.05], screen.yCenter.*[1, 1], 'Color', 'white', 'LineWidth', 2);
line(screen.xCenter.*[1, 1], screen.yCenter.*[0.95, 1.05], 'Color', 'white', 'LineWidth', 2);



