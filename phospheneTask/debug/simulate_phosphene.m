%% Mrugank
clear; close all; clc;% clear mex;
global parameters screen hostname kbx

subjID = '100';
session = '01';

[ret, hostname] = system('hostname');
if ret ~= 0
    hostname = getenv('HOSTNAME');
end
hostname = strtrim(hostname);

% Load PTB and toolboxes
if strcmp(hostname, 'syndrome')
    % Syndrome Mac
    addpath(genpath('/Users/Shared/Psychtoolbox'))
    addpath(genpath('/d/DATA/hyper/experiments/Mrugank/TMS/mgs_stimul/tmsRoom/Retinotopy'))
    addpath(genpath('/d/DATA/hyper/experiments/Mrugank/TMS/mgs_stimul/tmsRoom/markstim-master'))
elseif strcmp(hostname, 'tmsstim.cbi.fas.nyu.edu')
    % Mac Stimulus Display
    addpath(genpath('/Users/curtislab/TMS_Priority/exp_materials/'))
elseif strcmp(hostname, 'tmsubuntu')
    % Ubuntu Stimulus Display
    addpath(genpath('/usr/lib/psychtoolbox-3'))
    addpath(genpath('/home/curtislab/Desktop/mgs_stimul/tmsRoom/Retinotopy'))
    addpath(genpath('/home/curtislab/Desktop/mgs_stimul/tmsRoom/markstim-master'))
else
    disp('Running on unknown device. Psychtoolbox might not be added correctly!')
end

sca;
Screen('Preference','SkipSyncTests', 1) %% mrugank (01/29/2022): To suppress VBL Sync Error by PTB

% initialize parameters, screen parameters and detect peripherals
loadParameters(subjID, session);
initScreen()
initPeripherals()

FixCross = [screen.xCenter-1, screen.yCenter-4, screen.xCenter+1, screen.yCenter+4;...
    screen.xCenter-4, screen.yCenter-1, screen.xCenter+4, screen.yCenter+1];
Screen('FillRect', screen.win, parameters.fixation_color, FixCross');
Screen('Flip', screen.win);

power = 3;
noisee = wgn(screen.screenYpixels, screen.screenXpixels, power);
%noiseMask = imread(noisee);
%figure();
%imshow(noisee)
mask_location = zeros(screen.screenYpixels, screen.screenXpixels);

xpixs = 1:screen.screenXpixels;  % plotting range from -5 to 5
ypixs = 1:screen.screenXpixels;
[x, y] = meshgrid(xpixs, ypixs);  % Get 2-D mesh for x and y based on r
alpha = 4.2;
a = 17;
b = 3;
delx = screen.xCenter;
dely = screen.yCenter + 700;
x = x - delx; y = y-dely;
ellipse_phosph = (((x.*cos(alpha) + y.*sin(alpha)).^2)./a) + (((x.*sin(alpha) - y.*cos(alpha)).^2)./b) - 150.*x < 0;



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

for yy = 1:outputy
    if sum(output_phosph(:, yy) > 0)
        yy;
        xx_f = find(output_phosph(:, yy), 1, 'first');
        ellipse_bound = [ellipse_bound; [xx_f, yy]];
        xx_l = find(output_phosph(:, yy), 1, 'last');
        ellipse_bound = [ellipse_bound; [xx_l, yy]];
    end
end
noiseMask = Screen('MakeTexture', screen.win, noisee .* output_phosph');
%borderMask = Screen('MakeTexture', screen.win, screen.white .* output_phosph');

%noiseMask = Screen('MakeTexture', screen.win, output' .* screen.white);
Screen('DrawTexture', screen.win, noiseMask);
Screen('DrawDots', screen.win, ellipse_bound', 2, [200 0 0]);
%Screen('FramePoly', screen.win, [128 0 0], ellipse_bound);
%Screen('DrawTexture', screen.win, borderMask);
Screen('FillRect', screen.win, parameters.fixation_color, FixCross');
Screen('Flip', screen.win);
while 1
    KbQueueStart(kbx);
    [keyIsDown, keyCode]=KbQueueCheck(kbx);
    while ~keyIsDown
        [keyIsDown, keyCode]=KbQueueCheck(kbx);
        cmndKey = KbName(keyCode);
    end
    if strcmp(cmndKey, '1')
        break;
    end
%    break;
end

sca