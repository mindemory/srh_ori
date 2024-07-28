clear; close all; clc; sca;

% Check the system running on: currently accepted: syndrome, tmsubuntu
[ret, hostname] = system('hostname');
if ret ~= 0
    hostname = getenv('HOSTNAME');
end
hostname = strtrim(hostname);

% Initialize PTB and TMS/Eyetracking parameters
addpath(genpath('/Applications/Psychtoolbox')) %% mrugank (01/28/2022): load PTB
parameters.isDemoMode = true; % set to true if you want the screen to be transparent
Screen('Preference','SkipSyncTests', 1)
powerMate = 0;
parameters.TMS = 1;
parameters.transparency = 1;
parameters.viewingDistance = 50;
% Initialize screen and peripherals
screen = initScreen(parameters);
parameters.powerMate = powerMate;
[kbx, parameters] = initPeripherals(parameters);

cpd_array = [2, 3, 4, 5, 6, 7, 8]; % cycles per degree
contrast_array = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]; % contrast of grating
gausswidth_array = [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 1.0]*200/4; % width of the gaussian mask

contrastMain = 1;
cpdMain = 5;

stimSize = 200;
nStimuli = length(cpd_array);
nHyperparams = 3;
widthPerStimulus = screen.screenXpixels/nStimuli;
heightPerStimulus = screen.screenYpixels/nHyperparams;
Xarray = widthPerStimulus/2:widthPerStimulus:screen.screenXpixels;
Yarray = heightPerStimulus/2:heightPerStimulus:screen.screenYpixels;
while true
    for i1 = 1:nStimuli
        th_cpd = cpd_array(i1);
        th_targX = Xarray(i1);
        th_targY = Yarray(1);
        % drawTextures(parameters, screen, 'GratingTest', screen.white);
        drawTexturesTest(screen, th_targX, th_targY, stimSize, th_cpd, contrastMain);
    end

    for i2 = 1:nStimuli
        th_contrast = contrast_array(i2);
        th_targX = Xarray(i2);
        th_targY = Yarray(2);
        % drawTextures(parameters, screen, 'GratingTest', screen.white);
        drawTexturesTest(screen, th_targX, th_targY, stimSize, cpdMain, th_contrast);
    end


    for i3 = 1:nStimuli
        th_gwidth = gausswidth_array(i3);
        th_targX = Xarray(i3);
        th_targY = Yarray(3);
        % drawTextures(parameters, screen, 'GratingTest', screen.white);
        drawTexturesTest(screen, th_targX, th_targY, stimSize, cpdMain, contrastMain, th_gwidth);
    end
    Screen('Flip', screen.win);
end
