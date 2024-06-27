% Clear the workspace and the screen
sca;
close all;
clear;

% Here we call some default settings for setting up Psychtoolbox
Screen('Preference','SkipSyncTests', 1) %% mrugank (01/29/2022): To suppress VBL Sync Error by PTB
PsychDebugWindowConfiguration(0, 0.8);
% Get the screen numbers. This gives us a number for each of the screens
% attached to our computer. For help see: Screen Screens?
screens = Screen('Screens');

% Draw we select the maximum of these numbers. So in a situation where we
% have two screens attached to our monitor we will draw to the external
% screen. When only one screen is attached to the monitor we will draw to
% this. For help see: help max
screenNumber = max(screens);

% Define black and white (white will be 1 and black 0). This is because
% luminace values are (in general) defined between 0 and 1.
% For help see: help WhiteIndex and help BlackIndex
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);

% Open an on screen window and color it black.
% For help see: Screen OpenWindow?
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, black);
[screenWidth, screenHeight] = Screen('DisplaySize',screenNumber);
screenWidth = screenWidth/10; %mm to cm
screenHeight = screenHeight/10; %mm to cm
% Get the size of the on screen window in pixels.
% For help see: Screen WindowSize?
[screenXpixels, screenYpixels] = Screen('WindowSize', window);

% Get the centre coordinate of the window in pixels
% For help see: help RectCenter
[xCenter, yCenter] = RectCenter(windowRect);
viewDist = 55;
Hperdegree = viewDist * tand(1); %height for one degree in cm
Wperdegree = Hperdegree; %width for one degree in cm
pixWidth = screenWidth/screenXpixels; %cm/pixel
pixHeight = screenHeight/screenYpixels; %cm/pixel
ppd = Wperdegree/pixWidth;
Screen('BlendFunction', window, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

VAs = 1:1:14;
for va = VAs
    theta = 0:0.1:360;
    r_old = va * ppd;
    r_new_cm = viewDist * tand(va);
    r_new = r_new_cm/pixWidth;

    points_old_x = r_old.*cosd(theta) + xCenter;
    points_old_y = -r_old.*sind(theta) + yCenter;
    points_new_x = r_new.*cosd(theta) + xCenter;
    points_new_y = -r_new.*sind(theta) + yCenter;
    points_old = round([points_old_x; points_old_y]);
    points_new = round([points_new_x; points_new_y]);
    % Enable alpha blending for anti-aliasing
    % For help see: Screen BlendFunction?
    % Also see: Chapter 6 of the OpenGL programming guide

    % Set the color of our dot to full red. Color is defined by red green
    % and blue components (RGB). So we have three numbers which
    % define our RGB values. The maximum number for each is 1 and the minimum
    % 0. So, "full red" is [1 0 0]. "Full green" [0 1 0] and "full blue" [0 0
    % 1]. Play around with these numbers and see the result.
    dotColor_old = [255 0 0];
    dotColor_new = [0 0 255];

    % Determine a random X and Y position for our dot. NOTE: As dot position is
    % randomised each time you run the script the output picture will show the
    % dot in a different position. Similarly, when you run the script the
    % position of the dot will be randomised each time. NOTE also, that if the
    % dot is drawn at the edge of the screen some of it might not be visible.

    % Dot size in pixels
    dotSizePix = 2;

    % Draw the dot to the screen. For information on the command used in
    % this line type "Screen DrawDots?" at the command line (without the
    % brackets) and press enter. Here we used good antialiasing to get nice
    % smooth edges
    Screen('DrawDots', window, points_old, dotSizePix, dotColor_old, [], 2);
    Screen('DrawDots', window, points_new, dotSizePix, dotColor_new, [], 2);
    Screen('TextSize', window, 25);
    MSG = ['Visual angle = ' num2str(va)];
    DrawFormattedText(window, MSG, 'center', ...
        'center', [0 255 0]);
    Screen('Flip', window);
    WaitSecs(1.5);

    % Flip to the screen. This command basically draws all of our previous
    % commands onto the screen. See later demos in the animation section on more
    % timing details. And how to demos in this section on how to draw multiple
    % rects at once.
    % For help see: Screen Flip?
end
% Now we have drawn to the screen we wait for a keyboard button press (any
% key) to terminate the demo. For help see: help KbStrokeWait
KbStrokeWait;

% Clear the screen. "sca" is short hand for "Screen CloseAll". This clears
% all features related to PTB. Note: we leave the variables in the
% workspace so you can have a look at them if you want.
% For help see: help sca
sca;
