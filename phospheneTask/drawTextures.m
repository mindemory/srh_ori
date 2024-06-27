function drawTextures(parameters, screen, texture_name, x, y, XY)
% created by Mrugank (06/15/2022):
% drawTexture can be called with texture_name to draw either a fixation
% cross or a stimulus at periphery. color argument is optional. Default
% color for either stimulus is white.

%% Fixation Cross
fixcolor = screen.black;
% Get pixel width and height for inner and outer circle based of VA
r_pix_outer = va2pixel(parameters, screen, parameters.fixationCrossSizeDeg);
r_pix_inner = va2pixel(parameters, screen, parameters.fixationCrossSizeDeg/3);

% Coordinates for fixation cross
xCoords = [-r_pix_outer r_pix_outer 0 0];
yCoords = [0 0 -r_pix_outer r_pix_outer];
allCoords = [xCoords; yCoords];

% Coordinates for outer circle
baseRect_outer = [0 0 r_pix_outer*2 r_pix_outer*2];
maxDiameter_outer = ceil(max(baseRect_outer) * 1.1);
centeredRect_outer = CenterRectOnPoint(baseRect_outer, screen.xCenter, screen.yCenter);

% Coordinates for inner circle
baseRect_inner = [0 0 r_pix_inner*2 r_pix_inner*2];
maxDiameter_inner = ceil(max(baseRect_inner) * 1.1);
centeredRect_inner = CenterRectOnPoint(baseRect_inner, screen.xCenter, screen.yCenter);

% Draw Textures
switch texture_name
    case 'FixationCross'
        % Drawing Fixation Cross
        Screen('FillOval', screen.win, parameters.fixation_color, centeredRect_outer, maxDiameter_outer);
        if strcmp(computer, 'GLNXA64')
            Screen('DrawLines', screen.win, allCoords, round(r_pix_inner*1.5), ...
                fixcolor, [screen.xCenter screen.yCenter], 1); % 2 is for smoothing
        end
        Screen('FillOval', screen.win, parameters.fixation_color, centeredRect_inner, maxDiameter_inner);
        Screen('Flip', screen.win);
    case 'MousePointer'
        % Drawing Fixation Cross
        Screen('FillOval', screen.win, parameters.fixation_color, centeredRect_outer, maxDiameter_outer);
        if strcmp(computer, 'GLNXA64')
            Screen('DrawLines', screen.win, allCoords, round(r_pix_inner*1.5), ...
                fixcolor, [screen.xCenter screen.yCenter], 1); % 2 is for smoothing
        end
        Screen('FillOval', screen.win, parameters.fixation_color, centeredRect_inner, maxDiameter_inner);
        % Drawing Mouse Pointer
        Screen('FillOval', screen.win, parameters.pointer_color, ...
            [x-parameters.pointerSize y-parameters.pointerSize x+parameters.pointerSize y+parameters.pointerSize]);
        Screen('Flip', screen.win);
    case 'PhospheneDrawing'
        % Drawing Fixation Cross
        Screen('FillOval', screen.win, parameters.fixation_color, centeredRect_outer, maxDiameter_outer);
        if strcmp(computer, 'GLNXA64')
            Screen('DrawLines', screen.win, allCoords, round(r_pix_inner*1.5), ...
                fixcolor, [screen.xCenter screen.yCenter], 1); % 2 is for smoothing
        end
        Screen('FillOval', screen.win, parameters.fixation_color, centeredRect_inner, maxDiameter_inner);
        % Drawing Mouse Pointer
        Screen('FillOval', screen.win, parameters.pointer_color, ...
            [x-parameters.pointerSize y-parameters.pointerSize x+parameters.pointerSize y+parameters.pointerSize]);
        % Drawing Phosphene
        Screen('DrawLine', screen.win, parameters.pointer_color, ...
            XY(end-1,1), XY(end-1,2), XY(end,1), XY(end,2),1);
end
