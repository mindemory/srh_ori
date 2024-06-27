function recordPhosphenes(subjID, session, TMSamp)
%% Record Phoshpenes (mrugank: 03/07/2022)
% The code has been originally written by Masih. It has been adapted to
% make changes to suitably adjust to the new display, interact with the TMS
% device using MarkStim trigger and few other changes.

% The code creates a dark stimulus display with a red fixation cross. When
% the TMS parameter is set to 1 in loadParameters(), the code interacts with
% the Teensy trigger to first initiate a handshake. Make sure that the
% orange light is turned on on the TeensyTrigger before running the code.
% If the light is not on, reset the TeensyTrigger by pressing the orange
% button.

% It then asks for a new coil location, followed by whether the coil
% location is new or the trial is new. For a new trial, it will make the
% screen black and interact with the TMS device to send a pulse that is set
% on the TMS machine. It then waits for the subject to report if phosphene
% was seen. A left mouse-click implies a phosphene was seen while a right
% mouse-click implies a phosphene was not seen. If a phosphene was seen, it
% then activates a draw tool for the subject to draw the rough borders to
% report the phosphene seen. The drawing is ended upon second left click.

% The code runs till quit (q) is pressed by the experimenter. 

% Created by Mrugank Dake, Curtis Lab, NYU (10/11/2022)
clearvars -except subjID session TMSamp; 
close all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize parameters
subjID = num2str(subjID, '%02d');
session = num2str(session, '%02d');
if nargin < 3
    TMSamp = 30; % default TMS amplitude of 30% MSO
end
parameters = loadParameters(subjID, session);

% Check the system name to ensure correct paths are added.
[ret, hostname] = system('hostname');
if ret ~= 0
    hostname = getenv('HOSTNAME');
end
hostname = strtrim(hostname);

% Load PTB and toolboxes
if strcmp(hostname, 'syndrome') || strcmp(hostname, 'zod') || strcmp(hostname, 'catwoman') ...
        || strcmp(hostname, 'catwoman.psych.nyu.edu')
    thisdev = 'mac';
    % Syndrome Mac
    addpath(genpath('/Applications/Psychtoolbox'))
    parameters.isDemoMode = true; %set to true if you want the screen to be transparent
    parameters.TMS = 0; % set to 0 if there is no TMS stimulation
    data_path = '/d/DATD/datd/TMS_ori/data/phosphene_data';
    Screen('Preference','SkipSyncTests', 0)
    %Screen('Preference', 'SyncTestSettings', .0004);
elseif strcmp(hostname, 'tmsubuntu')
    thisdev = 'linux';
    % Ubuntu Stimulus Display
    addpath(genpath('/usr/share/psychtoolbox-3'))
    parameters.isDemoMode = false; %set to true if you want the screen to be transparent
    parameters.TMS = 1; % set to 0 if there is no TMS stimulation
    curr_dir = pwd;
    filesepinds = strfind(curr_dir,filesep);
    master_dir = curr_dir(1:(filesepinds(end-2)-1));
    trigger_path = [master_dir '/Mrugank/EEG_TMS_triggers'];
    addpath(genpath(trigger_path));
    data_path = [master_dir '/md_data/TMS_ori/data/phosphene_data'];
    PsychDefaultSetup(1);
else
    disp('Running on unknown device. Psychtoolbox might not be added correctly!')
    return;
end

% Initialize data paths
addpath(genpath(data_path));

% Initialize screen and peripherals
screen = initScreen(parameters);
[kbx, mbx, parameters] = initPeripherals(parameters, thisdev);

% Store screen and subject parameters in tmsRtnTpy
tmsRtnTpy.Params.ID = ['sub' subjID '_sess' session];
tmsRtnTpy.Params.screen = screen;
tmsRtnTpy.Params.taskParams = parameters;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Open TMS Port
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% detect the MagVenture and perform handshake.
if parameters.TMS > 0
    s = TMS('Open');
    TMS('Enable', s);
    TMS('Timing', s);
    TMS('Amplitude', s, TMSamp);
end

% Initialize trial and coil indices, incrementally increased on each run
trialInd = 0;
coilLocInd = 0;
ListenChar(-1) % prevent inputs from keyboard and mouse on main screen

parameters = initFiles(parameters, screen, data_path, kbx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start Experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Instructions for the subject
while 1
    KbQueueStart(kbx);
    [keyIsDown, ~]= KbQueueCheck(kbx);
    while ~keyIsDown
        showprompts(screen, 'WelcomeWindow')
        [keyIsDown, ~]= KbQueueCheck(kbx);
    end
    break;
end

% Run the task
while 1
    % check for any keypresses
    KbQueueStart(kbx);
    [keyIsDown, ~] = KbQueueCheck(kbx);
    % Check for new or old coil location
    while ~keyIsDown
        showprompts(screen, 'FirstMessage')
        [keyIsDown, keyCode] = KbQueueCheck(kbx);
        cmndKey = KbName(keyCode);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % New coil location
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(cmndKey, parameters.newloc_key)
        display(sprintf('\n---------------------------------------------'));
        display(sprintf('\n\tnew coil location initiated.'));
        coilLocInd = coilLocInd+1;
        while 1
            KbQueueStart(kbx);
            [keyIsDown, ~] = KbQueueCheck(kbx);
            while ~keyIsDown
                showprompts(screen, 'SecondMessage')
                [keyIsDown, keyCode] = KbQueueCheck(kbx);
                cmndKey = KbName(keyCode);
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % New trial
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if strcmp(cmndKey, parameters.trial_key)                
                drawTextures(parameters, screen, 'FixationCross');
                trialInd = trialInd+1;
                display(sprintf('\n\tready for puls'));
                % send a signal to the a USB to trigger the TMS pulse
                pause(parameters.waitBeforePulse);
                if parameters.TMS
                    TMS('Train', s); % Train of TMS pulses, set pulse protocol on MagVenture Timing page
                end
                WaitSecs(parameters.PulseDuration);
                display(sprintf('\n\ttrigger pulse sent to the TMS machine'));
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % wait for subject's response:
                % right click: not seen a phosphene, go to next trial
                % first left click: start drawing
                % second left click: end drawing
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                while 1
                    % show the mouse location and wait for subject's click
                    KbQueueStart(mbx);
                    [~, clickCode] = KbQueueCheck(mbx);
                    SetMouse(screen.xCenter, screen.yCenter, screen.win);
                    HideCursor(screen.win);
                    while ~any(clickCode)
                        [~, clickCode] = KbQueueCheck(mbx);
                        [x,y,~]=GetMouse(screen.win);
                        drawTextures(parameters, screen, 'MousePointer', x, y);
                    end
                    KbQueueStop(mbx);
                    Response.CoilLocation(trialInd) = coilLocInd;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Abort Trial
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if clickCode(parameters.right_key)
                        Response.Detection(trialInd) = 0;
                        Response.Drawing{trialInd} = nan;
                        showprompts(screen, 'NoPhosphene');
                        WaitSecs(2);
                        break
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Start Drawing
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    elseif clickCode(parameters.left_key)
                        Response.Detection(trialInd) = 1;
                        KbQueueStart(mbx);
                        [~, clickCode] = KbQueueCheck(mbx);
                        [x, y, ~] = GetMouse(screen.win);
                        XY = [x y];
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Finish Drawing
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        while ~clickCode(parameters.left_key)
                            [~, clickCode] = KbQueueCheck(mbx);
                            [x, y, ~] = GetMouse(screen.win);
                            XY = [XY; x y];
                            % Generate phosphene drawing
                            if XY(end,1) ~= XY(end-1,1) || XY(end,2) ~= XY(end-1,2)
                                drawTextures(parameters, screen, 'PhospheneDrawing', x, y, XY);
                            end
                            Screen('Flip', screen.win,[],1);
                        end
                        Response.Drawing{trialInd} = round(XY);
                        KbQueueStop(mbx);
                        break
                    end % end of recording the drawing process
                end
                Screen('Flip', screen.win); % clear Flip buffer
                drawTextures(parameters, screen, 'FixationCross');
                tmsRtnTpy.Response = Response;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % New coil location
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif strcmp(cmndKey, parameters.newloc_key) % get ready fo the next coil location if "n" is pressed
                showprompts(screen, 'NewLocation');
                break
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % End run
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif strcmp(cmndKey, parameters.quit_key) % quit the task if "q" is pressed
                break
            end
            drawTextures(parameters, screen, 'FixationCross');
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % End run
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if strcmp(cmndKey, parameters.quit_key) % quit the task if "q" is pressed
            showprompts(screen, 'Quit');
            WaitSecs(2);
            break
        end   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % End run
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif strcmp(cmndKey, parameters.quit_key)
        showprompts(screen, 'Quit');
        WaitSecs(2);
        break
    end
end
KbQueueRelease;
ListenChar(0);
sca
ShowCursor;

%% Saving data
save(parameters.fName,'tmsRtnTpy')

% Close TMS Port and End Experiment
if parameters.TMS
    TMS('Disable', s);
    TMS('Close', s);
end
sca;
end
