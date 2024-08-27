function ori_task(subjID, day, start_block, TMSamp, prac_status, powerMate, aperture)
clearvars -except subjID day start_block TMSamp powerMate prac_status aperture;
close all; clc;
% Created by Mrugank Dake, Curtis Lab, NYU (03/26/2024)
% Eyelink flags:
%               Fixation: XDAT 1
%               Sample:   XDAT 10/11
%               Delay 1:  XDAT 2
%               Delay 2:  XDAT 3
%               Response: XDAT 4
%               Feedback: XDAT 5
%               ITI:      XDAT 6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize parameters
subjID = num2str(subjID, "%02d"); % convert subjID to string

% for ii = 1:length(taskMap)
%     taskMap(ii).TMScond = 1;
% end
% save('/home/curtislab/Desktop/md_data/TMS_ori/data/phosphene_data/sub99/taskMap_sub99_day03.mat', 'taskMap');

parameters = loadParameters(subjID);
if nargin < 4
    TMSamp = 30; % default TMS amplitude of 30% MSO
end
if nargin < 5
    prac_status = 0; % 0: actual session, 1: practice session
end
if nargin < 6
    powerMate = 1;
end
if nargin < 7
    aperture = 0; % 0: full screen mode, 1: stimulus drawn on aperture
end

flicker = 0;
% Check the system running on: currently accepted: syndrome, tmsubuntu
[ret, hostname] = system('hostname');
if ret ~= 0
    hostname = getenv('HOSTNAME');
end
hostname = strtrim(hostname);

% Create a container mapping session number to session type
% sessContainer = containers.Map('KeyType', 'int64', 'ValueType', 'char');
% sessContainer(1) = 'notms';
% sessContainer(2) = 'early';
% sessContainer(3) = 'middle';
% sessContainer(4) = 'late';

% Initialize PTB and TMS/Eyetracking parameters
if strcmp(hostname, 'syndrome') || strcmp(hostname, 'zod') || strcmp(hostname, 'zod.psych.nyu.edu') % Lab iMac is meant for debugging
    addpath(genpath('/Applications/Psychtoolbox')) %% mrugank (01/28/2022): load PTB
    phosphene_data_path = ['/d/DATD/datd/TMS_ori/data/phosphene_data/sub' subjID];
    parameters.isDemoMode = true; % set to true if you want the screen to be transparent
%     parameters.TMS = 0; % set to 0 if there is no TMS stimulation
    parameters.EEG = 0; % set to 0 if there is no EEG recording
    parameters.eyetracker = 0; % set to 0 if there is no eyetracker
    Screen('Preference','SkipSyncTests', 1)
    if prac_status == 1
        ori_data_path = ['/d/DATD/datd/TMS_ori/data/ori_practice_data/sub' subjID];
    else
        ori_data_path = ['/d/DATD/datd/TMS_ori/data/ori_data/sub' subjID];
    end
elseif strcmp(hostname, 'tmsubuntu') % Running stimulus code for testing
%     powerMate = 1;
    addpath(genpath('/usr/share/psychtoolbox-3'))
    parameters.isDemoMode = false; %set to true if you want the screen to be transparent
%     parameters.TMS = 0; % set to 0 if there is no TMS stimulation
    % Relative paths for tmsubuntu
    curr_dir = pwd; filesepinds = strfind(curr_dir,filesep);
    master_dir = curr_dir(1:(filesepinds(end-2)-1));
    phosphene_data_path = [master_dir '/md_data/TMS_ori/data/phosphene_data/sub' subjID];
    % Path to MarkStim
    trigger_path = [master_dir '/Mrugank/EEG_TMS_triggers'];
    addpath(genpath(trigger_path));
    trigger_path_EEG = [trigger_path '/EEG'];
    if prac_status == 1
        parameters.EEG = 0;
        ori_data_path = [master_dir '/md_data/TMS_ori/data/ori_practice_data/sub' subjID];
    else
        parameters.EEG = 0;
        ori_data_path = [master_dir '/md_data/TMS_ori/data/ori_data/sub' subjID];
    end
    parameters.eyetracker = 1;
    PsychDefaultSetup(1);
else
    disp('Running on unknown device. Psychtoolbox might not be added correctly!')
    return;
end

% Initialize data paths
addpath(genpath(phosphene_data_path));
addpath(genpath(ori_data_path));

% Added temporarily
% powerMate = 1;
% Load taskMap
if prac_status == 1
    load([phosphene_data_path '/taskMapPractice_sub' subjID])
%     parameters.TMS = 0;
%     parameters.TMScond = 1; % 1 corresponds to No TMS
    end_block = length(taskMapPractice);
else
%     load([phosphene_data_path '/taskMap_sub' subjID, '_sess_' sessContainer(sess)])
    load([phosphene_data_path '/taskMap_sub' subjID '_day' num2str(day, '%02d')])

    end_block = length(taskMap);
end

% parameters.TMS = 0;
% Initialize screen and peripherals
screen = initScreen(parameters);
parameters.powerMate = powerMate;
[kbx, parameters] = initPeripherals(parameters);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start Experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for block = start_block:end_block

    if prac_status == 1
        parameters.TMS = 0;
        parameters.TMScond = 1; % 1 corresponds to NoTMS
    else
        parameters.TMScond = taskMap(block).TMScond;
        if parameters.TMScond > 1 % determine if this is a TMS task (for early, middle or late TMS)
            parameters.TMS = 1;
        else
            parameters.TMS = 0;
        end
    end
    
%     parameters.TMS = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Open TMS Port
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % detect the MagVenture and perform handshake.
    if parameters.TMS == 1
        s = TMS('Open');
        TMS('Enable', s);
        TMS('Timing', s);
        TMS('Amplitude', s, TMSamp);
    end
    
    % Delay duration based on TMScond
    if parameters.TMScond == 1
        d1_dur = 4;
    elseif parameters.TMScond == 2 % early TMS
        d1_dur = 0;
    elseif parameters.TMScond == 3 % middle TMS
        d1_dur = 2;
    elseif parameters.TMScond == 4 % late TMS
        d1_dur = 3;
    end
    d2_dur = parameters.delayDuration - d1_dur;


    timeReport = struct; % Hold time flags
    respReport = struct; % Hold subject responses
    if parameters.EEG
        EEGtimeReport = struct;
        trig_counter = 1;
        % EEG marker --> block begins
        fname = ['sudo python3 ' trigger_path_EEG '/eegflag0.py'];
        system(fname);
        EEGtimeReport.blockstart = GetSecs;
        trigReport(trig_counter) = 0;
        trig_counter = trig_counter + 1;
    end
    parameters.block = num2str(block, "%02d");

    % Create folders for the block and read taskMap for current block
    datapath = [ori_data_path '/day' num2str(day, '%02d')];
    if prac_status == 1
        parameters = initFiles(parameters, screen, datapath, kbx, block);
        tMap = taskMapPractice(1, block);
    else
        parameters = initFiles(parameters, screen, datapath, kbx, block);
        tMap = taskMap(1, block);
    end

    trialNum = length(tMap.stimVF);
    pointsBlock = [];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialize eyetracker
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    KbQueueFlush(kbx)
    if block == start_block
        while 1
            KbQueueStart(kbx);
            [keyIsDown, ~] = KbQueueCheck(kbx);
            while ~keyIsDown
                showprompts(screen, 'WelcomeWindow', parameters.TMS)
                [keyIsDown, ~] = KbQueueCheck(kbx);
            end
            break;
        end
    end

    % Initialize Eye Tracker and perform calibration
    if parameters.eyetracker ~= 0
        if ~parameters.eyeTrackerOn
            ListenChar(0);
            el = initEyeTracker(parameters, screen);
            FlushEvents;
            ListenChar(-1);
        else
            el.eye_used = 1;
            Eyelink('Openfile', parameters.edfFile);
        end
    end

    % Init start of experiment procedures
    if parameters.eyetracker
        Eyelink('StartRecording');
        WaitSecs(0.1);
        % synchronize time in edf file
        Eyelink('Message', 'SYNCTIME');
    end
    ListenChar(-1);

    % Show Block Start Screen
    if aperture == 1
        drawTextures(parameters, screen, 'Aperture');
    end
    showprompts(screen, 'BlockStart', block)
    WaitSecs(2);

    % Draw Fixation Cross
    if aperture == 1
        drawTextures(parameters, screen, 'Aperture');
    end
    drawTextures(parameters, screen, 'FixationCross');

    trialArray = 1:trialNum;
    ITI = Shuffle(repmat(parameters.itiDuration, [1 trialNum/2]));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Task Starts
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for trial = trialArray
        disp(['runing block: ' num2str(block, "%02d") ', trial: ' num2str(trial, "%02d")])

        % Send trial start to eyetracker
        if parameters.eyetracker
            Eyelink('command', 'record_status_message "TRIAL %i/%i "', ...
                trial, trialNum);
            Eyelink('Message', 'TRIAL %i ', trial);
        end

        if trial == 1 % for first trial, pause for 2 seconds
            WaitSecs(2);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % initial fixation window
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        initStartTime = GetSecs;

        % EEG marker --> Fixation
        if parameters.EEG
            fname = ['sudo python3 ' trigger_path_EEG '/eegflag1.py'];
            system(fname);
            EEGtimeReport.init(trial) = GetSecs;
            trigReport(trig_counter) = 1;
            trig_counter = trig_counter + 1;
        end

        %record to the edf file that sample is started
        if parameters.eyetracker %&& Eyelink('NewFloatSampleAvailable') > 0
            Eyelink('command', 'record_status_message "TRIAL %i/%i /fixation"', trial, trialNum);
            Eyelink('Message', 'XDAT %i ', 1);
            Eyelink('Message', 'TarX %s ', num2str(screen.xCenter));
            Eyelink('Message', 'TarY %s ', num2str(screen.yCenter));
        end

        % draw fixation cross
        if aperture == 1
            drawTextures(parameters, screen, 'Aperture');
        end
        drawTextures(parameters, screen, 'FixationCross');

        if GetSecs - initStartTime < parameters.initDuration
            WaitSecs(parameters.initDuration - (GetSecs-initStartTime));
        end
        timeReport(trial).initDuration = GetSecs-initStartTime;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % sample window
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        sampleStartTime = GetSecs;

        % EEG marker --> sample begins
        if parameters.EEG
            if tMap.stimVF(trial) == 0
                % stimulus is outPF
                trigger_code = 10;
            elseif tMap.stimVF(trial) == 1
                % stimulus is inPF
                trigger_code = 11;
            end
            fname = ['sudo python3 ' trigger_path_EEG '/eegflag' num2str(trigger_code) '.py'];
            system(fname);
            EEGtimeReport.sample(trial) = GetSecs;
            trigReport(trig_counter) = trigger_code;
            trig_counter = trig_counter + 1;
        end

        %record to the edf file that sample is started
        if parameters.eyetracker %&& Eyelink('NewFloatSampleAvailable') > 0
            Eyelink('command', 'record_status_message "TRIAL %i/%i /sample"', trial, trialNum);
            if tMap.stimVF(trial) == 0
                Eyelink('Message', 'XDAT %i ', 10);
            elseif tMap.stimVF(trial) == 1
                Eyelink('Message', 'XDAT %i ', 11);
            end
            Eyelink('Message', 'TarX %s ', num2str(screen.xCenter));
            Eyelink('Message', 'TarY %s ', num2str(screen.yCenter));
        end

        if aperture == 1
            drawTextures(parameters, screen, 'Aperture');
        end
        phase_val = 0;
        if flicker
            while GetSecs - sampleStartTime < parameters.sampleDuration * 0.99
                drawTextures(parameters, screen, 'Grating', screen.white, tMap, trial, phase_val);
                drawTextures(parameters, screen, 'FixationCross');
                phase_val = phase_val + 90;
                WaitSecs(0.1);
                drawTextures(parameters, screen, 'FixationCross');
            end
        else
            drawTextures(parameters, screen, 'Grating', screen.white, tMap, trial);
            drawTextures(parameters, screen, 'FixationCross');
        end
        

        if GetSecs - sampleStartTime < parameters.sampleDuration
            WaitSecs(parameters.sampleDuration - (GetSecs-sampleStartTime));
        end
        timeReport(trial).sampleDuration = GetSecs-sampleStartTime;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Delay1 window
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        delay1StartTime = GetSecs;

        if d1_dur > 0
            % EEG marker --> Delay 1 begins
            if parameters.EEG
                fname = ['sudo python3 ' trigger_path_EEG '/eegflag2.py'];
                system(fname);
                EEGtimeReport.delay1(trial) = GetSecs;
                trigReport(trig_counter) = 2;
                trig_counter = trig_counter + 1;
            end
    
            %record to the edf file that delay1 is started
            if parameters.eyetracker %&& Eyelink('NewFloatSampleAvailable') > 0
                Eyelink('command', 'record_status_message "TRIAL %i/%i /delay1"', trial, trialNum);
                Eyelink('Message', 'XDAT %i ', 2);
            end

            % Draw fixation cross
            if aperture == 1
                drawTextures(parameters, screen, 'Aperture');
            end
            drawTextures(parameters, screen, 'FixationCross');

            if GetSecs - delay1StartTime < d1_dur
                WaitSecs(d1_dur - (GetSecs-delay1StartTime));
            end
        end
        timeReport(trial).delay1Duration = GetSecs - delay1StartTime;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Delay2 window
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        delay2StartTime = GetSecs;
        
        if d2_dur > 0
            % EEG marker --> Delay 2 begins
            if parameters.EEG
                fname = ['sudo python3 ' trigger_path_EEG '/eegflag3.py'];
                system(fname);
                EEGtimeReport.delay2(trial) = GetSecs;
                trigReport(trig_counter) = 3;
                trig_counter = trig_counter + 1;
            end
    
            if parameters.TMS == 1
                TMS('Train', s); % Train of TMS pulses, set pulse protocol on MagVenture Timing page
            end
    
            %record to the edf file that noise mask is started
            if parameters.eyetracker
                Eyelink('command', 'record_status_message "TRIAL %i/%i /delay2"', trial, trialNum);
                Eyelink('Message', 'XDAT %i ', 3);
            end
    
            
            % Draw fixation cross
            if aperture == 1
                drawTextures(parameters, screen, 'Aperture');
            end
            drawTextures(parameters, screen, 'FixationCross');

            if GetSecs - delay2StartTime < d2_dur
                WaitSecs(d2_dur - (GetSecs - delay2StartTime));
            end

        end
        timeReport(trial).delay2Duration = GetSecs - delay2StartTime;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Response window
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        respStartTime = GetSecs;

        % EEG marker --> Response begins
        if parameters.EEG
            fname = ['sudo python3 ' trigger_path_EEG '/eegflag4.py'];
            system(fname);
            EEGtimeReport.response(trial) = GetSecs;
            trigReport(trig_counter) = 4;
            trig_counter = trig_counter + 1;
        end

        %record to the edf file that response cue is started
        if parameters.eyetracker
            Eyelink('command', 'record_status_message "TRIAL %i/%i /response"', trial, trialNum);
            Eyelink('Message', 'XDAT %i ', 4);
            Eyelink('Message', 'TarX %s ', num2str(screen.xCenter));
            Eyelink('Message', 'TarY %s ', num2str(screen.yCenter));
        end
        % Draw green fixation cross
        if aperture == 1
            drawTextures(parameters, screen, 'Aperture');
        end
        
        % Initialize the response cue to some orientation
        initOri = rand*180;
        currOri = initOri;
        respReport(trial).initOri = initOri;
        ori_holder = [];

        % Initialize the powerMate or the keyboard depending on which one
        % is being used for response
        if parameters.powerMate == 1
            [button, dialPosOrig] = PsychPowerMateMrugank('Get', parameters.powMate);
        else
            KbQueueCreate();
            KbQueueStart();
        end
        
        while GetSecs-respStartTime < parameters.respDuration 
            
            if parameters.powerMate == 1
                % if the device recording the response is Powermate
                [button, dialPos] = PsychPowerMateMrugank('Get', parameters.powMate);
                deltaDial = dialPos - dialPosOrig;
                currOri = rem(initOri - parameters.rotStep * deltaDial, 180);

                % Check when the subject made the response (the button
                % press on PowerMate)
                if button > 0
                    respTime = GetSecs - respStartTime;
                    if currOri < 0 || currOri > 179
                        currOri = currOri+sign(currOri)*(-1)*180;
                    end
                    ori_holder = [ori_holder currOri];
                    drawTextures(parameters, screen, 'Response', parameters.procolor, tMap, trial, currOri);
                    Screen('Flip', screen.win);
                    break;
                end

                % Check when the subject made the response (the spacebar
                % press)
                [keyIsDown, ~, keyCode] = KbCheck;
                if keyIsDown
                    if keyCode(KbName(parameters.space_key))
                        respTime = GetSecs - respStartTime;
                        if currOri < 0 || currOri > 179
                            currOri = currOri+sign(currOri)*(-1)*180;
                        end
                        ori_holder = [ori_holder currOri];
                        drawTextures(parameters, screen, 'Response', parameters.procolor, tMap, trial, currOri);
                        Screen('Flip', screen.win);
                        break;
                    end
                end
            else
                % Use keyboard if powerMate is off
                [keyIsDown, ~, keyCode] = KbCheck; %KbQueueCheck(kbx);
                if keyIsDown
                    if keyCode(KbName(parameters.leftarrow_key))
                        currOri = currOri + parameters.rotStep;
                    elseif keyCode(KbName(parameters.rightarrow_key))
                        currOri = currOri - parameters.rotStep;
                    elseif keyCode(KbName(parameters.space_key))
                        respTime = GetSecs - respStartTime;
                    end
                end
            end
            
            if currOri < 0 || currOri > 179
                currOri = currOri+sign(currOri)*(-1)*180;
            end
            ori_holder = [ori_holder currOri];
            drawTextures(parameters, screen, 'Response', parameters.procolor, tMap, trial, currOri);
            Screen('Flip', screen.win);
        end
        
        respReport(trial).oriArray = ori_holder; % Save the response vector
        respReport(trial).finalOri = currOri; % The final response subject selected
        
        if ~exist('respTime', 'var')
            respReport(trial).RT = parameters.respDuration;
        else
            respReport(trial).RT = respTime;
        end

        clearvars respTime;
        timeReport(trial).respDuration = GetSecs - respStartTime;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Feedback window
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        feedbackStartTime = GetSecs;

        % EEG marker --> Feedback begins
        if parameters.EEG
            fname = ['sudo python3 ' trigger_path_EEG '/eegflag5.py'];
            system(fname);
            EEGtimeReport.feedback(trial) = GetSecs;
            trigReport(trig_counter) = 5;
            trig_counter = trig_counter + 1;
        end
        
        %record to the edf file that feedback is started
        if parameters.eyetracker
            Eyelink('command', 'record_status_message "TRIAL %i/%i /feedback"', trial, trialNum);
            Eyelink('Message', 'XDAT %i ', 5);
        end

        % Get the error made by the subject
        errorOri = tMap.targOri_(trial) - mod(currOri, 180);
        respReport(trial).errorOri = errorOri;
        
        % Render stimulus
        if aperture == 1
            drawTextures(parameters, screen, 'Aperture');
        end
        drawTextures(parameters, screen, 'GratingFeedback', screen.white, tMap, trial);
        if respTime >= 3
            drawTextures(parameters, screen, 'Response', parameters.TimeOutcolor, tMap, trial, currOri);
        elseif abs(errorOri) < 10
            drawTextures(parameters, screen, 'Response', parameters.feebackcolor, tMap, trial, currOri);
        else
            drawTextures(parameters, screen, 'Response', parameters.procolor, tMap, trial, currOri);
        end
        pointsThisTrial = errorToPoints(abs(errorOri));
%         showprompts(screen, 'FeedbackErr', errorOri, 1)
        showprompts(screen, 'FeedbackErr', pointsThisTrial, 1)
        pointsBlock = [pointsBlock pointsThisTrial];

        if GetSecs - feedbackStartTime < parameters.feedbackDuration
            WaitSecs(parameters.feedbackDuration - (GetSecs - feedbackStartTime));
        end
        timeReport(trial).feedbackDuration = GetSecs-feedbackStartTime;

        if parameters.eyetracker
            Eyelink('Message', 'TRIAL_RESULT  0')
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Intertrial window
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        itiStartTime = GetSecs;

        % EEG marker --> ITI begins
        if parameters.EEG
            fname = ['sudo python3 ' trigger_path_EEG '/eegflag6.py'];
            system(fname);
            EEGtimeReport.iti(trial) = GetSecs;
            trigReport(trig_counter) = 6;
            trig_counter = trig_counter + 1;
        end

        %record to the edf file that iti is started
        if parameters.eyetracker
            Eyelink('command', 'record_status_message "TRIAL %i/%i /iti"', trial, trialNum);
            Eyelink('Message', 'XDAT %i ', 6);
            Eyelink('Message', 'TarX %s ', num2str(screen.xCenter));
            Eyelink('Message', 'TarY %s ', num2str(screen.yCenter));
        end
        % Draw a fixation cross
        if aperture == 1
            drawTextures(parameters, screen, 'Aperture');
        end
        drawTextures(parameters, screen, 'FixationCrossITI');

        if GetSecs - itiStartTime < ITI(trial)
            WaitSecs(ITI(trial) - (GetSecs - itiStartTime));
        end

        timeReport(trial).itiDuration = GetSecs - itiStartTime;
        timeReport(trial).trialDuration = GetSecs - initStartTime;
    end

    %% Saving Data and Closing everything
    % stop eyelink and save eyelink data
%     showprompts(screen, 'BlockEnd', block)
    showprompts(screen, 'BlockEnd', mean(pointsBlock));
    Beeper('med',0.5,0.1)
    
    % Close Eyelink
    if parameters.eyetracker
        Eyelink('StopRecording');
        Eyelink('ReceiveFile', parameters.edfFile);
        copyfile([parameters.edfFile '.edf'], [parameters.block_dir filesep parameters.edfFile '.edf']);
        Eyelink('Shutdown');
        disp(['Eyedata recieve for ' num2str(block,"%02d") ' OK!']);
    end

    % EEG marker --> Block ends
    if parameters.EEG
        fname = ['sudo python3 ' trigger_path_EEG '/eegflag7.py'];
        system(fname);
        EEGtimeReport.blockend = GetSecs;
        trigReport(trig_counter) = 7;
        trig_counter = trig_counter + 1;
    end

    % save timeReport
    matFile.parameters = parameters;
    matFile.screen = screen;
    matFile.timeReport = timeReport;
    matFile.respReport = respReport;
    save([parameters.block_dir filesep parameters.matFile],'matFile')

    % Save EEG flags
    if parameters.EEG == 1
        trigReport = trigReport(~isnan(trigReport));
        EEGsummary.timeReport = EEGtimeReport;
        EEGsummary.trigReport = trigReport;
        save([parameters.block_dir filesep parameters.EEGreportFile], 'EEGsummary')
    end

    % check for end of block
    KbQueueFlush(kbx);
    [keyIsDown, ~] = KbQueueCheck(kbx);
    while ~keyIsDown
        showprompts(screen, 'ContinueorEsc', block)
        [keyIsDown, keyCode] = KbQueueCheck(kbx);
        cmndKey = KbName(keyCode);
    end
    
    % Turn off TMS
    if parameters.TMS == 1
        TMS('Disable', s);
        TMS('Close', s);
    end

    if strcmp(cmndKey, parameters.space_key)
        continue;
    elseif strcmp(cmndKey, parameters.exit_key)
        
        % Close the powerMate if being used
        if parameters.powMate == 1
            PsychPowerMateMrugank('Close', parameters.powMate)
        end
        showprompts(screen, 'EndExperiment');
        WaitSecs(2);
        ListenChar(1);
        sca;
        return;
    end
end % end of block

% % Close TMS Port and End Experiment
% if parameters.TMS == 1
%     TMS('Disable', s);
%     TMS('Close', s);
% end
% Close the powerMate if being used
if parameters.powMate == 1
    PsychPowerMateMrugank('Close', parameters.powMate)
end
showprompts(screen, 'EndExperiment');
ListenChar(1);
WaitSecs(2);
sca;
Priority(0);
end