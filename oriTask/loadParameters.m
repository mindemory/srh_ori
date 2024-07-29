function parameters = loadParameters(subjID)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% program basic settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.eyeTrackerOn = 0;
parameters.transparency = 0.85; % transparency for debug mode
parameters.viewingDistance = 55; % viewDist (in cm)
parameters.waitBeforePulse = 3.00; % seconds
parameters.apertureSize = 28; % in degrees of visual angle
parameters.cpd = 7; % cycles per degree for the grating
parameters.rotStep = 1; % degrees to rotate the response by with each button press
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% study parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.studyname = 'MGS_TMS_V1ori';
parameters.subject = subjID;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stimulus parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.procolor = [255 0 0]; % red
parameters.anticolor = [255 0 0]; % red
parameters.cuecolor = [0 255 0]; % green
parameters.feebackcolor = [0 255 0]; % green
parameters.fixationSizeDeg = 0.6; % degrees of visual angle
parameters.boxColor = [0 0 0]; % black
parameters.fixbreakthresh = 1; % degrees of visual angle

% Adding parameters for stimulus
parameters.gratingSize = 12; % in dva
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TMS pulse parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.pulseFrequency = 20; % in Hz
parameters.pulseNum = 3; % Pulse count
parameters.pulseDuration = (parameters.pulseNum-1)/parameters.pulseFrequency; % in seconds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% timing parameters (in seconds)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.initDuration = 1;
parameters.sampleDuration = 0.25; %0.5;
parameters.delayDuration = 4;
% parameters.delay1Duration = parameters.delayDuration/2;
% parameters.delay2Duration = parameters.delayDuration/2;%-parameters.delay1Duration-parameters.pulseDuration;
parameters.respDuration = 2;
parameters.feedbackDuration = 1;
parameters.itiDuration = [1.5,2.5];
end




