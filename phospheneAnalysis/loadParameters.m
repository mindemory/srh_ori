function parameters = loadParameters(tmsRtnTpy)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% program basic settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.boundSize = 8; % degrees of visual angle
parameters.stimSize = 4; % degrees of visual angle
parameters.maxRadius = 16; % degrees of visual angle
parameters.minRadius = 6; % degrees of visual angle
parameters.valOris = [30, 60, 120, 150];
parameters.jitterOris = 10; % degress of jitter allowed (uniform jitter)

parameters.xCenter = tmsRtnTpy.Params.screen.xCenter;
parameters.yCenter = tmsRtnTpy.Params.screen.yCenter;
parameters.screenXpixels = tmsRtnTpy.Params.screen.screenXpixels;
parameters.screenYpixels = tmsRtnTpy.Params.screen.screenYpixels;
parameters.pixSize = tmsRtnTpy.Params.screen.pixSize; %cm/pixel
parameters.viewingDistance = tmsRtnTpy.Params.taskParams.viewingDistance;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of trials for each condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.numRepeatsPerBlock = 3;
parameters.numTrials = length(parameters.valOris) * 2 * parameters.numRepeatsPerBlock; % make sure it is a multiple of 2
parameters.numBlocks = 20;
parameters.numBlocksPractice = 10;
parameters.sessions = 4; % 1: notms, 2: early, 3: middle, 4: late
end
