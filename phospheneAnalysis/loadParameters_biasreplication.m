function parameters = loadParameters_biasreplication()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for the ori_task bias-replication protocol.
%
% Unlike loadParameters.m (the original phospheneAnalysis parameter set),
% this does NOT take a tmsRtnTpy struct -- the bias-replication pipeline
% never re-runs phosphene localization, so no raw screen/task struct from
% a localization session is available or needed here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Target orientation categories
parameters.valOris = [45, 135];

% Jitter is now discrete: +/-1 to +/-24 degrees, in integer steps (48
% values total, zero excluded), rather than the original continuous
% uniform +/-10 degree jitter.
parameters.jitterVals = [-24:-1, 1:24];
parameters.numJitters = length(parameters.jitterVals); % 48

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block / trial structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.numTrialsPerBlock = 24;
parameters.numBlocksPerCond = 8; % No-TMS and Middle-TMS each get 8 blocks
parameters.numBlocksPractice = 10;

% TMS condition codes match the ORIGINAL taskMap's TMScond meaning
% (1 = No-TMS, 2 = Early TMS, 3 = Middle TMS, 4 = Late TMS), so
% ori_task_biasreplication.m can reuse ori_task.m's TMScond -> delay
% lookup unmodified. Early/Late are simply never produced here.
parameters.tmsConds = [1, 3];
end
