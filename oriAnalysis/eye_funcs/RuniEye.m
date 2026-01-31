function [ii_sess] = RuniEye(p, taskMap, end_block, prac_status)

if nargin < 3
    end_block = 20;
end

if nargin < 4
    prac_status = 0;
end

% Eyelink flags:
%               Fixation: XDAT 1 = 1s
%               Sample:   XDAT 10/11 = 0.25
%               Delay 1:  XDAT 2 = 0/2/4
%               Delay 2:  XDAT 3 = 4/2/0
%               Response: XDAT 4 = 3
%               Feedback: XDAT 5 = 1
%               ITI:      XDAT 6 = 1.5/2.5

ifgFile = 'p_1000hz.ifg';
ii_params = ii_loadparams; % load default set of analysis parameters, only change what we have to
ii_params.valid_epochs =[1 10 11 2 3 4 5 6]; % Updated epochs (Mrugank: 04/05/2023)
ii_params.trial_start_value = 1; %XDAT value for trial start
ii_params.trial_end_value = 6;   % XDAT value for trial end
ii_params.drift_epoch = [1 10 11 2 3]; % XDAT values for drift correction (
ii_params.drift_fixation_mode  = 'mode';
ii_params.blink_thresh = 1;
ii_params.blink_window = [100 100]; % how long before/after blink (ms) to drop?
ii_params.plot_epoch = [10 11 2 3 4 5];  % what epochs do we plot for preprocessing?
ii_params.calibrate_limits = [5]; % when amount of adj exceeds this, don't actually calibrate (trial-wise); ignore trial for polynomial fitting (run)

% Mrugank (04/05/2023): Could possibly be updated later?
excl_criteria.i_dur_thresh = 850; % must be shorter than 150 ms
excl_criteria.i_amp_thresh = 2;   % must be longer than 5 dva [if FIRST saccade in denoted epoch is not at least this long and at most this duration, drop the trial]
excl_criteria.i_err_thresh = 10;   % i_sacc must be within this many DVA of target pos to consider the trial

excl_criteria.drift_thresh = 2.5;     % if drift correction norm is this big or more, drop
excl_criteria.delay_fix_thresh = 2.5; % if any fixation is this far from 0,0 during delay (epoch 3)

for block = 1:end_block
    disp(['Running block ' num2str(block, '%02d')])
    p.block = [p.dayfolder '/block' num2str(block,'%02d')];
    
    % Loading task, display and timeReport for the block
    matFile_extract = dir(fullfile(p.block, '*.mat'));
    matFile = [p.block filesep matFile_extract(end).name];
    load(matFile);
    parameters = matFile.parameters;
    screen = matFile.screen;
%     timeReport = matFile.timeReport;
%     trialInfo{block} = timeReport;
%     eyecond = taskMap(block).condition;
    
    ii_params.resolution = [screen.screenXpixels screen.screenYpixels];
    ii_params.ppd = parameters.viewingDistance * tand(1) / screen.pixSize;
    edfFileName = parameters.edfFile;
    edfFile_original = [p.block filesep edfFileName '.edf'];
    edf_block_fold = [p.save_eyedata '/block' num2str(block, '%02d')];
    if ~exist(edf_block_fold, 'dir')
        mkdir(edf_block_fold);
    end

    edfFile = [edf_block_fold filesep edfFileName '.edf'];
    if ~exist(edfFile, 'file')
        copyfile(edfFile_original, edfFile);
    end
    % what is the output filename?
    preproc_fn = edfFile(1:end-4);
    
    % run preprocessing!
    [ii_data, ii_cfg, ii_sacc] = run_iipreproc(edfFile, ifgFile, preproc_fn, ii_params, [], {'calibration'});

    % score trials
    % default parameters should work fine - but see docs for other
    % arguments you can/should give when possible
    [ii_trial{block},~] = ii_scoreMGS(ii_data,ii_cfg,ii_sacc,[],5,[],excl_criteria,[],'strict');

    clearvars ii_cfg ii_data;
end


%% Combining runs
disp('Combining runs')

% Creating ii_sess only if ii_trial is valid
if ~exist("ii_trial", "var")
    ii_sess = [];
else
    ii_sess = ii_combineruns(ii_trial);
end
end
