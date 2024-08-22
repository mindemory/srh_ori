function [ii_sess] = RuniEye(p, taskMap, end_block, prac_status)

if nargin < 3
    end_block = 20;
end

if nargin < 4
    prac_status = 0;
end

% List of epochs in the task
% Eyelink flags:
%               Fixation: XDAT 1 = 1s
%               Sample:   XDAT 2 = 0.5s
%               Delay 1:  XDAT 3 = 2s
%               Delay 2:  XDAT 4 = 2s
%               Response: XDAT 5 = 0.85s
%               Feedback: XDAT 6 = 0.8s
%               ITI:      XDAT 7 = 1s or 2s (50-50 split)

ifgFile = 'p_1000hz.ifg';
ii_params = ii_loadparams; % load default set of analysis parameters, only change what we have to
ii_params.valid_epochs =[1 10 11 2 3 4 5 6]; % Updated epochs (Mrugank: 04/05/2023)
ii_params.trial_start_value = 1; %XDAT value for trial start
ii_params.trial_end_value = 6;   % XDAT value for trial end
ii_params.drift_epoch = [1 10 11 2 3]; % XDAT values for drift correction (
ii_params.drift_fixation_mode  = 'mode';
ii_params.calibrate_epoch = 5; % XDAT value for when we calibrate (feedback stim)
ii_params.calibrate_select_mode = 'last'; % how do we select fixation with which to calibrate?
ii_params.calibrate_mode = 'run';%'run'; % scale: trial-by-trial, rescale each trial; 'run' - run-wise polynomial fit
ii_params.blink_thresh = 0.1;
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
%     [ii_data, ii_cfg, ii_sacc] = run_iipreproc(edfFile, ifgFile, preproc_fn, ii_params, [], {'calibration'});
    [ii_data, ii_cfg, ii_sacc] = run_iipreproc(p, edfFile, ifgFile, preproc_fn, ii_params, []);

    % score trials
    % default parameters should work fine - but see docs for other
    % arguments you can/should give when possible
    [ii_trial{block},~] = ii_scoreMGS(ii_data,ii_cfg,ii_sacc,[],5,[],excl_criteria,[],'strict');
    ii_trial{block}.instimVF = taskMap(block).stimVF;
    if prac_status == 0
        ii_trial{block}.istms = ones(length(taskMap(block).stimVF), 1) * taskMap(block).TMScond;
    else
        ii_trial{block}.istms = zeros(length(taskMap(block).stimVF), 1);
    end
    clearvars ii_cfg ii_data;
end


%% Combining runs
disp('Combining runs')

% Creating ii_sess only if ii_trial is valid
if ~exist("ii_trial", "var")
    ii_sess = [];
else
    ii_sess = ii_combineruns(ii_trial);
    ii_sess = compute_polar(ii_sess);
    
    disp(['Total trials = ', num2str(size(ii_sess.i_sacc_err, 1))])
    disp(['nan trials ii_sess.i_sacc_err = ', num2str(sum(isnan(ii_sess.i_sacc_err)))])
    disp(['nan trials ii_sess.f_sacc_err = ', num2str(sum(isnan(ii_sess.f_sacc_err)))])
    
    % Flag trials with bad drift correction
    ii_sess.bad_drift_correct = double(cell2mat(cellfun(@(x) ismember(11, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with bad calibration
    ii_sess.bad_calibration = double(cell2mat(cellfun(@(x) ismember(12, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with fixation breaks
    ii_sess.break_fix = double(cell2mat(cellfun(@(x) ismember(13, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with no primary saccades
    ii_sess.no_prim_sacc = double(cell2mat(cellfun(@(x) ismember(20, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with small or short saccades
    ii_sess.small_sacc = double(cell2mat(cellfun(@(x) ismember(21, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with large MGS errors
    ii_sess.large_error = double(cell2mat(cellfun(@(x) ismember(22, x), ii_sess.excl_trial, 'UniformOutput', false)));

    % Put a reject trial flag: no primary saccade or a large saccade error
    ii_sess.rejtrials = double(cell2mat(cellfun(@(x) any(ismember([20, 22], x)), ii_sess.excl_trial, 'UniformOutput', false)));
    
    % Check TMS condition
    ii_sess.TMS_condition = cell(size(ii_sess.instimVF));
    ii_sess.TMS_condition(ii_sess.instimVF == 1 & ii_sess.istms == 1) = {'TMS_intoVF'};
    ii_sess.TMS_condition(ii_sess.instimVF == 0 & ii_sess.istms == 1) = {'TMS_outVF'};
    ii_sess.TMS_condition(ii_sess.istms == 0) = {'No TMS'};
end
end
