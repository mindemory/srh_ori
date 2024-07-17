function [ii_sess] = RuniEye_loadonly(p, taskMap, end_block)

if nargin < 3
    end_block = 10;
end

% List of epochs in the task
% XDAT 1: Fixation window =  1s
% XDAT 10: Sample window =  0.5s (would correspond to 'S 11', 'S 12', 'S
% 13', 'S 14' in EEG flags depending on the stimulus VF and block type
% XDAT 2: Delay 1 = 2s
% XDAT 3: Delay 2 = 2s
% XDAT 4: Response window = 0.85s
% XDAT 5: Feedback window = 0.8s
% XDAT 6: ITI window = 2s or 3s (50-50 split)
ifgFile = 'p_1000hz.ifg';
ii_params = ii_loadparams; % load default set of analysis parameters, only change what we have to
ii_params.valid_epochs =[1 10 2 3 4 5 6]; % Updated epochs (Mrugank: 04/05/2023)
ii_params.trial_start_value = 1; %XDAT value for trial start
ii_params.trial_end_value = 6;   % XDAT value for trial end
ii_params.drift_epoch = [1 10 2 3]; % XDAT values for drift correction (
ii_params.drift_fixation_mode  = 'mode';
ii_params.calibrate_epoch = 5; % XDAT value for when we calibrate (feedback stim)
ii_params.calibrate_select_mode = 'last'; % how do we select fixation with which to calibrate?
ii_params.calibrate_mode = 'scale'; % scale: trial-by-trial, rescale each trial; 'run' - run-wise polynomial fit
ii_params.blink_thresh = 0.1;
ii_params.blink_window = [100 100]; % how long before/after blink (ms) to drop?
ii_params.plot_epoch = [10 2 3 4 5];  % what epochs do we plot for preprocessing?
ii_params.calibrate_limits = [2.5]; % when amount of adj exceeds this, don't actually calibrate (trial-wise); ignore trial for polynomial fitting (run)

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
    timeReport = matFile.timeReport;
    trialInfo{block} = timeReport;
    eyecond = taskMap(block).condition;
    
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
    ii_init;

    % import data
    matfile_iEye = [preproc_fn '_iEye.mat'];
    if exist(matfile_iEye, 'file') == 2
        load(matfile_iEye);
    else
        [ii_data,ii_cfg] = ii_import_edf(edfFile, ifgFile,[edfFile(1:end-4) '_iEye.mat']);
        fig = figure('visible','off'); plot(ii_data.XDAT, 'ro-'); xlabel('Time (ms)'); ylabel('XDAT'); title('Sanity check of XDAT');
        saveas(fig, [preproc_fn '_sanitycheck.png'],'png');
    end
    clearvars ii_cfg ii_data;
end
end
