function [ii_data,ii_cfg,ii_sacc] = run_iipreproc_noplots(edf_fn,cfg_fn,preproc_fn,ii_params,trialinfo,skip_steps)
% Same preprocessing stream as run_iipreproc.m (identical iEye calls, order
% and parameters), without the QC figures. Use where MATLAB cannot render
% figures (e.g. headless batch runs); the saved ii_data / ii_cfg / ii_sacc
% are the same as run_iipreproc's.

ii_init;
if nargin < 4 || isempty(ii_params), ii_params = ii_loadparams; end
if ischar(ii_params), ii_params = ii_loadparams(ii_params); end
if nargin < 6 || isempty(skip_steps), skip_steps = {}; end

% import data
matfile_iEye = [preproc_fn '_iEye.mat'];
if exist(matfile_iEye, 'file') == 2
    load(matfile_iEye);
else
    [ii_data,ii_cfg] = ii_import_edf(edf_fn,cfg_fn,[edf_fn(1:end-4) '_iEye.mat']);
end

[ii_data,ii_cfg] = ii_trim(ii_data,ii_cfg,ii_params.valid_epochs,ii_params.epoch_chan);
[ii_data,ii_cfg] = ii_rescale(ii_data,ii_cfg,{'X','Y'},ii_params.resolution,ii_params.ppd);
[ii_data,ii_cfg] = ii_rescale(ii_data,ii_cfg,{'TarX','TarY'},ii_params.resolution,ii_params.ppd);
[ii_data,ii_cfg] = ii_invert(ii_data,ii_cfg,'Y');
[ii_data,ii_cfg] = ii_invert(ii_data,ii_cfg,'TarY');
[ii_data,ii_cfg] = ii_censorchans(ii_data,ii_cfg,{'X','Y'},...
    {[-0.5 0.5]*ii_params.resolution(1)/ii_params.ppd,...
     [-0.5 0.5]*ii_params.resolution(2)/ii_params.ppd});
[ii_data, ii_cfg] = ii_blinkcorrect(ii_data,ii_cfg,{'X','Y'},'Pupil', ...
      ii_params.blink_thresh,ii_params.blink_window(1),ii_params.blink_window(2),'prctile');
[ii_data,ii_cfg] = ii_definetrial(ii_data,ii_cfg,...
    ii_params.trial_start_chan, ii_params.trial_start_value,...
    ii_params.trial_end_chan,   ii_params.trial_end_value);
[ii_data,ii_cfg] = ii_smooth(ii_data,ii_cfg,{'X','Y'},ii_params.smooth_type,ii_params.smooth_amt);
[ii_data,ii_cfg] = ii_velocity(ii_data,ii_cfg,'X_smooth','Y_smooth');
[ii_data,ii_cfg] = ii_findsaccades(ii_data,ii_cfg,'X_smooth','Y_smooth',...
    ii_params.sacc_velocity_thresh,ii_params.sacc_duration_thresh,ii_params.sacc_amplitude_thresh);
[ii_data,ii_cfg] = ii_findfixations(ii_data,ii_cfg,{'X','Y'},ii_params.fixation_mode);

if ~ismember('drift',skip_steps)
    [ii_data,ii_cfg] = ii_selectfixationsbytrial( ii_data, ii_cfg, ii_params.epoch_chan,...
        ii_params.drift_epoch, ii_params.drift_fixation_mode );
    [ii_data,ii_cfg] = ii_driftcorrect(ii_data,ii_cfg,{'X','Y'},ii_params.drift_select_mode,...
        ii_params.drift_target);
end
if nargin>=5 && ~isempty(trialinfo)
    [ii_data,ii_cfg] = ii_addtrialinfo(ii_data,ii_cfg,trialinfo);
end
if ~ismember('calibration',skip_steps)
    [ii_data,ii_cfg] = ii_selectfixationsbytrial(ii_data,ii_cfg,ii_params.epoch_chan,...
        ii_params.calibrate_epoch,ii_params.calibrate_select_mode,ii_params.calibrate_window,...
        ii_params.calibrate_target,{'X_fix','Y_fix'});
    if ismember(ii_params.calibrate_mode,{'scale','rotate'})
        [ii_data,ii_cfg] = ii_calibratebytrial(ii_data,ii_cfg,{'X','Y'},...
            ii_params.calibrate_target,ii_params.calibrate_mode, ii_params.calibrate_limits);
    elseif ismember(ii_params.calibrate_mode,'run')
        [ii_data,ii_cfg] = ii_calibratebyrun(ii_data,ii_cfg,{'X','Y'},...
            ii_params.calibrate_target,ii_params.calibrate_deg,ii_params.calibrate_limits);
    end
end

ii_cfg.params = ii_params;
ii_savedata(ii_data,ii_cfg,preproc_fn);
[ii_data,ii_cfg,ii_sacc] = ii_extractsaccades(ii_data,ii_cfg,{'X','Y'},...
    ii_params.extract_sacc_mode_start,ii_params.extract_sacc_mode_end,ii_params.epoch_chan);
ii_savesacc(ii_cfg,ii_sacc,[preproc_fn '_sacc.mat']);
end
