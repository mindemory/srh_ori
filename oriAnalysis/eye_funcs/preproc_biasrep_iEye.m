function preproc_biasrep_iEye(data_root, ieye_root, n_workers)
% Runs the iEye preprocessing used for Study 1 (A02_EyeAnalysis -> RuniEye ->
% run_iipreproc (via run_iipreproc_noplots: same steps, no QC figures), calibration
% skipped) on every block of the bias-replication
% study. Same ii_params as RuniEye.m; screen resolution and pixels-per-degree
% come from each block's matFile, as in RuniEye. Saccade scoring
% (ii_scoreMGS) is not run - it is MGS-specific and not needed here.
%
% Output per block: <data_root>/analysis/subXX/dayYY/EyeData/blockZZ/
%   <edf>.edf (copy), <edf>_iEye.mat (import), <edf>.mat (ii_data, ii_cfg),
%   <edf>_sacc.mat, QC figures. Blocks already processed are skipped.
%
% Usage (MATLAB):  preproc_biasrep_iEye('/Users/mrugank/Documents/TMS ori/bias_rep', ...
%                      '/Users/mrugank/Documents/localhyper/iEye', 4)

if nargin < 1, data_root = '/Users/mrugank/Documents/TMS ori/bias_rep'; end
if nargin < 2, ieye_root = '/Users/mrugank/Documents/localhyper/iEye'; end
if nargin < 3, n_workers = 4; end

addpath(genpath(ieye_root));
addpath(fileparts(mfilename('fullpath')));            % run_iipreproc
setenv('PATH', [getenv('PATH') ':/usr/local/bin']);    % edf2asc

% Same parameters as RuniEye.m
ii_params = ii_loadparams;
ii_params.valid_epochs = [1 10 11 2 3 4 5 6];
ii_params.trial_start_value = 1;
ii_params.trial_end_value = 6;
ii_params.drift_epoch = [1 10 11 2 3];
ii_params.drift_fixation_mode = 'mode';
ii_params.blink_thresh = 1;
ii_params.blink_window = [100 100];
ii_params.plot_epoch = [10 11 2 3 4 5];
ii_params.calibrate_limits = 5;
ifgFile = 'p_1000hz.ifg';

% Enumerate blocks
ori_dir = fullfile(data_root, 'data', 'ori_biasreplication_data');
jobs = struct('block_dir', {}, 'out_dir', {});
subs = dir(fullfile(ori_dir, 'sub*'));
for s = 1:numel(subs)
    days = dir(fullfile(ori_dir, subs(s).name, 'day*'));
    for d = 1:numel(days)
        blocks = dir(fullfile(ori_dir, subs(s).name, days(d).name, 'block*'));
        for b = 1:numel(blocks)
            jobs(end+1).block_dir = fullfile(blocks(b).folder, blocks(b).name); %#ok<AGROW>
            jobs(end).out_dir = fullfile(data_root, 'analysis', subs(s).name, days(d).name, 'EyeData', blocks(b).name);
        end
    end
end
fprintf('%d blocks found\n', numel(jobs));

if n_workers > 0 && isempty(gcp('nocreate')), parpool('local', n_workers); end
status = cell(numel(jobs), 1);
parfor (j = 1:numel(jobs), n_workers)   % n_workers = 0 runs serially
    status{j} = run_one(jobs(j), ii_params, ifgFile, ieye_root);
end
fid = fopen(fullfile(data_root, 'analysis', 'iEye_preproc_log.txt'), 'w');
for j = 1:numel(jobs), fprintf(fid, '%s\t%s\n', jobs(j).out_dir, status{j}); end
fclose(fid);
fprintf('%d ok, %d skipped, %d failed\n', sum(strcmp(status, 'ok')), ...
    sum(strcmp(status, 'skipped')), sum(startsWith(status, 'FAILED')));
end


function msg = run_one(job, ii_params, ifgFile, ieye_root)
try
    addpath(genpath(ieye_root));
    setenv('PATH', [getenv('PATH') ':/usr/local/bin']);
    mf = dir(fullfile(job.block_dir, 'matFile*.mat'));
    S = load(fullfile(job.block_dir, mf(end).name));
    parameters = S.matFile.parameters; screen = S.matFile.screen;
    ii_params.resolution = [screen.screenXpixels screen.screenYpixels];
    ii_params.ppd = parameters.viewingDistance * tand(1) / screen.pixSize;
    edf_src = fullfile(job.block_dir, [parameters.edfFile '.edf']);
    if ~exist(job.out_dir, 'dir'), mkdir(job.out_dir); end
    edf = fullfile(job.out_dir, [parameters.edfFile '.edf']);
    preproc_fn = edf(1:end-4);
    if exist([preproc_fn '.mat'], 'file') && exist([preproc_fn '_sacc.mat'], 'file')
        msg = 'skipped'; return
    end
    if ~exist(edf, 'file'), copyfile(edf_src, edf); end
    run_iipreproc_noplots(edf, ifgFile, preproc_fn, ii_params, [], {'calibration'});   % run_iipreproc minus QC figures
    msg = 'ok';
catch err
    msg = ['FAILED: ' strrep(err.message, newline, ' ')];
end
end
