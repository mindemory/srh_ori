function A01_GetTimeReport(subjID, prac_status)
% Created by Mrugank (02/24/2022): The file 
clearvars -except subjID proc_type prac_status; close all; %clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 2
    prac_status = 0; % default is no practice aka actual run
end
subjID = num2str(subjID, "%02d");

% Check the system running on: currently accepted: syndrome, tmsubuntu
[ret, hostname] = system('hostname');
if ret ~= 0
    hostname = getenv('HOSTNAME');
end
hostname = strtrim(hostname);
tmp = pwd; tmp2 = strfind(tmp,filesep);
direct.master = tmp(1:(tmp2(end)-1));

if strcmp(hostname, 'zod.psych.nyu.edu') || strcmp(hostname, 'loki.psych.nyu.edu') ...
   || strcmp(hostname, 'syndrome') || strcmp(hostname, 'zod')% If running on Syndrome
    direct.datd = '/d/DATD/datd/TMS_ori';
else % If running on World's best MacBook
    direct.datd = '/Users/mrugankdake/Documents/Clayspace/EEG_TMS/datd/TMS_ori';
end

sessContainer = containers.Map('KeyType', 'int64', 'ValueType', 'char');
sessContainer(1) = 'notms';
sessContainer(2) = 'early';
sessContainer(3) = 'middle';
sessContainer(4) = 'late';

% Initialize all the paths
direct.data = [direct.datd '/data'];
direct.phosphene = [direct.data '/phosphene_data/sub' subjID];
if prac_status == 1
    direct.analysis = [direct.datd '/analysis/practice/sub' subjID];
    direct.ori = [direct.data '/ori_practice_data/sub' subjID];
else
    direct.analysis = [direct.datd '/analysis/sub' subjID];
    direct.ori = [direct.data '/ori_data/sub' subjID];
end
addpath(genpath(direct.data));

direct.figures_png = [direct.analysis '/timeStruct/png'];
direct.figures_fig = [direct.analysis '/timeStruct/fig'];

% Paths to save analysis and plots
if exist(direct.analysis, 'dir') ~= 7
    mkdir(direct.analysis)
end

if exist(direct.figures_png, 'dir') ~= 7
    mkdir(direct.figures_png)
end

if exist(direct.figures_fig, 'dir') ~= 7
    mkdir(direct.figures_fig)
end

% List directories
day_dir_list = dir(direct.ori);
day_dir_names = {day_dir_list([day_dir_list.isdir]).name};
day_dirs = day_dir_names(startsWith(day_dir_names, 'day'));
days = length(day_dirs);
end_blocks = [];
for d = 1:days
    bl_dir_list = dir([direct.ori filesep day_dirs{d}]);
    bl_dir_names = {bl_dir_list([bl_dir_list.isdir]).name};
    block_dirs = bl_dir_names(startsWith(bl_dir_names, 'block'));
    end_blocks = [end_blocks, length(block_dirs)];
end
max_end_block = max(end_blocks);
start_block = 1;

% For plotting
nbinss = 50;

flag_day = [];
flag_block = [];
flag_trl = [];
for day = 1:days
    direct.sess = [direct.ori '/day' num2str(day, '%02d')];
    end_block = end_blocks(day);
    for block = start_block:end_block
        block_path = [direct.sess '/block' num2str(block, "%02d")];
        block_dir_list = dir(block_path);
        block_dir_names = {block_dir_list(~[block_dir_list.isdir]).name};
        matfileName = block_dir_names(startsWith(block_dir_names, 'matFile'));
        load([block_path filesep matfileName{1}]);
        % Load timeReport for this sess and block
        parameters = matFile.parameters;
        timeReport = matFile.timeReport;
        % Get a list of all duration variables and initialize timeStruct
        if day == 1 && block == start_block
            dur_vars = fieldnames(timeReport);
            for ii = 1:length(dur_vars)
                timeStruct.(dur_vars{ii}) = NaN(days, max_end_block, length(timeReport));
            end
        end
        % Extract all variables from timeReport for each sess and block into
        % timeStruct
        for ii = 1:length(dur_vars)
            timeStruct.(dur_vars{ii})(day, block, :) = timeReport.(dur_vars{ii});
            % Check for trials with issues
            if ii < length(dur_vars) - 1 % there is no trialDuration parameter for parameters
                % Also let's ignore flagging trials based on respDuration
                % because that does not make sense here
                if ~strcmp(dur_vars{ii}, 'respDuration')
                    if strcmp(dur_vars{ii}, 'delay1Duration') || strcmp(dur_vars{ii}, 'delay2Duration')
    %                     check_error = find(timeReport.(dur_vars{ii}) > median(timeReport.(dur_vars{ii}))*1.005);
                        this_dur = arrayfun(@(x) timeReport(x).(dur_vars{ii}), 1:length(timeReport));
                        check_error = arrayfun(@(x) (timeReport(x).(dur_vars{ii}) > median(this_dur) * 1.005), 1:length(timeReport));
                    else
    %                     check_error = find(timeReport.(dur_vars{ii}) > parameters.(dur_vars{ii})*1.005);% == 1);
                        check_error = arrayfun(@(x) (timeReport(x).(dur_vars{ii}) > parameters.(dur_vars{ii}) * 1.005), 1:length(timeReport));
                    end
                    if sum(check_error) > 0 %~isempty(check_error)
                        for e = 1:length(check_error)
                            if check_error(e) > 0
                                flag_day = [flag_day, day];
                                flag_block = [flag_block, block];
                                flag_trl = [flag_trl, check_error(e)];
                            end
                        end
                    end
                end
            end
        end
        clear matFile;
    end
    % Generate a figure of time stamps
    fig = figure('visible', 'off');
    h = sgtitle(['sub' subjID '_ sess' num2str(day, "%02d") ]);
    %set(h, 'Position', [0.5, -0.03, 0]);
    for ii = 1:length(dur_vars)
        subplot(3, 3, ii)
        histogram(rmmissing(timeStruct.(dur_vars{ii})(day,:)), 'BinWidth', 0.005);
        title(dur_vars{ii});
        xlabel('Time (s)')
        ylabel('# of Trials')
    end
    fig_fname_png = [direct.figures_png '/timeStruct_sub' subjID '_sess' num2str(day, "%02d") '.png'];
    saveas(fig, fig_fname_png);
    fig_fname_fig = [direct.figures_fig '/timeStruct_sub' subjID '_sess' num2str(day, "%02d") '.fig'];
    saveas(fig, fig_fname_fig);
end

% Create a structure to save flagged trials
flg = struct;
flg.day = flag_day; flg.block = flag_block; flg.trls = flag_trl;
flg_fname = [direct.analysis '/flagged_trls.mat'];
save(flg_fname, 'flg');

time_fname = [direct.analysis '/timeStruct.mat'];
save(time_fname, 'timeStruct');
end