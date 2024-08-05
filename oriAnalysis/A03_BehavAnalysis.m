function A03_BehavAnalysis(subjID, prac_status)
% Created by Mrugank (07/31/2024)
clearvars -except subjID prac_status;
close all; clc;

if nargin < 2
    prac_status = 0;
end

p.subjID = num2str(subjID, '%02d');
% p.day = day;

% Initialize all the relevant paths
% Check the system running on: currently accepted: zod and my macbook
[ret, hostname] = system('hostname');
if ret ~= 0
    hostname = getenv('HOSTNAME');
end
p.hostname = strtrim(hostname);

% Initialize all the paths
if strcmp(p.hostname, 'syndrome') || strcmp(p.hostname, 'vader') || strcmp(p.hostname, 'thanos') || ...
        strcmp(p.hostname, 'zod') || strcmp(p.hostname, 'zod.psych.nyu.edu') || strcmp(p.hostname, 'loki.psych.nyu.edu')% If running on Syndrome or Vader or Zod
    p.datd = '/d/DATD/datd/TMS_ori';
    p.fieldtrip = '/d/DATA/hyper/software/fieldtrip-20220104/';
else % If running on World's best MacBook
    p.datd = '/Users/mrugankdake/Documents/Clayspace/EEG_TMS/datd/TMS_ori';
    p.fieldtrip = '/Users/mrugankdake/Documents/MATLAB/fieldtrip-20220104/';
end
p.data = [p.datd '/data'];
addpath(genpath(p.data));

p.phosphene = [p.data '/phosphene_data/sub' p.subjID];
if prac_status == 1
    p.ori = [p.data '/ori_practice_data/sub' p.subjID];
    p.analysis = [p.datd '/analysis/practice'];
else
    p.ori = [p.data '/ori_data/sub' p.subjID];
    p.analysis = [p.datd '/analysis'];
end
addpath(genpath(p.analysis));

% analysis file
p.df_sub = [p.analysis '/sub' p.subjID filesep 'sub' p.subjID '_behav.csv'];
if exist(p.df_sub, 'file')
    disp('Data already exists. Skipping the analysis');
else
    data = table();
    day_dirs = dir(p.ori);
    day_dirs = day_dirs([day_dirs.isdir] & startsWith({day_dirs.name}, 'day') & ...
        ~ismember({day_dirs.name}, {'.', '..'}));

    for didx = 1:length(day_dirs)
        block_dirs = dir([p.ori '/' day_dirs(didx).name]);
        block_dirs = block_dirs([block_dirs.isdir] & startsWith({block_dirs.name}, 'block') & ...
            ~ismember({block_dirs.name}, {'.', '..'}));

        % Load taskMap for this day
        taskMap = load([p.phosphene '/taskMap_sub' p.subjID '_day' num2str(didx, '%02d') '.mat']);
        taskMap = taskMap.taskMap;

        for bidx = 1:length(block_dirs)
            disp(bidx)
            data_dir = [p.ori '/' day_dirs(didx).name '/' block_dirs(bidx).name];
            data_files = dir([data_dir '/*.mat']);
            relev_fidx = find(arrayfun(@(x) startsWith(data_files(x).name, 'matFile'), 1:length(data_files)));
            data_file = [data_dir '/' data_files(relev_fidx).name];
            matFile = load(data_file);
%             if isfield(matFile, 'matFile')
                matFile = matFile.matFile;
%             else
%                 meh = 0;
%             end
               
            % Extract screen coordinates
            xCenter = matFile.screen.xCenter;
            yCenter = matFile.screen.yCenter;

            numTrials = length(matFile.timeReport);
            tempData = table();

            % Extract subjID, day, and block
            tempData.subjID = repmat(subjID, numTrials, 1);
            tempData.day = repmat(didx, numTrials, 1);
            tempData.block = repmat(bidx, numTrials, 1);

            % Extract taskMap features
            tempData.TMScond = repmat(taskMap(bidx).TMScond, numTrials, 1);
            tempData.stimPF = taskMap(bidx).stimVF;
            tempData.targOriCategory = taskMap(bidx).targOri';
            tempData.targOriActual = taskMap(bidx).targOri_';
            tempData.stimSize = repmat(taskMap(bidx).stimSize, numTrials, 1);
            tempData.targX = taskMap(bidx).targX;
            tempData.targY = taskMap(bidx).targY;
            tempData.rho = sqrt(((taskMap(bidx).targX - xCenter).^2 + ...
                                (taskMap(bidx).targY - yCenter).^2));
            tempData.polang = atan2d(taskMap(bidx).targY - yCenter, ...
                                    taskMap(bidx).targX - xCenter);
            tempData.polang(tempData.polang < 0) = tempData.polang(tempData.polang < 0) + 360;

            % Extract timeReport values
            fieldnames_timeReport = fieldnames(matFile.timeReport);
            for fnidx = 1:length(fieldnames_timeReport)
                tempData.(fieldnames_timeReport{fnidx}) = arrayfun(@(x) matFile.timeReport(x).(fieldnames_timeReport{fnidx}), 1:numTrials)';
            end

            % Extract subject responses
            fieldnames_respReport = fieldnames(matFile.respReport);
            for fnidx = 1:length(fieldnames_respReport)
                tempData.(fieldnames_respReport{fnidx}) = arrayfun(@(x) matFile.respReport(x).(fieldnames_respReport{fnidx}), 1:numTrials, 'UniformOutput', false)';
            end

            data = [data; tempData];
            clearvars tempData;

        end
        clearvars taskMap;
    end

    % Save data
    writetable(data, p.df_sub);
end


% data_dir = [p.ori ]/

end