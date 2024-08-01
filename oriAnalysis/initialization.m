function [p, taskMap] = initialization(p, analysis_type, prac_status)

% Adding all the folders and the helper function
tmp = pwd; tmp2 = strfind(tmp,filesep);
p.master = tmp(1:(tmp2(end)-1));
root_dir = tmp(1:(tmp2(end-2)-1));
addpath(genpath(p.master)); 

% Check the system running on: currently accepted: syndrome and my macbook
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
p.dayfolder = [p.ori '/day' num2str(p.day, '%02d')];

if prac_status == 1
    taskMapfilename = [p.phosphene '/taskMapPractice_sub' p.subjID '.mat'];
    load(taskMapfilename);
    taskMap = taskMapPractice;
%     p.tmstatus = 0; %0 if off, 1 if on
else
    taskMapfilename = [p.phosphene '/taskMap_sub' p.subjID '_day' num2str(p.day, '%02d') '.mat'];
    load(taskMapfilename);
%     p.tmstatus = taskMap(1).TMScond; %0 if off, 1 if on
end


% Folder to save analysis data
p.save = [p.analysis '/sub' p.subjID '/day' num2str(p.day, '%02d')];
p.save_eyedata = [p.save '/EyeData'];
if ~exist(p.save, 'dir')
    mkdir(p.save);
end

% Add toolbox to path (either iEye or fieldtrip)
if strcmp(analysis_type, 'eye')
    p.iEye = [root_dir '/iEye'];
    addpath(genpath(p.iEye));    
elseif strcmp(analysis_type, 'eeg')
    p.saveEEG = [p.datd '/EEGfiles'];
    addpath(p.fieldtrip);
    ft_defaults;
end
end