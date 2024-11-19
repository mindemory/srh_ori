% Created by Mrugank (11/18/2024)
clear; close all; clc;


% p.subjID = num2str(subjID, '%02d');
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

p.phosphene = [p.data '/phosphene_data/'];
p.ori = [p.data '/ori_data/'];
p.analysis = [p.datd '/analysis'];

addpath(genpath(p.analysis));

% Check for unique subjects in analysis folder
subDirs = dir(fullfile(p.analysis, 'sub*'));
disp(['We have ' len(subDirs) ' subjects.'])
