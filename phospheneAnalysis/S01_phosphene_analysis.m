function S01_phosphene_analysis(subjID, session)
clearvars -except subjID session anti_type; close all; clc;
% NOTE: This code can only run on Syndrome
warning('off','all')

subjID = num2str(subjID, "%02d");
session = num2str(session, "%02d");

%%% Check the system name to ensure correct paths are added.
[ret, hostname] = system('hostname');
if ret ~= 0
    hostname = getenv('HOSTNAME');
end
hostname = strtrim(hostname);

if strcmp(hostname, 'zod.psych.nyu.edu') || strcmp(hostname, 'zod')
    master_dir = '/d/DATD/datd/TMS_ori/';
else
    master_dir = '/Users/mrugankdake/Documents/Clayspace/EEG_TMS/datd/TMS_ori/';
end

data_path = [master_dir '/data/phosphene_data/sub' subjID];
figures_path = [master_dir '/Figures/phosphene_data/sub' subjID];

%% Compute Overlapping Phosphene Area and Target Locations
calcPhospheneArea(subjID, session, data_path, figures_path);