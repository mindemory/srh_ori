function S01_phosphene_analysis_biasreplication(subjID, session, coilLocInd)
% Bias-replication wrapper mirroring S01_phosphene_analysis.m.
% Reuses the subject's already-measured phosphene location (coilLocInd)
% from their original PhospheneReport rather than recomputing it.
clearvars -except subjID session coilLocInd; close all; clc;
% NOTE: This code can only run on Syndrome
warning('off', 'all')

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

%% Reuse phosphene location + build bias-replication taskMap structure
calcPhospheneArea_biasreplication(subjID, session, coilLocInd, data_path);
end
