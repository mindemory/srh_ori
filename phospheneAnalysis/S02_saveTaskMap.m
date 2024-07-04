function S02_saveTaskMap(subjID, session, coilLocInd)
clearvars -except subjID session coilLocInd; close all; clc;

% Choose a session and coilLocInd that needs to be used for a given subject
%%% Adding all necessary paths
subjID = num2str(subjID, "%02d");
session = num2str(session, "%02d");

%%% Check the system name to ensure correct paths are added.
[ret, hostname] = system('hostname');
if ret ~= 0
    hostname = getenv('HOSTNAME');
end
hostname = strtrim(hostname);

sessContainer = containers.Map('KeyType', 'int64', 'ValueType', 'char');
sessContainer(1) = 'notms';
sessContainer(2) = 'early';
sessContainer(3) = 'middle';
sessContainer(4) = 'late';

if strcmp(hostname, 'zod.psych.nyu.edu') || strcmp(hostname, 'zod')
    master_dir = '/d/DATD/datd/TMS_ori/';
end

phosphene_data_path = [master_dir '/data/phosphene_data/sub' subjID];

addpath(genpath(phosphene_data_path));

load([phosphene_data_path '/PhospheneReport_sub' subjID '_sess' session])
for sess = 1:4
    taskMap = PhosphReport(coilLocInd).taskMap(sess, :);
    saveName_taskMap = [phosphene_data_path '/taskMap_sub' subjID '_sess_' sessContainer(sess)];
    save(saveName_taskMap,'taskMap')
end
taskMapPractice = PhosphReport(coilLocInd).taskMapPractice;
%% Save results
saveName_taskMapPractice = [phosphene_data_path '/taskMapPractice_sub' subjID];
save(saveName_taskMapPractice,'taskMapPractice')
end