function S02_saveTaskMap(subjID, session, coilLocInd)
clearvars -except subjID session coilLocInd; close all; clc;

rng('shuffle');
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

% Create taskMap for the main sessions, organized by session
load([phosphene_data_path '/PhospheneReport_sub' subjID '_sess' session])
for sess = 1:4
    taskMap = PhosphReport(coilLocInd).taskMap(sess, :);
    saveName_taskMap = [phosphene_data_path '/taskMap_sub' subjID '_sess_' sessContainer(sess)];
    save(saveName_taskMap,'taskMap')
end
clearvars taskMap;
% For the main task, we want to organize the sessions such that the
% experiment is performed over 2 days. On each day, the participant
% performs 3 blocks of NoTMS, 5 blocks of early TMS and 5 blocks of middle
% TMS, the order of which is randomized across subjects
condBlocks = [3, 5, 5];
sessUsed = [1, 2, 3];
for dd = 1:2
    thisSess = sessUsed(randperm(length(sessUsed))); % shuffle the session order for this day
    for ii = 1:length(sessUsed)
        % Read taskMap for the given session 
        tMap_thiscond = PhosphReport(coilLocInd).taskMap(thisSess(ii), ...
                                    condBlocks(thisSess(ii))*(dd-1)+1: ...
                                    condBlocks(thisSess(ii))*dd);
        % Add it to the main taskMap for this day recursively
        if ii == 1
            taskMap = tMap_thiscond;
        else
            taskMap = [taskMap, tMap_thiscond];
        end
    end
    saveName_taskMap = [phosphene_data_path '/taskMap_sub' subjID '_day' num2str(dd, '%02d')];
    save(saveName_taskMap,'taskMap')
end


% Create taskMap for the practice session
taskMapPractice = PhosphReport(coilLocInd).taskMapPractice;
saveName_taskMapPractice = [phosphene_data_path '/taskMapPractice_sub' subjID];
save(saveName_taskMapPractice,'taskMapPractice')
end