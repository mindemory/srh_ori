function S02_saveTaskMap_biasreplication(subjID, session, coilLocInd)
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

if strcmp(hostname, 'zod.psych.nyu.edu') || strcmp(hostname, 'zod')
    master_dir = '/d/DATD/datd/TMS_ori/';
end

phosphene_data_path = [master_dir '/data/phosphene_data/sub' subjID];

addpath(genpath(phosphene_data_path));

% Load the bias-replication phosphene report (built by
% calcPhospheneArea_biasreplication.m). taskMap(1,:) holds the 8 No-TMS
% blocks, taskMap(2,:) holds the 8 Middle-TMS blocks.
load([phosphene_data_path '/PhospheneReport_biasrep_sub' subjID '_sess' session]) % loads PhosphReport

condCodeForSess = [1, 3]; % TMScond value stored at taskMap sess-index 1 and 2 respectively
sessForCond = containers.Map(condCodeForSess, [1, 2]);

% For the bias-replication task, each day gets 4 blocks of No-TMS and 4
% blocks of Middle-TMS, GROUPED (one condition run straight through, then
% the other). Which condition goes first is randomized per subject on
% day01; day02 gets the inverted order, so a subject who got No-TMS first
% on day01 gets Middle-TMS first on day02, and vice versa.
condOrderDay1 = condCodeForSess(randperm(length(condCodeForSess)));
condOrderDay2 = fliplr(condOrderDay1);
dayOrders = {condOrderDay1, condOrderDay2};

for dd = 1:2
    blockRange = (dd-1)*4 + 1 : dd*4; % day01 uses blocks 1-4 of each condition, day02 uses blocks 5-8
    thisOrder = dayOrders{dd};
    taskMap = [];
    for cc = 1:length(thisOrder)
        sessIdx = sessForCond(thisOrder(cc));
        tMap_thiscond = PhosphReport(coilLocInd).taskMap(sessIdx, blockRange);
        if cc == 1
            taskMap = tMap_thiscond;
        else
            taskMap = [taskMap, tMap_thiscond];
        end
    end
    saveName_taskMap = [phosphene_data_path '/taskMap_biasrep_sub' subjID '_day' num2str(dd, '%02d')];
    save(saveName_taskMap, 'taskMap')
end

% Create taskMap for the practice session
taskMapPractice = PhosphReport(coilLocInd).taskMapPractice;
saveName_taskMapPractice = [phosphene_data_path '/taskMapPractice_biasrep_sub' subjID];
save(saveName_taskMapPractice, 'taskMapPractice')
end
