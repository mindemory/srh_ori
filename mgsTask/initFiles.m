function parameters = initFiles(parameters, screen, data_path, kbx, block)
% Create all necessary directories (a directory for each subject, containg Results and TaskMaps)
% specify the directories to be used
block_dir = [data_path '/block' num2str(block, "%02d")];

% Create directories and sub-directories
if exist(data_path,'dir') ~= 7 % if the "data_path" directory doesn't exist, create one
    mkdir(data_path);
end

if exist(block_dir, 'dir')==7 % if the "block_dir" directory doesn't exist, create one. If it exists, ask if it can be overwritten.
    KbQueueStart(kbx);
    [~, keyCode] = KbQueueCheck(kbx);
    cmndKey = KbName(keyCode);
    while ~strcmp(cmndKey, parameters.space_key)
        showprompts(screen, 'BlockExists', block);
        [~, keyCode] = KbQueueCheck(kbx);
        cmndKey = KbName(keyCode);
        if strcmp(cmndKey, parameters.exit_key)
            sca;
            break;
        end
    end
else
    mkdir(block_dir);
end

% File names that will be saved.
date_time = datestr(now, 'mmddyy_HHMM');
edfFile = [parameters.subject parameters.block date_time(1:4)];
matFile = ['matFile_subj' parameters.subject '_block' parameters.block '.mat'];
eyeErrorFile = ['eyetrackerrorsFile_subj' parameters.subject '_block' parameters.block '.mat'];

parameters.block_dir = block_dir;
parameters.edfFile = edfFile;
parameters.matFile = matFile;
parameters.eyeErrorFile = eyeErrorFile;
end