function parameters = initFiles(parameters, screen, data_path, kbx)
% Create all necessary directories (a directory for each subject, containg Results and TaskMaps)
% specify the directories to be used
sessionDIR = [data_path filesep 'sub' num2str(parameters.subject,"%02d")];

% Create directories and sub-directories
if exist(data_path,'dir') ~= 7 % if the "data_path" directory doesn't exist, create one
    mkdir(data_path);
end

if exist(sessionDIR, 'dir')==7 % if the "sessionDIR" directory doesn't exist, create one. If it exists, ask if it can be overwritten.
    KbQueueStart(kbx);
    [~, keyCode] = KbQueueCheck(kbx);
    cmndKey = KbName(keyCode);
    while ~strcmp(cmndKey, parameters.space_key)
        showprompts(screen, 'SessionExists');
        [~, keyCode] = KbQueueCheck(kbx);
        cmndKey = KbName(keyCode);
        if strcmp(cmndKey, parameters.exit_key)
            sca;
            break;
        end
    end
else
    mkdir(sessionDIR);
end

% File name that will be saved
parameters.fName = [sessionDIR '/tmsRtnTpy_sub' num2str(parameters.subject,"%02d") '_sess' num2str(parameters.session,"%02d")];
end