%% Edits by Mrugank (01/29/2022)
% Suppressed VBL Sync Error by PTB, added sca, clear; close all;

%% Check the system name to ensure correct paths are added.
clear; close all; clc;
[ret, hostname] = system('hostname');   
if ret ~= 0
    hostname = getenv('HOSTNAME');
end
hostname = strtrim(hostname);

%% Load PTB and toolboxes
if strcmp(hostname, 'syndrome')
    % Location of PTB on Syndrome
    addpath(genpath('/Users/Shared/Psychtoolbox')) %% mrugank (01/28/2022): load PTB
    addpath(genpath('/d/DATA/hyper/experiments/Mrugank/TMS/mgs_stimul/tmsRoom'))
elseif strcmp(hostname, 'tmsstim.cbi.fas.nyu.edu')
    % Location of toolboxes on TMS Stimul Mac
    addpath(genpath('/Users/curtislab/TMS_Priority/exp_materials/'))
elseif strcmp(hostname, 'tmsubuntu')
    addpath(genpath('/usr/lib/psychtoolbox-3'))
    addpath(genpath('/home/curtislab/Desktop/mgs_stimul/tmsRoom'))
else
    disp('Running on unknown device. Psychtoolbox might not be added correctly!')
end

sca;
KbName('UnifyKeyNames');

%   get keyboard pointer
if strcmp(hostname, 'tmsubuntu')
    devices = PsychHID('Devices', 4);
    devIdx = find(strcmp({devices(:).product}, ...
        'Mitsumi Electric Apple Extended USB Keyboard') == 1);
    kbx = 1
else
    devices = PsychHID('Devices');
    devIdx = find([devices(:).usageValue] == 6);
    %  Initialize keyboard
    if ~isempty(devIdx)
        % For remote keyboard
        kbx = devIdx(3);
        %kbx = devIdx(3);% MODIFY ACCORDING YOUR COMPUTER SETUP!!!
    else
        kbx = 0;
    end
end



%   create keyboard events queue
PsychHID('KbQueueCreate',kbx);
PsychHID('KbQueueStart',kbx);

% while 1
%     [keyIsDown, keyCode ]=PsychHID('KbQueueCheck' ,kbx);
%     if keyIsDown
%         keyName = KbName(keyCode);
%         if strcmp(keyName,'g')
%             display('ready for pulse')
%             break
%         end
%     end
%     
% end
% display('pulse sent')

for tc = 1:20
    display(['tc  ' num2str(tc) '  began' ]);
    PsychHID('KbQueueStart',kbx);
    [keyIsDown, keyCode ]=PsychHID('KbQueueCheck' ,kbx);
    cmndKey = KbName(keyCode);
    if strcmp(cmndKey,'`~')
        cmndKey = nan;
        display('Press backTick to resume')
        PsychHID('KbQueueStart',kbx);
        while ~strcmp(cmndKey,'`~')
            [keyIsDown, keyCode ]=PsychHID('KbQueueCheck' ,kbx);
            cmndKey = KbName(keyCode);
            disp(2)
        end
        
    else
        display('waiting ...')
        pause(2);
        display(['tc  ' num2str(tc) '  done' ]);        
        display('----------------------');
    end
end

