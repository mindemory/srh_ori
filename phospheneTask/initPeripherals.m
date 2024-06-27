function [kbx, mbx, parameters] = initPeripherals(parameters, thisdev)
% Detects peripherals: mouse and keyboard and runs KbQueueCreate on
% each of them. Code is adapted for Mac and Ubuntu.

%global parameters kbx mbx hostname
KbName('UnifyKeyNames');

%% Mac Optimized
if strcmp(thisdev, 'mac')
    % get keyboard and mouse pointers for the current setup
    devices = PsychHID('Devices');
    devIdx(1) = find([devices(:).usageValue] == 6);
    devIdx(2) = find([devices(:).usageValue] == 2);
    
    parameters.left_key = 1;
    parameters.right_key = 2;
    parameters.trial_key = '1!';
    parameters.newloc_key = '2@';
    parameters.quit_key = '3#';
    
%% Ubuntu Optimized
elseif strcmp(thisdev, 'linux')
    % Ubuntu does not support PsychHID. Instead GetKeyboardIndices and
    % GetGamepadIndices is used.
    % get keyboard and mouse pointers for the current setup
    [~, ~, kboards] = GetKeyboardIndices();
    [~, ~, gpads] = GetGamepadIndices();
    for i = 1:length(kboards)
        if strcmp(kboards{1, i}.product, 'Mitsumi Electric Apple Extended USB Keyboard') || strcmp(kboards{1, i}.product, 'Dell Dell Wired Multimedia Keyboard')
            devIdx(1) = kboards{1, i}.index;
        end
    end
    
    for i = 1:length(gpads)
        if strcmp(gpads{1, i}.product, 'PixArt Dell MS116 USB Optical Mouse')
            devIdx(2) = gpads{1, i}.index;
        end
    end
    
    parameters.left_key = 1;
    parameters.right_key = 3;
    parameters.trial_key = '1';
    parameters.newloc_key = '2';
    parameters.quit_key = '3';
    
%% Undetected device
else
    disp('Running on unknown device. Psychtoolbox might not be added correctly!')
end

%% Initialize peripherals %%
% Initialize keyboard
if ~isempty(devIdx(1))
    kbx = devIdx(1);% MODIFY ACCORDING YOUR COMPUTER SETUP!!!
else
    error('External Keyboard not found')
end

% Initialize mouse
if ~isempty(devIdx(2))
    mbx = devIdx(2);% MODIFY ACCORDING YOUR COMPUTER SETUP!!!
else
    error('External Mouse not found')
end

% create keyboard and mouse events queue
KbQueueCreate(kbx);
KbQueueCreate(mbx);

% Response keys of interest
parameters.space_key = 'space';
parameters.exit_key = 'ESCAPE';
end
