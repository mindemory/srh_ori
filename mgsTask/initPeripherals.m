function [kbx, parameters] = initPeripherals(parameters)
% Created by Mrugank Dake, Curtis Lab, NYU (10/11/2022)

% Detects peripherals: mouse and keyboard and runs KbQueueCreate on
% each of them. Code is adapted for Mac and Ubuntu.
KbName('UnifyKeyNames');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mac Optimized
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(computer, 'MACI64')
    % get keyboard and mouse pointers for the current setup
    devices = PsychHID('Devices');
    devIdx(1) = find([devices(:).usageValue] == 6);      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ubuntu Optimized
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(computer, 'GLNXA64') % Check this
    % Ubuntu does not support PsychHID. Instead GetKeyboardIndices and
    % GetGamepadIndices is used.
    % get keyboard and mouse pointers for the current setup
    [~, ~, kboards] = GetKeyboardIndices();
    for i = 1:length(kboards)
        if strcmp(kboards{1, i}.product, 'Mitsumi Electric Apple Extended USB Keyboard') || strcmp(kboards{1, i}.product, 'Dell Dell Wired Multimedia Keyboard')
            devIdx(1) = kboards{1, i}.index;
        end
    end   
% Undetected device
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

% create keyboard and mouse events queue
KbQueueCreate(kbx);

% Response keys of interest
parameters.space_key = 'space';
parameters.exit_key = 'ESCAPE';
end