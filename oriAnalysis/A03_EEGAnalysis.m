function A03_EEGAnalysis(subjID, day)
% Created by Mrugank:
% For Working Memory Memory-Guided Saccade EEG analysis
% The script performs preprocessing of EEG data for with and without TMS
% datasets for each subject and each session. The "step" argument lets you
% pass in which steps of preprocessing you would like to run. The current
% order is:
% Step 1: Concatenate data: if multiple sessions
% Step 2: Preprocess and epoch data to stimulus onset
% Step 3: Remove pulse artifact and interpolate (for TMS only)
% Step 4: Bandpass filter the data [0.5 100] Hz and remove line noise at
% 60Hz
% Artifact rejection (needs to be done manually and any flagged channels
% and trials should be added to flagged_trls_chans.m file
% Step 5: Reepoch data: reference to CAR and reepoch after removing bad
% trials
% Step 6: Event-related potential (ERP) for epoched data
% Step 7: Time-frequency analysis using wavelet method: ERSP, ITC and phase
% are computed
clearvars -except subjID day; close all;

%% Intialization

p.subjID = num2str(subjID,'%02d');
p.day = day;

[p, ~] = initialization(p, 'eeg', 0);

steps = {'concat'};

% List of files to be saved
% Step 1: Loading data 'subXX_dayXX_raweeg.mat'
% Step 2: Remove low frequency drifts 'subXX_dayXX_highpass.mat'
% Step 3: Epoch data 'subXX_dayXX_epoched.mat'

% File names
fName.folder                    = [p.saveEEG '/sub' p.subjID '/day' num2str(p.day,'%02d')];
if ~exist(fName.folder, 'dir')
    mkdir(fName.folder)
end
fName.general                   = [fName.folder '/sub' p.subjID '_day' num2str(p.day,'%02d')];
fName.concat                    = [fName.general '.vhdr'];

%% Concatenate EEG data
if any(strcmp(steps, 'concat'))
    if ~exist(fName.concat, 'file')
        disp('Concatenated file does not exist. Concatenating EEG data.')
        ConcatEEG(p, fName);
    else
        disp('Concatenated file exists. Skipping this step.')
    end
end

end
