function A02_EyeAnalysis(subjID, day, end_block, prac_status, only_edf2asc)
% Created by Mrugank (04/09/2023) 
clearvars -except subjID day end_block prac_status only_edf2asc;
close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3
    end_block = 20; % default number of blocks on any given day
end
if nargin < 4
    prac_status = 0; % default is no practice aka actual run
end
if nargin < 5
    only_edf2asc = 0; % default is run whole analysis
end

p.subjID = num2str(subjID, '%02d');
p.day = day;

% Initialize all the relevant paths
[p, taskMap] = initialization(p, 'eye', prac_status);

%% Load ii_sess files
tic
ii_sess_saveName = [p.save '/ii_sess_sub' p.subjID '_day' num2str(p.day, '%02d')];
ii_sess_saveName_mat = [ii_sess_saveName '.mat'];
if exist(ii_sess_saveName_mat, 'file') == 2
    disp('Loading existing ii_sess file.')
    load(ii_sess_saveName_mat);
else
    if ~only_edf2asc
        disp('ii_sess file does not exist. running ieye')
        ii_sess = RuniEye(p, taskMap, end_block, prac_status);
        save(ii_sess_saveName,'ii_sess')
    else
        disp('Running import edf2asc only.')
        RuniEye_loadonly(p, taskMap, end_block);
    end
end


%% QC plots
if ~only_edf2asc
    % Run QC
    which_excl = [20 22];
    disp('Running QC')
    RunQC_EyeData(ii_sess, p, which_excl, {'all_trials'});
    %RunQC_EyeData(ii_sess, p, which_excl);
end
toc
end