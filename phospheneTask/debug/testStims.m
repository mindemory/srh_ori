function testStims()

global screen
global parameters

subjID = input(sprintf('\nsubject: '),'s');
session = input(sprintf('\nsession: '),'s');

dataDIR = ['Results/subj_' subjID];
data = [dataDIR '/Stim_subj_' subjID '_sess_' session];
load(data);

loadParameters()
initScreen()

% fixation cross
FixCross = [screen.xCenter-1,screen.yCenter-4,screen.xCenter+1,screen.yCenter+4;...
            screen.xCenter-4,screen.yCenter-1,screen.xCenter+4,screen.yCenter+1];
Screen('FillRect', screen.win, [0,0,128], FixCross');
Screen('Flip', screen.win);
pause
for coilLocInd = 1:length(Stim)
    
    thisStim = Stim{coilLocInd};
    stimCoords = thisStim.stimCoords;
    
    for trialInd = 1:size(stimCoords,1)
        rect = [stimCoords(trialInd,1)-2 stimCoords(trialInd,2)-2 stimCoords(trialInd,1)+2 stimCoords(trialInd,2)+2];
        Screen('FillOval', screen.win, screen.white, rect);
        Screen('FillRect', screen.win, [0,0,128], FixCross');
        Screen('Flip', screen.win);
        display(['Coil location : ' num2str(coilLocInd)]);
        display(['Sample number : ' num2str(trialInd)]);
        display('------------------------');
        pause
    end
end
sca
