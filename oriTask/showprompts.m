function showprompts(screen, prompt_name, num, doflip)
% Created by Mrugank Dake, Curtis Lab, NYU (10/11/2022)
if nargin < 4
    doflip = 1;
end
switch prompt_name
    case 'WelcomeWindow'
        if num == 1
            text = 'Welcome to the task. This is a TMS session. \n Press SPACE to continue ...';
        else
            text = 'Welcome to the task. This is a no TMS session. \n Press SPACE to continue ...';
        end
    case 'BlockExists'
        text = ['Block ' num2str(num) ' already exists. Press SPACE if you still want to continue, or ESCAPE to quit.'];
    case 'BlockStart'
        text = ['Block = ' num2str(num)];
    case 'BlockEnd'
%         text = ['End of block ' num2str(num) '. Please wait ...'];
        if num > 88
            text = ['End of block! Total points = ' num2str(round(num), "%i") ' . Amazing!!!'];
        else
            text = ['End of block! Total points = ' num2str(round(num), "%i")];
        end
    case 'ContinueorEsc'
        text = ['End of block ' num2str(num) '. Please take a break. Press SPACE to continue.'];
    case 'FeedbackErr'
        if num > 88
            text = ['Great job! ' num2str(round(num), "%i") ' points'];
        else
            text = ['Can do better! ' num2str(round(num), "%i") ' points'];
        end
    case 'CurrOri'
        text = ['Current Orientation = ' num2str(round(num), '%i') 'º'];
    case 'EyeCalibStart'
        text = sprintf('Please press any key and \n keep focused at the central fixation point for 5 seconds');
    case 'EyeCalibEnd'
        text = sprintf('Thank you. When ready, press key to continue.');
    case 'TrialPause'
        text = 'Experiment is paused! Press SPACE to resume.';
    case 'EndExperiment'
        text = 'Thank you for your participation!';
end
fontsize = 30;
text_color = screen.white;
Screen('TextSize', screen.win, fontsize);
DrawFormattedText(screen.win, text, 'center', 'center', text_color);
if doflip == 1
    Screen('Flip', screen.win);
end
end