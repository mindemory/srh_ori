%%
%created by israr ul haq. israruh@gmail.com. 4th April 2016
%function to run a five second fixation task in order to get the coordinates of gaze position when looking at center
%also gets the average pupil size during this time. 
%TODO:figure out a better way to import the fixation cross info
%OPEN QUESTIONS: 
%create the instructions text in here? 
%make the duration an input argument? (not going to be straightforward as the 
%size of the array and gaze position check is linked to the duration) 
%get rid of kbs? 
function [avgGazeCenter,avgPupilSize] = etFixCalibTask(parameters, screen, kbx, el, eye_used) 
    %clear the event buffer
    FlushEvents;
    %Screen('FillRect', screen.win, screen.grey, screen.screenRect);

    %create instructions for the 5 second task
    KbQueueFlush(kbx);
    KbQueueStart(kbx);
    [keyIsDown, ~]=KbQueueCheck(kbx);
    while ~keyIsDown
        showprompts(screen, 'EyeCalibStart');
        [keyIsDown, ~]=KbQueueCheck(kbx);
    end
    %fixation dot white during the 5 second task. use the fixation image as
    %used in the regular trials
    drawTextures(parameters, screen, 'FixationCross');
    
    fixationStartT = GetSecs;
    forAvgGazeCenter = zeros(9,2);
    forAvgPupilSize = zeros(9,1);
    trialCounterFix = 1;
    while GetSecs < fixationStartT + 5 %get 9 instances of gaze position readings, put them in the array and then get an average reading. cant use exact times as refresh times vary, so have to go with screen.wins of 10 ms instead
      if ((GetSecs - fixationStartT) < 0.41 && (GetSecs - fixationStartT) > 0.4) ...
         || ((GetSecs - fixationStartT) < 0.91 && (GetSecs - fixationStartT) > 0.9) ...
         || ((GetSecs - fixationStartT) < 1.41 && (GetSecs - fixationStartT) > 1.4) ...
         || ((GetSecs - fixationStartT) < 1.91 && (GetSecs - fixationStartT) > 1.9) ...
         || ((GetSecs - fixationStartT) < 2.41 && (GetSecs - fixationStartT) > 2.4) ...
         || ((GetSecs - fixationStartT) < 2.91 && (GetSecs - fixationStartT) > 2.9) ...
         || ((GetSecs - fixationStartT) < 3.41 && (GetSecs - fixationStartT) > 3.4) ...
         || ((GetSecs - fixationStartT) < 3.91 && (GetSecs - fixationStartT) > 3.9) ...
         || ((GetSecs - fixationStartT) < 4.41 && (GetSecs - fixationStartT) > 4.4)
         if parameters.eyetracker && Eyelink('NewFloatSampleAvailable') > 0
            % get the sample in the form of an event structure
            evt = Eyelink('NewestFloatSample');
            if eye_used ~= -1 % do we know which eye to use yet?
                % if we do, get current gaze position from sample
                gazePosX = evt.gx(eye_used+1); % +1 as we're accessing MATLAB array
                gazePosY = evt.gy(eye_used+1);
                pupilSize = evt.pa(eye_used+1);

                % do we have valid data and is the pupil visible?
                if gazePosX~=el.MISSING_DATA && gazePosY~=el.MISSING_DATA && evt.pa(eye_used+1)>0
                    forAvgGazeCenter(trialCounterFix,1)=gazePosX;
                    forAvgGazeCenter(trialCounterFix,2)=gazePosY;
                    forAvgPupilSize(trialCounterFix) = pupilSize;
                    trialCounterFix = trialCounterFix + 1;

                    %disp(forAvgGazeCenter)
                    %disp(avgGazeCenter)

                end
            end
         end
      end
    end
    avgGazeCenter = mean(forAvgGazeCenter);
    avgPupilSize = mean(forAvgPupilSize);
    %create end of the task screen
    %create instructions for the 5 second task
    KbQueueFlush(kbx);
    KbQueueStart(kbx);
    [keyIsDown, ~]=KbQueueCheck(kbx);
    while ~keyIsDown
        showprompts(screen, 'EyeCalibEnd');
        [keyIsDown, ~]=KbQueueCheck(kbx);
    end

end
        