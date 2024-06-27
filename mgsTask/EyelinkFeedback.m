function EyelinkFeedback()

if Eyelink('isconnected') == el.dummyconnected % in dummy mode use mousecoordinates
    [x,y,button] = GetMouse(window);
    evt.type=el.ENDSACC;
    evt.genx=x;
    evt.geny=y;
    evtype=el.ENDSACC;
else % check for events
    evtype=Eyelink('getnextdatatype');
end
if evtype==el.ENDSACC		% if the subject finished a saccade check if it fell on an object
    if Eyelink('isconnected') == el.connected % if we're really measuring eye-movements
        evt = Eyelink('getfloatdata', evtype); % get data
    end
    % check if saccade landed on an object
    choice=-1;
    noobject=0;
    i=1;
    while 1
        if 1==IsInRect(evt.genx,evt.geny, object(i).rect )
            choice=i;
            break;
        end
        i=i+1;
        if i>length(object)
            noobject=1;
            break;
        end
    end
    if lastchoice>0 && (choice~=lastchoice || noobject==1) % toggle object color
        if object(lastchoice).on==1 % restore screen
            Screen('CopyWindow', buffer, window, object(lastchoice).rect, object(lastchoice).rect);
            object(lastchoice).on=0;
            lastchoice=-1;
            doflip=1;
        end
    end
    if choice>0 && choice~=lastchoice % toggle object color
        if object(choice).on==0 % toggle object on screen
            Screen('CopyWindow', altbuffer, window, object(choice).rect, object(choice).rect);
            object(choice).on=1;
            doflip=1;
        end
        lastchoice=choice;
    end
    if doflip==1
        Screen('Flip',  window, [], 1);
        doflip=0;
    end
end % saccade?
end