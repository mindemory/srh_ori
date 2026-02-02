function ConcatEEG(p, fName, run_check)
% Created by Mrugank: 12/08/2022: Extracts header, data and event files for
% each subject ran on each day and concatenates them into one file.

if nargin < 3
    run_check = 1;
end

delete([p.EEGData '/._*'])
fNames = dir([p.EEGData '/*.vhdr']);

sample_count = 0;
for ii=1:length(fNames)
    EEGfile = [p.EEGData filesep fNames(ii).name];
    disp(['Working on file ' fNames(ii).name]);
    
    % extract header, data and event from EEGfile
    hdr = ft_read_header(EEGfile);
    dat = ft_read_data(EEGfile);
    evt = ft_read_event(EEGfile);
    if ii == 1
        event_concat = evt;
    else
        for jj=1:length(evt)
            evt(jj).sample = evt(jj).sample + sample_count;
        end
        event_concat = [event_concat, evt];
    end
    sample_count = sample_count + hdr.nSamples;
    if ii == 1
        data_concat = dat;
    else
        data_concat = [data_concat, dat];
    end
end
hdr.nSamples = sample_count;

% Fix the redundant flag issue (Added on 03/17/2023)
drop_evts = [];
for evt_this = 1:length(event_concat)
    % Eliminate New Segment elements
    if strcmp(event_concat(evt_this).type, 'New Segment')
        drop_evts = [drop_evts, evt_this];
    end
    % Remove S 15 and R 15 flags, coming from closing the port
    if strcmp(event_concat(evt_this).value, 'S 15') || strcmp(event_concat(evt_this).value, 'R 15')
        drop_evts = [drop_evts, evt_this];
    end
end
event_concat(drop_evts) = [];

% Remove the flags that are repeated
drop_evts = [];
for evt_this = 1:length(event_concat)
    if evt_this > 1
        if strcmp(event_concat(evt_this).value, event_concat(evt_this-1).value)
            drop_evts = [drop_evts, evt_this];
        end
    end
end
event_concat(drop_evts) = [];

disp(['We have a total of ' num2str(length(event_concat)) ' flags in this dataset']);

% Check timing of flags
if run_check
    disp('Checking timings for flags.')
    flags = check_flags(hdr, event_concat);
    save([p.save '/EEGflags.mat'],'flags')
else
    disp('Timing of flags not checked. Analyses based on precise time-stamps could be incorrect.')
end

ft_write_data(fName.concat, data_concat, 'header', hdr, 'event', event_concat);
end
