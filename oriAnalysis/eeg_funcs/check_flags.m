function flags = check_flags(hdr, event_concat)
    % Collect flag type, flag number and sample/time stamps from EEG data
    flags = struct;
    flags.type = [];
    flags.num = [];
    flags.sample = [];
    % Sample rate of EEG data
    flags.Fs = hdr.Fs; % In Hz

    % Extracting flags and sample stamps from event_concat file
    flag_list = [event_concat.value];
    sample_list = [event_concat.sample];    
    
    % Make sure the flags are correct, flag_list should be a multiple of 4,
    % since all flags are of form 'Sxxx' or 'Rxxx'
    if mod(length(flag_list), 4) ~= 0
        disp('Program should have interrupted here! Issues with flags, put a stop here and debug!')
    end

    for ii = 1:floor(length(flag_list)/4)
        this_flag = flag_list(4*ii-3:4*ii);
        flags.type = [flags.type, this_flag(1)]; % extract flag type either S or R
        flags.num = [flags.num, str2num(this_flag(2:4))]; % extract flag number [1, 255]
        flags.sample = [flags.sample, sample_list(ii)]; % Get sample stamp, recording starts at 0
    end
    flags.time = [flags.sample]/flags.Fs; % in seconds
end

