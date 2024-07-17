% This was created specifically for sub04, day02, block07
% Here, eyelink storage got full, so the recording stopped after trial 30
% for some reason.
% To make sure that these trials can still be used in the downstream
% pipeline, this code attempts 10 fake trials without saccade traces.
% Hence, the added 10 trials will be eliminated from the analysis, but will
% function has empty holders.

% Run this script after the data is imported from edf file.
% Continue running ii_preproc from ii_trim


% Run this section to examine the flag types and number of samples to add
% for the same
% flag_changer = diff(ii_data.XDAT);
% samp_length = NaN(length(unique(ii_data.XDAT)), 100);
% counter_xdat = zeros(length(unique(ii_data.XDAT)), 1);
% sample_holder = [];
% for ii = 1:length(flag_changer)
%     if flag_changer(ii) == 1 || flag_changer(ii) == -6
%         this_xdat = ii_data.XDAT(ii);
%         counter_xdat(this_xdat+1) = counter_xdat(this_xdat+1)+1;
%         if isempty(sample_holder)
%             samp_length(this_xdat+1, counter_xdat(this_xdat+1)) = ii;
%         else
%             samp_length(this_xdat+1, counter_xdat(this_xdat+1)) = ii - sample_holder(end);
%         end
%         sample_holder = [sample_holder, ii];
%     end
% end

num_trls_to_add = 10;
len_vals = [1000, 500, 2000, 2000, 850, 800, 2000];
for tr = 1:num_trls_to_add
    for epoc = 1:7
        ii_data.X = [ii_data.X; mean(ii_data.X, 'all', 'omitnan') * ones(len_vals(epoc), 1)];
        ii_data.Y = [ii_data.Y; mean(ii_data.Y, 'all', 'omitnan') * ones(len_vals(epoc), 1)];
        ii_data.Pupil = [ii_data.Pupil; mean(ii_data.Pupil, 'all', 'omitnan') * ones(len_vals(epoc), 1)];
        ii_data.TarX = [ii_data.TarX; mean(ii_data.TarX(ii_data.XDAT == epoc)) * ones(len_vals(epoc), 1)];
        ii_data.TarY = [ii_data.TarY; mean(ii_data.TarY(ii_data.XDAT == epoc)) * ones(len_vals(epoc), 1)];
        ii_data.XDAT = [ii_data.XDAT; epoc * ones(len_vals(epoc), 1)];
        
    end
end
ii_cfg.sel = zeros(length(ii_data.XDAT), 1);
ii_cfg.tsel = zeros(length(ii_data.XDAT), 1);