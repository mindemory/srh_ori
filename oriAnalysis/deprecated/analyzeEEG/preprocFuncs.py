import os, mne
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

montage = mne.channels.make_standard_montage("easycap-M1")

def autoRejectChans(data):
    thresh_qtl = 0.05
    numVars = 10
    var_winSize = data.get_data().shape[1] // numVars
    var_ch = []
    for start in range(0, data.get_data().shape[1], var_winSize):
        var_ch.append(np.var(data.get_data()[:, start:start+var_winSize], axis=1))
    mean_var = np.mean(var_ch, axis=0)
    
    quantiles_var = np.quantile(mean_var, [thresh_qtl, 1-thresh_qtl])
    for idx, chan in enumerate(data.ch_names):
        if mean_var[idx] < quantiles_var[0] or mean_var[idx] > quantiles_var[1]:
            data.info['bads'].append(chan)
    # Eliminate bad channels
    data.drop_channels(data.info['bads'])
    return data
        

def getRawData(p):
    # List all the files in the directory
    f_list = os.listdir(p['EEGfiles'])
    f_list = [f for f in f_list if f.endswith('.vhdr')]
    if len(f_list) == 1:
        raw_fpath = os.path.join(p['EEGpy'], f_list[0].replace('.vhdr', '.fif'))
        if os.path.exists(raw_fpath):
            print(f"Loading pre-existing data ...")
            raw_data = mne.io.read_raw_fif(raw_fpath, preload=True)
        else:
            print(f"Running pre-processing ...")
            raw_data = mne.io.read_raw_brainvision(p['EEGfiles'] + '/' + f_list[0], preload=True)
            raw_data.drop_channels(['LM', 'RM'])
            raw_data.set_montage(montage)
            raw_data.filter(l_freq=0.5, h_freq=55, verbose=False, n_jobs=4)
            raw_data = autoRejectChans(raw_data)
            # raw_data.set_eeg_reference(ref_channels=['Cz'])
            # raw_data = mne.add_reference_channels(raw_data, ref_channels='Cz')
            # Rereference the data to the average of all channels
            # raw_data.set_eeg_reference(ref_channels='average')
            # raw_data.save(raw_fpath, overwrite=False)
    return raw_data


def getTrials(raw_data, events_dict, events):
    block_onset_id = events_dict['BlockOnset']
    block_end_id = events_dict['BlockEnd']
    # Find the start and end events
    startBlock_events = events[events[:, 2] == block_onset_id]
    endBlock_events = events[events[:, 2] == block_end_id]
    sampBuffer = 1000
    # Create custom epochs
    # block_epochs_list = []
    trials_list = []

    for blkIdx, startBlock_event in enumerate(startBlock_events):
        print(f"Running block {blkIdx + 1} of {len(startBlock_events)}")
        # Find the next end event after the current start event
        endBlock_event = endBlock_events[endBlock_events[:, 0] > startBlock_event[0]][0]

        # Create events_block to be used for trial definition
        events_block = events[(events[:, 0] > startBlock_event[0]) & (events[:, 0] < endBlock_event[0])]

        # Define the time window for the block epoch and epoch the block
        tminBlock = (startBlock_event[0] - raw_data.first_samp) / raw_data.info['sfreq']
        tmaxBlock = (endBlock_event[0] - raw_data.first_samp) / raw_data.info['sfreq']
        block = raw_data.copy().crop(tminBlock, tmaxBlock)
        # Apply a highpass filter to the block
        # Find the start and end events for each trial
        startTrial_events = events_block[events_block[:, 2] == events_dict['Fixation']]
        endTrial_events = events_block[events_block[:, 2] == events_dict['Feedback']]

        # for startTrial_event in startTrial_events:
        for trlIdx, startTrial_event in enumerate(startTrial_events):
            print(f"    Running trial {trlIdx + 1} of {len(startTrial_events)}")
            # Find the next end event after the current start event
            endTrial_event = endTrial_events[endTrial_events[:, 0] > startTrial_event[0]][0]

            # Define the time window for the trial epoch
            tminTrial = (startTrial_event[0] - sampBuffer - block.first_samp) / block.info['sfreq']
            tmaxTrial = (endTrial_event[0] + sampBuffer - block.first_samp) / block.info['sfreq']

            # Create the trial epoch
            trial = block.copy().crop(tminTrial, tmaxTrial)
            trials_list.append(trial)
    return trials_list

# def computePower():
#     for chIdx, channel in enumerate(range(trial_data.shape[0])):
#         channel_data = trial_data[channel, :]
#         filtered_data = mne.filter.filter_data(channel_data, sfreq, lFreq, hFreq, verbose=False)
#         for start in range(0, len(filtered_data) - int(window_size*sfreq) + 1, int(overlap*sfreq)):
#             if channel == 0:
#                 # trial_time.append(start/sfreq)
#                 twindow_power[trlIdx, start] = start/sfreq
#             end = start + int(window_size*sfreq)
#             window_data = filtered_data[start:end]
#             power = np.mean(np.abs(window_data)**2)
#     return power

# def getPower(trials_list, window_size, overlap, freq_band):
#     if freq_band == 'alpha':
#         lFreq, hFreq = 8, 12
#     elif freq_band == 'beta':
#         lFreq, hFreq = 12, 30
#     elif freq_band == 'gamma':
#         lFreq, hFreq = 30, 55
#     elif freq_band == 'theta':
#         lFreq, hFreq = 4, 8
#     elif freq_band == 'delta':
#         lFreq, hFreq = 0.5, 4
#     else:
#         lFreq, hFreq = 0.5, 55
#     # For each trial in trials_list compute alpha power over time with 100 ms windows and 50 ms overlap
#     temporalPower = np.empty((len(trials_list), len(trials_list[0].ch_names), len(trials_list[0].times)))
#     twindow_power = np.empty((len(trials_list), len(trials_list[0].times)))

#     for trlIdx, trial in enumerate(range(len(trials_list))):
#         print(f"Computing alpha power for trial {trial+1}/{len(trials_list)}")

#         trial_data = trials_list[trial].get_data()
#         sfreq = trials_list[trial].info['sfreq']

        

#                 temporalPower[trlIdx, chIdx, start:end] = power
#     return temporalPower, twindow_power


def getPower(trials_list, window_size, overlap, freq_band):
    if freq_band == 'alpha':
        lFreq, hFreq = 8, 12
    elif freq_band == 'beta':
        lFreq, hFreq = 12, 30
    elif freq_band == 'gamma':
        lFreq, hFreq = 30, 55
    elif freq_band == 'theta':
        lFreq, hFreq = 4, 8
    elif freq_band == 'delta':
        lFreq, hFreq = 0.5, 4
    else:
        lFreq, hFreq = 0.5, 55
    # For each trial in trials_list compute alpha power over time with 100 ms windows and 50 ms overlap
    temporalPower = np.empty((len(trials_list), len(trials_list[0].ch_names), len(trials_list[0].times)))
    twindow_power = np.empty((len(trials_list), len(trials_list[0].times)))

    for trlIdx, trial in enumerate(range(len(trials_list))):
        print(f"Computing alpha power for trial {trial+1}/{len(trials_list)}")

        trial_data = trials_list[trial].get_data()
        sfreq = trials_list[trial].info['sfreq']

        for chIdx, channel in enumerate(range(trial_data.shape[0])):
            channel_data = trial_data[channel, :]
            filtered_data = mne.filter.filter_data(channel_data, sfreq, lFreq, hFreq, verbose=False)
            for start in range(0, len(filtered_data) - int(window_size*sfreq) + 1, int(overlap*sfreq)):
                if channel == 0:
                    # trial_time.append(start/sfreq)
                    twindow_power[trlIdx, start] = start/sfreq
                end = start + int(window_size*sfreq)
                window_data = filtered_data[start:end]
                power = np.mean(np.abs(window_data)**2)

                temporalPower[trlIdx, chIdx, start:end] += power#(temporalPower[trlIdx, chIdx, start:end] + power) / 2
    # temporalPower /= int(window_size/overlap) - 1
    return temporalPower, twindow_power


def RunRSA(data, behav_df, freqband, cond, typeElecs, trange=[0,4.5], baseCorr=True):
    if freqband == 'alpha':
        freqs = np.arange(8, 13, 0.5)
    elif freqband == 'beta':
        freqs = np.arange(14, 25, 0.5)
    n_cycles = freqs / 2

    if cond == 'byPF':
        cond1_trials = behav_df[behav_df['stimPF'] == 1].index
        cond2_trials = behav_df[behav_df['stimPF'] == 0].index
        condDict = {'inPF': cond1_trials, 'outPF': cond2_trials}
    elif cond == 'byOri':
        cond1_trials = behav_df[behav_df['oriCategory'] == 30].index
        cond2_trials = behav_df[behav_df['oriCategory'] == 60].index
        cond3_trials = behav_df[behav_df['oriCategory'] == 120].index
        cond4_trials = behav_df[behav_df['oriCategory'] == 150].index
        condDict = {'30': cond1_trials, '60': cond2_trials, '120': cond3_trials, '150': cond4_trials}
    elif cond == 'byOriandPF':
        cond1_trials = behav_df[(behav_df['oriCategory'] == 30) & (behav_df['stimPF'] == 1)].index
        cond2_trials = behav_df[(behav_df['oriCategory'] == 60) & (behav_df['stimPF'] == 1)].index
        cond3_trials = behav_df[(behav_df['oriCategory'] == 120) & (behav_df['stimPF'] == 1)].index
        cond4_trials = behav_df[(behav_df['oriCategory'] == 150) & (behav_df['stimPF'] == 1)].index
        cond5_trials = behav_df[(behav_df['oriCategory'] == 30) & (behav_df['stimPF'] == 0)].index
        cond6_trials = behav_df[(behav_df['oriCategory'] == 60) & (behav_df['stimPF'] == 0)].index
        cond7_trials = behav_df[(behav_df['oriCategory'] == 120) & (behav_df['stimPF'] == 0)].index
        cond8_trials = behav_df[(behav_df['oriCategory'] == 150) & (behav_df['stimPF'] == 0)].index
        condDict = {'30 inPF': cond1_trials, '60 inPF': cond2_trials, '120 inPF': cond3_trials, '150 inPF': cond4_trials,
                    '30 outPF': cond5_trials, '60 outPF': cond6_trials, '120 outPF': cond7_trials, '150 outPF': cond8_trials}
    elif cond == 'byOri_broad':
        cond1_trials = behav_df[(behav_df['oriCategory'] == 30) | (behav_df['oriCategory'] == 60)].index
        cond2_trials = behav_df[(behav_df['oriCategory'] == 120) | (behav_df['oriCategory'] == 150)].index
        condDict = {'30-60': cond1_trials, '120-150': cond2_trials}

    if typeElecs == 'occ_hemi':
        leftElecs = ['O1', 'PO3', 'PO7', 'P1', 'P3', 'P7']
        rightElecs = ['O2', 'PO4', 'PO8', 'P2', 'P4', 'P8']
    elif typeElecs == 'par_hemi':
        leftElecs = ['CP1', 'CP3', 'CP5', 'P1', 'P3', 'P5']
        rightElecs = ['CP2', 'CP4', 'CP6', 'P2', 'P4', 'P6']
    elif typeElecs == 'occ':
        Elecs = ['O1', 'PO3', 'PO7', 'P1', 'P3', 'P7', 'O2', 'PO4', 'PO8', 'P2', 'P4', 'P8']
    elif typeElecs == 'par':
        Elecs = ['CP1', 'CP3', 'CP5', 'P1', 'P3', 'P5', 'CP2', 'CP4', 'CP6', 'P2', 'P4', 'P6']

    data = {}

    if 'hemi' in typeElecs:
        for condIdx in range(len(condDict)):
            leftBandPower = mne.time_frequency.tfr_morlet(data[condDict[condIdx]],
                                                    freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=False, picks=leftElecs, decim=3, n_jobs=7)
            rightBandPower = mne.time_frequency.tfr_morlet(data[condDict[condIdx]],
                                                    freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=False, picks=rightElecs, decim=3, n_jobs=7)
            dataLeft = leftBandPower.data.mean(axis=(1, 2))  # Mean across freqs and channels
            dataRight = rightBandPower.data.mean(axis=(1, 2))  # Mean across freqs and channels
            if baseCorr:
                base_tstart = leftBandPower.time_as_index(-1)[0]
                base_tend = leftBandPower.time_as_index(0)[0]
                dataLeft_baseline = dataLeft[:, base_tstart:base_tend].mean(axis=1)
                dataRight_baseline = dataRight[:, base_tstart:base_tend].mean(axis=1)
                dataLeft = 10 * np.log10(dataLeft[:, trange[0]:trange[1]] / dataLeft_baseline[:, np.newaxis])
                dataRight = 10 * np.log10(dataRight[:, trange[0]:trange[1]] / dataRight_baseline[:, np.newaxis])
            else:
                dataLeft = 10 * np.log10(dataLeft[:, trange[0]:trange[1]])
                dataRight = 10 * np.log10(dataRight[:, trange[0]:trange[1]])
            # data{}

