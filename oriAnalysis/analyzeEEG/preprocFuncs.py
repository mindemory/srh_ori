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