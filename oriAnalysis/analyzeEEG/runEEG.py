import os, sys, mne, socket, time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nilearn import plotting

from initialization import load_paths
from preprocFuncs import getRawData, getTrials, getPower
from plotters import plotPower, plotRSA

def main():
    subjID = 99
    day = 3
    p = load_paths(subjID, day)
    # Load behavioral data
    master_df = pd.read_csv(os.path.join(p['meta'], 'master_df.csv'))
    behav_df = master_df[(master_df['subjID'] == subjID) & (master_df['day'] == 1)]
    relevCols = [col for col in behav_df.columns if 'oriArray_' not in col]
    behav_df = behav_df[relevCols]
    # Load raw data
    raw_data = getRawData(p)

    # Define events
    events, tah = mne.events_from_annotations(raw_data)
    events_dict = {
        'BlockOnset': 1001,
        'BlockEnd': 7,
        'Fixation': 1,
        'Delay1': 2,
        'Delay2': 3,
        'Response': 4,
        'Feedback': 5,
        'ITI': 6,
    }
    trl_events = {
        'inPF': 11,
        'outPF': 10,
    }

    # Define electrodes of interest
    left_occ_elecs = ['O1', 'PO3', 'PO7', 'P1', 'P3', 'P7']
    right_occ_elecs = ['O2', 'PO4', 'PO8', 'P2', 'P4', 'P8']

    trials_list = getTrials(raw_data, events_dict, events)

    window_size = 0.5
    overlap = 0.1
    freq_band = 'alpha'
    bandPower, twindow_power = getPower(trials_list, window_size, overlap, freq_band)
    # bandPower: (n_trials, n_channels, n_timepoints)

    leftOccElecsIdx = [trials_list[0].ch_names.index(ch) for ch in left_occ_elecs]
    rightOccElecsIdx = [trials_list[0].ch_names.index(ch) for ch in right_occ_elecs]
    # leftOccBandPower = bandPower[:, leftOccElecsIdx, :]
    # rightOccBandPower = bandPower[:, rightOccElecsIdx, :]

    # # leftPower = left_occ_alpha_power.copy()
    # # rightPower = right_occ_alpha_power.copy()

    # # left_occ_alpha_power = (leftPower-rightPower) / (leftPower+rightPower+1e-10)
    # # right_occ_alpha_power = (rightPower-leftPower) / (rightPower+leftPower+1e-10)

    # # # left_occ_alpha_power = (left_occ_alpha_power - right_occ_alpha_power) / (left_occ_alpha_power + right_occ_alpha_power + 1e-10)
    # # # right_occ_alpha_power = (right_occ_alpha_power - left_occ_alpha_power) / (right_occ_alpha_power + left_occ_alpha_power + 1e-10)
    # baseline_time = np.where((twindow_power[0] > 1) & (twindow_power[0] < 2))
    # baseline_average = (np.mean(leftOccBandPower[:, :, baseline_time]) + np.mean(rightOccBandPower[:, :, baseline_time])) / 2

    # # left_occ_alpha_power /= baseline_average
    # # right_occ_alpha_power /= baseline_average

    # leftOccBandPower = (leftOccBandPower - baseline_average) / baseline_average
    # rightOccBandPower = (rightOccBandPower - baseline_average) / baseline_average
    # # # Get relevant trials
    
    # # conds = 'in-out'
    # conds_power = 'byOriandPF' #'byOri'
    # plotPower(leftOccBandPower, rightOccBandPower, twindow_power, behav_df, freq_band, conds_power)

    conds_RSA = 'byPF'
    plotRSA(bandPower, twindow_power, behav_df, conds_RSA, leftOccElecsIdx, rightOccElecsIdx)

if __name__ == '__main__':
    main()
# if '__init__' == '__main__':
#     main()