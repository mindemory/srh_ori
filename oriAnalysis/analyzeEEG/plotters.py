import matplotlib.pyplot as plt
import numpy as np

def addFlags(ax, twindow):
    fixTime = twindow[(twindow> 1)][0]
    sampTime = twindow[(twindow> 2)][0]
    delayTime = twindow[(twindow> 2.25)][0]
    respTime = twindow[(twindow> 6.25)][0]
    ax.axvline(x=fixTime, color='k', linestyle='--')
    ax.axvline(x=sampTime, color='k', linestyle='--')
    ax.axvline(x=delayTime, color='k', linestyle='--')
    ax.axvline(x=respTime, color='k', linestyle='--')

def plotPower(leftBandPower, rightBandPower, twindow_power, behav_df, band, conds):

    if conds == 'byHemi':
        trlIdx_outPF = behav_df[behav_df['stimPF'] == 0].index 
        trlIdx_inPF = behav_df[behav_df['stimPF'] == 1].index
        trlIdx = [trlIdx_inPF, trlIdx_outPF]
        nrows, ncols = 2, 1
        nsubplots = nrows * ncols
        condLabels = ['inPF', 'outPF']
    elif conds == 'byOri':
        trlIdx_30 = behav_df[behav_df['targOriCategory'] == 30].index
        trlIdx_60 = behav_df[behav_df['targOriCategory'] == 60].index
        trlIdx_120 = behav_df[behav_df['targOriCategory'] == 120].index
        trlIdx_150 = behav_df[behav_df['targOriCategory'] == 150].index
        trlIdx = [trlIdx_30, trlIdx_60, trlIdx_120, trlIdx_150]
        nrows, ncols = 2, 2
        nsubplots = nrows * ncols
        condLabels = ['30', '60', '120', '150']
    elif conds == 'byOriandPF':
        trlIdx_30_inPF = behav_df[(behav_df['targOriCategory'] == 30) & (behav_df['stimPF'] == 1)].index
        trlIdx_60_inPF = behav_df[(behav_df['targOriCategory'] == 60) & (behav_df['stimPF'] == 1)].index
        trlIdx_120_inPF = behav_df[(behav_df['targOriCategory'] == 120) & (behav_df['stimPF'] == 1)].index
        trlIdx_150_inPF = behav_df[(behav_df['targOriCategory'] == 150) & (behav_df['stimPF'] == 1)].index
        trlIdx_30_outPF = behav_df[(behav_df['targOriCategory'] == 30) & (behav_df['stimPF'] == 0)].index
        trlIdx_60_outPF = behav_df[(behav_df['targOriCategory'] == 60) & (behav_df['stimPF'] == 0)].index
        trlIdx_120_outPF = behav_df[(behav_df['targOriCategory'] == 120) & (behav_df['stimPF'] == 0)].index
        trlIdx_150_outPF = behav_df[(behav_df['targOriCategory'] == 150) & (behav_df['stimPF'] == 0)].index
        trlIdx = [trlIdx_30_inPF, trlIdx_60_inPF, trlIdx_120_inPF, trlIdx_150_inPF, trlIdx_30_outPF, trlIdx_60_outPF, trlIdx_120_outPF, trlIdx_150_outPF]
        nrows, ncols = 4, 2
        nsubplots = nrows * ncols
        condLabels = ['30 inPF', '60 inPF', '120 inPF', '150 inPF', '30 outPF', '60 outPF', '120 outPF', '150 outPF']
    else:
        print(f"Invalid condition. Expected 'byHemi', 'byOri', 'byOriandPF'. Got {conds}")

    f, axs = plt.subplots(nrows, ncols, figsize=(8, 6))
    axs = axs.flatten()
    for i in range(nsubplots):
        meanLeft = np.nanmean(leftBandPower[trlIdx[i], 0, :], axis=0)
        meanRight = np.nanmean(rightBandPower[trlIdx[i], 0, :], axis=0)
        semLeft = np.nanstd(leftBandPower[trlIdx[i], 0, :], axis=0) / np.sqrt(len(trlIdx[i]))
        semRight = np.nanstd(rightBandPower[trlIdx[i], 0, :], axis=0) / np.sqrt(len(trlIdx[i]))
        axs[i].plot(twindow_power[0], meanLeft, 'r-', label='Left '+condLabels[i])
        axs[i].fill_between(twindow_power[0], meanLeft - semLeft, meanLeft + semLeft, color='r', alpha=0.5)
        axs[i].plot(twindow_power[0], meanRight, 'b-', label='Right '+condLabels[i])
        axs[i].fill_between(twindow_power[0], meanRight - semRight, meanRight + semRight, color='b', alpha=0.5)
        axs[i].set_ylim([-1, 1])
        axs[i].axhline(y=0, color='k', linestyle='--')
        addFlags(axs[i], twindow_power[0])
        axs[i].set_title('Occipital ' + band + ' power')
        axs[i].legend()
    plt.show()


def plotRSAOld(bandPower, twindow_power, behav_df, conds_RSA, leftOccElecsIdx, rightOccElecsIdx):
    correct_baseline = 1
    if correct_baseline:
        baseline_time = np.where((twindow_power[0] >= 1) & (twindow_power[0] < 2))[0]
        baseline_ = bandPower[:, :, baseline_time]
        baseline_average = np.mean(baseline_, axis=(0, 2))
        baseline_average = np.expand_dims(baseline_average, axis=(0,2))
        # baseline_average = np.expand_dims(baseline_average, axis=2)
        bandPower = (bandPower - baseline_average) / baseline_average
        start_time = 2
    else:
        start_time = 1
        # twindow_power = 
    # baseline_average = (np.mean(leftOccBandPower[:, :, baseline_time]) + np.mean(rightOccBandPower[:, :, baseline_time])) / 2

    # b

    if conds_RSA == 'byPF':
        trlIdx_outPF = behav_df[behav_df['stimPF'] == 0].index 
        trlIdx_inPF = behav_df[behav_df['stimPF'] == 1].index
        bandPower_inPF = bandPower[trlIdx_inPF, :, :]
        bandPower_outPF = bandPower[trlIdx_outPF, :, :]
        bandPower = np.concatenate((bandPower_inPF, bandPower_outPF), axis=0)
        len_inPF = len(trlIdx_inPF)

    allElecsIDx = leftOccElecsIdx + rightOccElecsIdx
    tslice = 1000
    qt1 = 1
    qt2 = 0
    qt_thresh = 0.1
    for idx, start in enumerate(range(start_time, len(twindow_power[0]) - tslice + 1, tslice)):
        this_bandPower = np.mean(bandPower[:, allElecsIDx, start:start+tslice], axis=2)
        this_RSA = this_bandPower @ this_bandPower.T
        quantiles_this_RSA = np.quantile(this_RSA.flatten(), [qt_thresh, 1-qt_thresh])
        qt1 = min(qt1, quantiles_this_RSA[0])
        qt2 = max(qt2, quantiles_this_RSA[1])
    
    numPlots = len(twindow_power[0])//tslice
    ncols = 4
    nrows = int(numPlots//ncols+1)
    f, axs = plt.subplots(nrows, ncols, figsize=(12, 8))
    axs = axs.flatten()
    for idx, start in enumerate(range(start_time, len(twindow_power[0]) - tslice + 1, tslice)):
        this_bandPower = np.mean(bandPower[:, allElecsIDx, start:start+tslice], axis=2)
        this_RSA = this_bandPower @ this_bandPower.T
        # quantiles_this_RSA = np.quantile(this_RSA.flatten(), [qt_thresh, 1-qt_thresh])
        # axs[idx].hist(this_RSA.flatten(), bins=100)
        # this_RSA = np.triu(this_RSA, k=1)
        # this_RSA = this_RSA[this_RSA != 0]
        axs[idx].imshow(this_RSA, cmap='RdBu_r', vmin=qt1, vmax=qt2)
        # axs[idx].imshow(this_RSA, cmap='RdBu_r', vmin=quantiles_this_RSA[0], vmax=quantiles_this_RSA[1])
        axs[idx].axhline(y=len_inPF, color='k', linestyle='--')
        axs[idx].axvline(x=len_inPF, color='k', linestyle='--')
        axs[idx].set_title(f'RSA: {twindow_power[0][start]}-{twindow_power[0][start+tslice]}')
        # axs[idx].set_colorbar()
        # axs[idx].colorbar()

    plt.show()
        # for i in range(2):
        #     meanLeft = np.nanmean(bandPower[i][:, leftOccElecsIdx, start:start+tslice], axis=0)
        #     meanRight = np.nanmean(bandPower[i][:, rightOccElecsIdx, start:start+tslice], axis=0)
        #     meanDiff = meanLeft - meanRight
        #     semDiff = np.nanstd(meanDiff, axis=0) / np.sqrt(len(meanDiff))
        #     axs[start//tslice].plot(twindow_power[0][start:start+tslice], meanDiff, label='PF '+str(i))
        #     axs[start//tslice].fill_between(twindow_power[0][start:start+tslice], meanDiff - semDiff, meanDiff + semDiff, alpha=0.5)
        #     axs[start//tslice].set_ylim([-1, 1])
        #     axs[start//tslice].axhline(y=0, color='k', linestyle='--')
        #     addFlags(axs[start//tslice], twindow_power[0][start:start+tslice])
        #     axs[start//tslice].set_title('Occipital RSA')
        #     axs[start//tslice].legend()

def plotRSA(data, tarray, dim=0):
    print(tarray[0], tarray[-1])
    plt.figure(figsize=(8, 8))
    if dim == 0:
        RSA_data = np.corrcoef(data)
    else:
        RSA_data = np.corrcoef(data.T)
    qt_thresh = 0.2
    low_qt = np.quantile(RSA_data[RSA_data != 0], qt_thresh)
    high_qt = np.quantile(RSA_data[RSA_data != 0], 1-qt_thresh)
    print(low_qt, high_qt)
    plt.imshow(RSA_data, cmap='RdBu_r', vmin=low_qt, vmax=high_qt)
    if dim == 1:
        tick_vals = np.arange(int(tarray[0]), int(tarray[-1]+1))
        tick_idx = [int(np.where(tarray == val)[0]) for val in tick_vals]
        tick_labels = [str(val) for val in tick_vals]
        plt.xticks(tick_idx, tick_labels)
        plt.yticks(tick_idx, tick_labels)

    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()

def plotDiffRSA(data1, data2, dim=0):
    if dim == 0:
        RSA_inPF = np.corrcoef(data1)
        RSA_outPF = np.corrcoef(data2)
    else:
        RSA_inPF = np.corrcoef(data1.T)
        RSA_outPF = np.corrcoef(data2.T)
    RSA_diff = RSA_inPF - RSA_outPF
    qt_thresh = 0.15
    low_qt = np.quantile(RSA_diff, qt_thresh)
    high_qt = np.quantile(RSA_diff, 1-qt_thresh)
    print(low_qt, high_qt)
    plt.figure(figsize=(8, 8))
    plt.imshow(RSA_diff, cmap='RdBu_r', vmin=low_qt, vmax=high_qt)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()