import numpy as np
import pandas as pd
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde as gkde

from helpers import compute_errors, compute_tcount, calculate_mean_and_se

msize = 10
axes_fontsize = 14
title_fontsize = 18
#sns.set()

def distribution_plots(df):
    subjIDs = df['subjID'].unique()
    errs_to_plot = ['ierr', 'ferr', 'igain', 'fgain', 'ipea', 
                    'fpea', 'itheta', 'iamp', 'idir', 'isacc_peakvel']
    
    n_rows = 2
    n_cols = 5
    alpha_val = 0.8
    for ss in range(len(subjIDs)):
        subj_df =  df[df['subjID']==subjIDs[ss]]
        fig, axes = plt.subplots(n_rows, n_cols, figsize = (25, 10))
        plt.suptitle(subjIDs[ss])
        nbins = 50
        this_idx = 0
        for n1 in range(n_rows):
            for n2 in range(n_cols):
                this_err = errs_to_plot[this_idx]
                axes[n1, n2].hist(subj_df[subj_df['TMS_condition']=='No TMS'][this_err], alpha=alpha_val, label='No TMS', bins=nbins)
                axes[n1, n2].hist(subj_df[subj_df['TMS_condition']=='TMS intoVF'][this_err], alpha=alpha_val, label='TMS intoVF', bins=nbins)
                axes[n1, n2].hist(subj_df[subj_df['TMS_condition']=='TMS outVF'][this_err], alpha=alpha_val, label='TMS outVF', bins=nbins)
                axes[n1, n2].legend()
                axes[n1, n2].set_xlabel(this_err)
                this_idx += 1
        plt.show()

        fig2, axes = plt.subplots(n_rows, n_cols, figsize = (20, 10))
        plt.suptitle(subjIDs[ss])
        this_idx = 0
        for n1 in range(n_rows):
            for n2 in range(n_cols):
                this_err = errs_to_plot[this_idx]
                sns.violinplot(data = subj_df, x = 'TMS_condition', y=this_err, ax = axes[n1, n2], hue = 'ispro', split = 'inner')
                this_idx+=1
        plt.show()
        #sns.histplot(data=tmp_df, x=tmp_df['fsacc_err'], alpha = 0.5, label='No TMS')
        # sns.histplot(data=df, x=df[df['TMS_condition']=='TMS intoVF']['fsacc_err'], alpha = 0.5, label='TMS intoVF')
        # sns.histplot(data=df, x=df[df['TMS_condition']=='TMS outVF']['fsacc_err'], alpha = 0.5, label='TMS outVF')
        
def daywise_heatmap(df, sub_list, metric):
    '''This function creates a heatmap of the metric for each subject on each day and block.'''
    matrix_df = pd.DataFrame(index=sub_list)
    for sub in sub_list:
        for day in range(1, 3):
            df_sub_day = df[(df['subjID'] == sub) & (df['day'] == day)]
           
            for rnum in df_sub_day['rnum'].unique():
                if metric == 'trial_count':
                    metric_summary = df_sub_day[df_sub_day['rnum'] == rnum]['tnum'].count()
                else:
                    metric_summary = df_sub_day[df_sub_day['rnum'] == rnum][metric].mean()
                matrix_df.at[sub, f"Day {day} Block {rnum}"] = metric_summary
    matrix_df.fillna(0, inplace=True)

    day_dfs = {}
    for day in range(1, 3):
        day_columns = [col for col in matrix_df.columns if f"Day {day}" in col]
        day_dfs[day] = matrix_df[day_columns]

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for day, ax in zip(day_dfs.keys(), axs.flatten()):
        if metric == 'trial_count':
            sns.heatmap(day_dfs[day], cmap='viridis', cbar = False, annot=True, ax=ax)
        else:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            sns.heatmap(day_dfs[day], cmap='viridis', cbar = True, cbar_ax = cbar_ax, annot=True, ax=ax)
        ax.set_title(f'Day {day}')
        ax.set_xlabel('Block')
        ax.set_ylabel('Subject')
    plt.show()


def plot_metric(ax, x, y_data, color, label, y_error=None):
    '''
    This function plots data and error as shaded line around it.
    '''
    y_data = y_data.where(y_data != 0, np.nan)
    ax.plot(x, y_data, label=label, color=color)
    if y_error is not None:
        y_error = y_error.where(y_error != 0, np.nan)
        ax.fill_between(x, y_data - y_error, y_data + y_error, color=color, alpha=0.3)

def plot_block_splitters(ax, col_names, df, sub):
    for col_idx, col in enumerate(col_names):
        block_num = int(col.split(' ')[-1])
        day_num = int(col.split(' ')[1])

        block_df = df[(df['day'] == day_num) & (df['rnum'] == block_num) & (df['subjID'] == sub)]
        
        # if block_df.shape[0] == 0:
        color = 'lightblue'
        
        ax.axvspan(col_idx - 0.5, col_idx + 0.5, facecolor=color, alpha=0.3)

        if col.endswith("Block 1"):
            # if block_df['istms'].eq(0).any():
            #     ax.axvline(x=col_idx-0.5, color='orange', linewidth=2)
            # else:
            ax.axvline(x=col_idx-0.5, color='black', linewidth=2)
        else:
            ax.axvline(x=col_idx-0.5, color='grey', linestyle='--')


def daywise_trend(df, sub_list, metric):

    if metric == 'trial_count':
        mean_errors = compute_tcount(df, sub_list)
    else:
        mean_errors, std_errors = compute_errors(df, sub_list, metric)
    col_names = mean_errors.columns
    
    fig, axs = plt.subplots(len(sub_list), 1, figsize=(15, 5 * len(sub_list)))
    for idx, sub in enumerate(sub_list):
        ax = axs[idx]
        x = range(len(col_names))

        y = mean_errors.loc[sub]
        if metric == 'trial_count':
            plot_metric(ax, x, y, 'black', 'Calib Count')
        else:
            yerr_calib = std_errors.loc[sub]
            plot_metric(ax, x, y, 'black', 'Calib Count', yerr_calib)

        plot_block_splitters(ax, col_names, df, sub)

        ax.set_title(f'Subject {sub}')
        ax.set_xlabel('Day and Block')
        ax.set_ylabel(metric)
        if metric == 'trial_count':
            ax.set_ylim([0, 42])
        elif metric == 'ierr' or metric == 'ferr':
            ax.set_ylim([0, 4])
        elif metric == 'isacc_rt' or metric == 'fsacc_rt':
            ax.set_ylim([0, 1])
        ax.set_xticks(x)
        ax.set_xticklabels(col_names, rotation=90)
        ax.legend()

    plt.tight_layout()
    plt.show()


def daywise_trend_dual_metric(df_calib, df_calib_all5, df_nocalib, df_nocalib_all5, sub_list, metrics):
    metric1, metric2 = metrics
    mean_calib1, std_calib1 = compute_errors(df_calib, df_calib_all5, sub_list, metric1)
    mean_nocalib1, std_nocalib1 = compute_errors(df_nocalib, df_nocalib_all5, sub_list, metric1)
    mean_calib2, std_calib2 = compute_errors(df_calib, df_calib_all5, sub_list, metric2)
    mean_nocalib2, std_nocalib2 = compute_errors(df_nocalib, df_nocalib_all5, sub_list, metric2)
    col_names = mean_calib1.columns

    fig, axs = plt.subplots(len(sub_list), 1, figsize=(15, 5 * len(sub_list)))
    for idx, sub in enumerate(sub_list):
        ax1 = axs[idx]
        ax2 = ax1.twinx()

        x = range(len(col_names))
        y_calib1 = mean_calib1.loc[sub]
        yerr_calib1 = std_calib1.loc[sub]
        y_nocalib1 = mean_nocalib1.loc[sub]
        yerr_nocalib1 = std_nocalib1.loc[sub]
        plot_metric(ax1, x, y_calib1,'black', f'{metric1} Calib', yerr_calib1)
        plot_metric(ax1, x, y_nocalib1,'blue', f'{metric1} Nocalib', yerr_nocalib1)
        
        y_calib2 = mean_calib2.loc[sub]
        yerr_calib2 = std_calib2.loc[sub]
        y_nocalib2 = mean_nocalib2.loc[sub]
        yerr_nocalib2 = std_nocalib2.loc[sub]
        plot_metric(ax2, x, y_calib2,'black', f'{metric2} Calib', yerr_calib2)
        plot_metric(ax2, x, y_nocalib2,'blue', f'{metric2} Nocalib', yerr_nocalib2)
        
        ax1.set_title(f'Subject {sub}')
        ax1.set_xlabel('Day and Block')
        ax1.set_ylabel(metric1)
        ax2.set_ylabel(metric2)

        ax1.set_xticks(x)
        ax1.set_xticklabels(col_names, rotation=90)

    plt.tight_layout()
    plt.show()


def plot_error_metric(df, df_all5, sublist, sublist_all5, metric):
    
    conds_all5 = {
        'No TMS': df_all5[(df_all5['TMS_condition'] == 'No TMS') & (df_all5['day'].isin([1, 2, 3]))],
        'mid inVF': df_all5[(df_all5['TMS_condition'] == 'TMS intoVF') & (df_all5['day'].isin([1, 2, 3]))],
        'mid outVF': df_all5[(df_all5['TMS_condition'] == 'TMS outVF') & (df_all5['day'].isin([1, 2, 3]))],
        'early inVF': df_all5[(df_all5['TMS_condition'] == 'TMS intoVF') & (df_all5['day'] == 4)],
        'early outVF': df_all5[(df_all5['TMS_condition'] == 'TMS outVF') & (df_all5['day'] == 4)],
        'mid dangit inVF': df_all5[(df_all5['TMS_condition'] == 'TMS intoVF') & (df_all5['day'] == 5)],
        'mid dangit outVF': df_all5[(df_all5['TMS_condition'] == 'TMS outVF') & (df_all5['day'] == 5)],
    }

    conds_all5_analysis = {
        'No TMS inVF': df_all5[(df_all5['TMS_condition'] == 'No TMS') & (df_all5['instimVF'] == 1) & (df_all5['day'].isin([1, 2, 3]))],
        'No TMS outVF': df_all5[(df_all5['TMS_condition'] == 'No TMS') & (df_all5['instimVF'] == 0) & (df_all5['day'].isin([1, 2, 3]))],
        'mid inVF': df_all5[(df_all5['TMS_condition'] == 'TMS intoVF') & (df_all5['day'].isin([1, 2, 3]))],
        'mid outVF': df_all5[(df_all5['TMS_condition'] == 'TMS outVF') & (df_all5['day'].isin([1, 2, 3]))],
        'early inVF': df_all5[(df_all5['TMS_condition'] == 'TMS intoVF') & (df_all5['day'] == 4)],
        'early outVF': df_all5[(df_all5['TMS_condition'] == 'TMS outVF') & (df_all5['day'] == 4)],
        'mid dangit inVF': df_all5[(df_all5['TMS_condition'] == 'TMS intoVF') & (df_all5['day'] == 5)],
        'mid dangit outVF': df_all5[(df_all5['TMS_condition'] == 'TMS outVF') & (df_all5['day'] == 5)],
    }

    conds = {
        'No TMS pro': df[(df['TMS_condition'] == 'No TMS') & (df['ispro'] == 1) & (df['day'].isin([1, 2, 3]))],
        'No TMS anti': df[(df['TMS_condition'] == 'No TMS') & (df['ispro'] == 0) & (df['day'].isin([1, 2, 3]))],
        'TMS inVF pro': df[(df['TMS_condition'] == 'TMS intoVF') & (df['ispro'] == 1) & (df['day'].isin([1, 2, 3]))],
        'TMS inVF anti': df[(df['TMS_condition'] == 'TMS intoVF') & (df['ispro'] == 0) & (df['day'].isin([1, 2, 3]))],
        'TMS outVF pro': df[(df['TMS_condition'] == 'TMS outVF') & (df['ispro'] == 1) & (df['day'].isin([1, 2, 3]))],
        'TMS outVF anti': df[(df['TMS_condition'] == 'TMS outVF') & (df['ispro'] == 0) & (df['day'].isin([1, 2, 3]))]
    }
    if metric == 'ierr':
        y_range = [0, 3]
    elif metric == 'ferr':
        y_range = [0, 2]
    elif metric == 'isacc_rt':
        y_range = [0, 0.6]

    results_all5 = {cond: data.groupby('subjID').apply(calculate_mean_and_se, error_metric=metric) for cond, data in conds_all5.items()}
    combined_all5 = pd.concat(results_all5, names=['Condition']).reset_index()
    combined_all5['time'] = combined_all5['Condition'].apply(lambda x: 'notms' if 'No TMS' in x else ('early' if 'early' in x else ('mid dangit' if 'mid dangit' in x else 'mid')))
    combined_all5['VF'] = combined_all5['Condition'].apply(lambda x: 1 if 'inVF' in x else 0)
    
    results_all5_analysis = {cond: data.groupby('subjID').apply(calculate_mean_and_se, error_metric=metric) for cond, data in conds_all5_analysis.items()}
    combined_all5_analysis = pd.concat(results_all5_analysis, names=['Condition']).reset_index()
    combined_all5_analysis['time'] = combined_all5_analysis['Condition'].apply(lambda x: 'notms' if 'No TMS' in x else ('early' if 'early' in x else ('mid dangit' if 'mid dangit' in x else 'mid')))
    combined_all5_analysis['VF'] = combined_all5_analysis['Condition'].apply(lambda x: 1 if 'inVF' in x else 0)
    
    results = {cond: data.groupby('subjID').apply(calculate_mean_and_se, error_metric=metric) for cond, data in conds.items()}
    combined = pd.concat(results, names=['Condition']).reset_index()
    combined['Type'] = combined['Condition'].apply(lambda x: 'pro' if 'pro' in x else 'anti')
    combined['TMS'] = combined['Condition'].apply(lambda x: 'NoTMS' if 'No TMS' in x else ('TMS inVF' if 'inVF' in x else 'TMS outVF'))
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    sns.pointplot(data=combined_all5, x='Condition', y='mean', linestyle="none", capsize=.2, errorbar="se", ax = axs[0])
    sns.stripplot(data=combined_all5, x='Condition', y='mean', color="black", jitter=False, size=6, ax = axs[0])
    sns.lineplot(data=combined_all5, x='Condition', y='mean', hue='subjID', 
                    palette=['gray']*len(sublist_all5), legend=False, dashes=True, linewidth=1, ax = axs[0])
    axs[0].set_title(f'Mean {metric} with Standard Error (N = {len(sublist_all5)} subjects)')
    axs[0].set_ylabel(f'Mean {metric}')
    axs[0].set_xlabel('Condition')
    axs[0].set_ylim(y_range)

    sns.pointplot(data=combined, x = 'Condition', y = 'mean', linestyle = 'none', hue = 'Type', capsize=.2, errorbar="se", ax = axs[1])
    sns.stripplot(data=combined, x='Condition', y='mean', palette="dark:black", hue = 'Type', jitter=False, size=6, ax = axs[1])
    axs[1].set_title(f'Mean {metric} for Pro and Anti Conditions (N = {len(sublist)} subjects)')
    axs[1].set_ylabel(f'Mean {metric}')
    axs[1].set_xlabel('Condition')
    axs[1].set_ylim(y_range)
    
    plt.tight_layout()
    plt.show()

    return combined, combined_all5, combined_all5_analysis


def plot_permutation_result(tstat_permuted, tstat_real, pval_2side, pval_1side, ii, pairs_to_test, ax):
    title_text = pairs_to_test[ii][2]
    sns.histplot(tstat_permuted[:, ii], element='step', fill=False, ax=ax)
    ax.axvline(x=tstat_real[ii], color='k', linestyle='--')
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.text(x1 + (x2-x1) * 0.01, y2 - (y2-y1) * 0.15, 
            f't-stat = {tstat_real[ii]:.3f}\npval (both) = {pval_1side[ii]:.3f}', 
            fontsize=9, color='black')
    # ax.text(x1 + (x2-x1) * 0.01, y2 - (y2-y1) * 0.15, 
    #         f't-stat = {tstat_real[ii]:.3f}\npval (both) = {pval_2side[ii]:.3f}\npval (greater) = {pval_1side[ii]:.3f}', 
    #         fontsize=9, color='black')
    ax.set_title(title_text)