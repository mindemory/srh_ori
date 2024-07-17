import numpy as np
import pandas as pd
from scipy import stats
import time

def old_rotate_to_zero(df):
    TarTheta = np.arctan2(df['TarY'], df['TarX'])

    rotated_cols = ['TarX_rotated', 'TarY_rotated', 'isaccX_rotated', 'fsaccX_rotated', 'fsaccY_rotated']
    for col in rotated_cols:
        df[col] = 0.0
    
    for idx in df.index:
        rotation_matrix = np.array(
            [
                [np.cos(-TarTheta[idx]), -np.sin(-TarTheta[idx])],
                [np.sin(-TarTheta[idx]), np.cos(-TarTheta[idx])]
            ]
        )
    for col_prefix in ['Tar', 'isacc', 'fsacc']:
        original_points = np.array([df.at[idx, f'{col_prefix}X'], df.at[idx, f'{col_prefix}Y']])
        rotated_points = rotation_matrix.dot(original_points)
        df.at[idx, f'{col_prefix}X_rotated'] = rotated_points[0]
        df.at[idx, f'{col_prefix}Y_rotated'] = rotated_points[1]
    return df

def rotate_to_zero(tX, tY, saccX, saccY):
    tX = tX.values
    tY = tY.values
    saccX = saccX.values
    saccY = saccY.values

    tX_rot = np.zeros(len(tX),)
    tY_rot = np.zeros(len(tY),)
    saccX_rot = np.zeros(len(saccX),)
    saccY_rot = np.zeros(len(saccY),)

    TarTheta = np.arctan2(tY, tX)
    TarRadius = np.sqrt(tY**2+tX**2)
    scale_Radius = np.max(TarRadius)
    
    for idx in range(len(tX)):
        rotation_matrix = np.array(
            [
                [np.cos(-TarTheta[idx]), -np.sin(-TarTheta[idx])],
                [np.sin(-TarTheta[idx]), np.cos(-TarTheta[idx])]
            ]
        )
        scale_factor = (0 - TarRadius[idx]) #/ scale_Radius
        tX_scaled = tX[idx] + (scale_factor * np.cos(TarTheta[idx]))
        tY_scaled = tY[idx] + (scale_factor * np.sin(TarTheta[idx]))
        #original_t = np.array([tX[idx], tY[idx]])
        original_t = np.array([tX_scaled, tY_scaled])
        rotate_t = rotation_matrix.dot(original_t)
        tX_rot[idx] = rotate_t[0] 
        tY_rot[idx] = rotate_t[1]
        saccX_scaled = saccX[idx] + (scale_factor * np.cos(TarTheta[idx]))
        saccY_scaled = saccY[idx] + (scale_factor * np.sin(TarTheta[idx]))
        # original_sacc = np.array([saccX[idx], saccY[idx]])
        original_sacc = np.array([saccX_scaled, saccY_scaled])
        rotate_sacc = rotation_matrix.dot(original_sacc)
        saccX_rot[idx] = rotate_sacc[0]
        saccY_rot[idx] = rotate_sacc[1]
    return tX_rot, tY_rot, saccX_rot, saccY_rot

def compute_angular_range(angle_vals):
    # Created by Mrugank (06/03/2023) while on a 15 hour flight!
    # First step is making all angles positive
    #print(angle_vals[0])
    angle_vals = np.asarray(angle_vals.values)
    #print(angle_vals)
    angle_vals = angle_vals+np.pi 
    min_val = np.min(angle_vals)
    max_val = np.max(angle_vals)
    #print(min_val, max_val)
    if max_val - min_val < np.pi:
        # if the two are in the same hemicircle, then angular_width is simply the difference between the maximum and minimum
        angular_width = max_val - min_val
    else:
        idx_pi_plus = np.where(angle_vals>np.pi)
        angle_vals[idx_pi_plus] = -(2*np.pi - angle_vals[idx_pi_plus])
        min_rotated_val = np.min(angle_vals)
        max_rotated_val = np.max(angle_vals)
        angular_width = max_rotated_val - min_rotated_val
    return angular_width

def rotate_to_scale(df):
    metrics = ['Tar', 'isacc', 'fsacc']
    for mm in metrics:
        df[mm + 'Radius'] = np.sqrt(df[mm+'Y']**2 + df[mm+'X']**2)
        df[mm + 'Theta'] = np.arctan2(df[mm+'Y'], df[mm+'X'])
        # Initialize columns to store rotated anlges
        df[mm + 'X_rotated']=0
        df[mm + 'Y_rotated']=0
    
    tot_trs = 1000
    
    subjIDs = df['subjID'].unique()
    instim_idx = np.empty((len(subjIDs), tot_trs)) # num columns here is hard-coded, this is outright stupid!
    outstim_idx = np.empty((len(subjIDs), tot_trs))
    instim_mean_theta = np.zeros((len(subjIDs), ))
    outstim_mean_theta = np.zeros((len(subjIDs), ))
    instim_theta_range = np.zeros((len(subjIDs), ))
    outstim_theta_range = np.zeros((len(subjIDs, )))

    for ss in range(len(subjIDs)):
        this_subjID = subjIDs[ss]
        inidx = df.index[(df['instimVF']==1) & (df['subjID'] == this_subjID)]
        outidx = df.index[(df['instimVF']==0) & (df['subjID'] == this_subjID)]
        instim_idx[ss, 1:len(inidx)] = inidx
        outstim_idx[ss, 1:len(outidx)] = outidx 
        instim_mean_theta[ss] = stats.circmean(df.loc[inidx, ['TarTheta']], high=np.pi, low=-np.pi)
        outstim_mean_theta[ss] = stats.circmean(df.loc[outidx, ['TarTheta']], high=np.pi, low=-np.pi)
        instim_theta_range[ss] = compute_angular_range(df.loc[inidx, ['TarTheta']])
        outstim_theta_range[ss] = compute_angular_range(df.loc[outidx, ['TarTheta']])
    
    for ii in range(len(df['TarTheta'])):
        subj_idx = np.where(subjIDs == df.loc[ii, ['subjID'][0]])[0][0]
        # Get angle for target and corresponding rotation matrix
        if ii in instim_idx[subj_idx, :]:
            this_angle = -1 * instim_mean_theta[subj_idx]
        elif ii in outstim_idx[subj_idx, :]:
            this_angle = -1 * outstim_mean_theta[subj_idx]
        radial_error = np.max(df['TarRadius']) - df.loc[ii, ['TarRadius']][0]
        rotation_matrix = np.array([[np.cos(this_angle), -np.sin(this_angle)],
                                    [np.sin(this_angle), np.cos(this_angle)]])
        for mm in metrics:
            x = df.loc[ii, [mm+'X']][0] + radial_error * np.cos(df.loc[ii, [mm+'Theta']][0])
            y = df.loc[ii, [mm+'Y']][0] + radial_error * np.sin(df.loc[ii, [mm+'Theta']][0])
            
            input_points = np.vstack((x, y))
            rotated_points = np.dot(rotation_matrix, input_points)
            df.loc[ii, [mm + 'X_rotated']] = rotated_points[0]
            df.loc[ii, [mm + 'Y_rotated']] = rotated_points[1]
    for mm in metrics:
        df[mm + 'Radius_rotated'] = np.sqrt(df[mm + 'Y_rotated']**2 + df[mm + 'X_rotated']**2)
        df[mm + 'Theta_rotated'] = np.arctan2(df[mm + 'Y_rotated'], df[mm + 'X_rotated']) + np.pi
    
    angular_df = pd.DataFrame({'subjID': subjIDs, 'instim_mean_theta': instim_mean_theta, 'outstim_mean_theta': outstim_mean_theta, 'instim_theta_range': instim_theta_range,
                              'outstim_theta_range': outstim_theta_range})
    return df, angular_df


def calculate_mean_and_se(group, error_metric):
    mean = group[error_metric].mean()
    se = group[error_metric].std()
    return pd.Series({'mean': mean, 'se': se})

def compute_errors(df, sub_list, metric):
    ''' Computes mean and standard error of the mean for each subject in each block '''
    mean_errors_df = pd.DataFrame(index=sub_list)
    std_errors_df = pd.DataFrame(index=sub_list)
    for sub in sub_list:
        for day in range(1, 3):
            df_sub_day = df[(df['subjID'] == sub) & (df['day'] == day)]
            if day == 1:
                max_run = 15
            elif day == 2:
                max_run = 25
            for rnum in range(1, max_run+1):
                if rnum in df_sub_day['rnum'].unique():
                    mean_errors_df.at[sub, f"Day {day} Block {rnum}"] = df_sub_day[df_sub_day['rnum'] == rnum][metric].mean()
                    std_error = df_sub_day[df_sub_day['rnum'] == rnum][metric].std() / np.sqrt(df_sub_day[df_sub_day['rnum'] == rnum].shape[0])
                    std_errors_df.at[sub, f"Day {day} Block {rnum}"] = std_error
                else:
                    mean_errors_df.at[sub, f"Day {day} Block {rnum}"] = 0
                    std_errors_df.at[sub, f"Day {day} Block {rnum}"] = 0
    mean_errors_df.fillna(0, inplace=True)
    std_errors_df.fillna(0, inplace=True)
    return mean_errors_df, std_errors_df


def compute_tcount(df, sub_list):
    ''' Computes trial count and returns a dataframe with trial counts for each subject in each block '''
    mean_errors_df = pd.DataFrame(index=sub_list)
    for sub in sub_list:
        for day in range(1, 3):
            df_sub_day = df[(df['subjID'] == sub) & (df['day'] == day)]
            if day == 1:
                max_run = 15
            elif day == 2:
                max_run = 25
            for rnum in range(1, max_run+1):
                if rnum in df_sub_day['rnum'].unique():
                    mean_errors_df.at[sub, f"Day {day} Block {rnum}"] = df_sub_day[df_sub_day['rnum'] == rnum]['tnum'].count()
                else:
                    mean_errors_df.at[sub, f"Day {day} Block {rnum}"] = 0
    mean_errors_df.fillna(0, inplace=True)
    return mean_errors_df

def get_permutation_tstat(df_in, metric, cond1, cond2):
        data1 = df_in[df_in['condition'] == cond1].reset_index()
        data2 = df_in[df_in['condition'] == cond2].reset_index()
        
        result1 = data1.groupby(['subjID']).apply(calculate_mean_and_se, error_metric=metric)
        result2 = data2.groupby(['subjID']).apply(calculate_mean_and_se, error_metric=metric)

        #tstat, _ = stats.ttest_rel(result1['mean'], result2['mean'], nan_policy = 'omit', alternative='two-sided')
        tstat, _ = stats.ttest_rel(result1['se'], result2['se'], nan_policy = 'omit', alternative='two-sided')
        return tstat

def perform_permutation_test(df, pro_vs_anti, pairs_to_test, metric, iter_count, df_type):
    start = time.time()
    n_tests = len(pairs_to_test)
    tstat_permuted = np.zeros((iter_count, n_tests))
    tstat_real = np.zeros((n_tests, ))
    pval_2side = np.zeros((n_tests, ))
    pval_1side = np.zeros((n_tests, ))
    if pro_vs_anti == 'pro':
        df = df[df['ispro']==1]
    elif pro_vs_anti == 'anti':
        df = df[df['ispro']==0]
    df_master = df.copy()#[['subjID', 'day', 'ierr', 'ferr', 'isacc_rt', 'istms', 'instimVF']]

    if df_type == 'all5':
        cond_array = [
            (df_master['istms'] == 0) & (df_master['instimVF'] == 1) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 0) & (df_master['instimVF'] == 0) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 1) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 0) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 1) & (df_master['day'] == 4),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 0) & (df_master['day'] == 4),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 1) & (df_master['day'] == 5),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 0) & (df_master['day'] == 5)
        ]
        cond_names = [
            'notms inVF', 'notms outVF', 'mid inVF', 'mid outVF', 
            'early inVF', 'early outVF', 'mid dangit inVF', 'mid dangit outVF'
        ]
    elif df_type == 'first_three':
        cond_array = [
            (df_master['istms'] == 0) & (df_master['instimVF'] == 1) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 0) & (df_master['instimVF'] == 0) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 1) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 0) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 1) & (df_master['day'] == 4),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 0) & (df_master['day'] == 4),
        ]
        cond_names = [
            'notms inVF', 'notms outVF', 'mid inVF', 'mid outVF', 'early inVF', 'early outVF'
        ]
    elif df_type == 'clubbed':
        cond_array = [
            (df_master['istms'] == 0) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 1) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 0) & (df_master['day'].isin([1, 2, 3])),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 1) & (df_master['day'] == 4),
            (df_master['istms'] == 1) & (df_master['instimVF'] == 0) & (df_master['day'] == 4),
        ]
        cond_names = [
            'notms', 'mid inVF', 'mid outVF', 'early inVF', 'early outVF'
        ]

    df_master['condition'] = np.select(cond_array, cond_names, default='Other')

    for jj in range(n_tests):
        cond1 = pairs_to_test[jj][0]
        cond2 = pairs_to_test[jj][1]
        df_temp = df_master[(df_master['condition'] == cond1) | (df_master['condition'] == cond2)]
        tstat_real[jj] = get_permutation_tstat(df_temp, metric, cond1, cond2)
        for ii in range(iter_count):
            df_shuffle = df_temp.copy()
            df_shuffle['condition'] = np.random.permutation(df_shuffle['condition'])
            tstat_permuted[ii, jj] = get_permutation_tstat(df_shuffle, metric, cond1, cond2)
        # Compute pvalue of the real statistic given the tstat_permuted
        if tstat_real[jj] >= 0:
            samps_bothside = sum(tstat_permuted[:, jj] >= tstat_real[jj]) + sum(tstat_permuted[:, jj] < -tstat_real[jj])
        else:
            samps_bothside = sum(tstat_permuted[:, jj] >= -tstat_real[jj]) + sum(tstat_permuted[:, jj] < tstat_real[jj])
        samps_oneside = sum(tstat_permuted[:, jj] >= tstat_real[jj])
        pval_2side[jj] = (samps_bothside/iter_count)
        pval_1side[jj] = (samps_oneside/iter_count)

    print(f"Total time taken for running {iter_count} permutations: {round(time.time()-start, 3)} s")

    return tstat_real, tstat_permuted, pval_1side, pval_2side