# Load the modules
from scipy.io import loadmat
import numpy as np
import pandas as pd
import os

# Hide deprecation warnings!
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import socket
hostname = socket.gethostname()

p = {}
if hostname == 'syndrome' or hostname == 'zod.psych.nyu.edu' or hostname == 'zod':
    p['datc'] =  '/d/DATD/datd/TMS_V3AB'
else:
    p['datc'] =  '/Users/mrugankdake/Documents/Clayspace/EEG_TMS/datd/TMS_V3AB'
p['data'] = p['datc'] + '/data'
p['analysis'] = p['datc'] + '/analysis'
p['meta'] = p['datc'] + '/analysis/meta_analysis'

if not os.path.exists(p['meta']):
    os.makedirs(p['meta'])

summary_df = pd.read_csv(os.path.join(p['analysis'] + '/V3AB_TMS_meta - Summary.csv'))
All_metadata = {row['Subject ID']: row for _, row in summary_df.iterrows()}

mdf_fname = 'master_df.csv'
df_name = 'df_fname'
p[df_name] = os.path.join(p['meta'], mdf_fname)

if os.path.exists(p[df_name]):
    print('Loading existing dataframe! If this is not desired, delete the current mater_df.csv')
    master_df = pd.read_csv(p[df_name])
else:
    print('Creating a new dataframe.')
    # Find subjects and days that have been run so far
    sub_dirs = [d for d in os.listdir(os.path.join(p['analysis'])) if d.startswith("sub")]
    # Added temporarily while data collection is in progress
    sub_dirs = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub09', 'sub10', 'sub11', 'sub12', 'sub13', 
                'sub14', 'sub16', 'sub17', 'sub18', 'sub19', 'sub21', 'sub22', 'sub23'] 
    subjIDs = sorted([int(ss[-2:]) for ss in sub_dirs])
    print(f"We have {len(subjIDs)} subjects so far: {subjIDs}")
    print()

    session_dfs = []
    # Create a master dataframe
    for ss in subjIDs:
        # Extract meta-data for this subject
        metadata = All_metadata[ss]
        subdir_name = f'sub{ss:02}'
        subject_row = summary_df[summary_df['Subject ID'] == ss].iloc[0]

        subjdir = os.path.join(p['analysis'], subdir_name) # Path of current subject directory
        phosphene_data_path = os.path.join(p['data'], 'phosphene_data',  subdir_name)

        # Load timestructure for this subject
        timeStructFname = os.path.join(subjdir, 'timeStruct.mat')
        timeStruct = loadmat(timeStructFname)['timeStruct']

        # Get list of all days this subject was run
        day_dirs = sorted(d for d in os.listdir(subjdir) if d.startswith("day"))
        days = []
        for dd in day_dirs:
            th_day = int(dd[-2:])
            print(f'Running subj = {ss}, day = {dd[-2:]}')
            daydir = os.path.join(subjdir, dd)

            sessfName = os.path.join(daydir, f'ii_sess_{subdir_name}_{dd}.mat')
            ii_sess = loadmat(sessfName)['ii_sess']
            tcount = len(ii_sess['t_num'][0, 0].T[0])

            # Extract times for this day
            idur = timeStruct['initDuration'][0, 0][th_day-1, :, :]
            sdur = timeStruct['sampleDuration'][0, 0][th_day-1, :, :]
            d1dur = timeStruct['delay1Duration'][0, 0][th_day-1, :, :]
            d2dur = timeStruct['delay2Duration'][0, 0][th_day-1, :, :]
            rdur = timeStruct['respDuration'][0, 0][th_day-1, :, :]
            fdur = timeStruct['feedbackDuration'][0, 0][th_day-1, :, :]
            itidur = timeStruct['itiDuration'][0, 0][th_day-1, :, :]
            trdur = timeStruct['trialDuration'][0, 0][th_day-1, :, :]

            sess_data = {
                'subjID': [ss] * tcount,
                'day': [th_day] * tcount,
                # Add the meta-data
                'gender': [metadata['Sex']] * tcount,
                'race': [metadata['Race']] * tcount,
                'handedness': [metadata['Handedness']] * tcount,
                'hemistimulated': [metadata['Hemisphere stimulated']] * tcount,
                'age': [metadata['Age']] * tcount,
                'weight': [metadata['Weight (kg)']] * tcount,
                'PT': [metadata['PT']] * tcount,
                'StimIntensity': [metadata['Stim Intensity']] * tcount,
                'Design': [metadata['Design Chosen']] * tcount,
                'NoTMSBlocks': [metadata['NoTMS Blocks']] * tcount,
                'TMSBlocks': [metadata['TMS Blocks']] * tcount,
                # Trial data
                'rnum': ii_sess['r_num'][0, 0].T[0],
                'tnum': ii_sess['t_num'][0, 0].T[0],
                'istms': ii_sess['istms'][0, 0].T[0], # 1 if TMS is on
                'instimVF': ii_sess['instimVF'][0, 0].T[0], # 1 if saccade is in VF
                # Trial flags
                'bad_drift_correct': ii_sess['bad_drift_correct'][0, 0].T[0], # Flag 11
                'bad_calibration': ii_sess['bad_calibration'][0, 0].T[0], # Flag 12
                'breakfix': ii_sess['break_fix'][0, 0].T[0], # Flag 13
                'no_prim_sacc': ii_sess['no_prim_sacc'][0, 0].T[0], # Flag 20
                'small_sacc': ii_sess['small_sacc'][0, 0].T[0], # Flag 21
                'large_error': ii_sess['large_error'][0, 0].T[0], # Flag 22
                'rejtrials': ii_sess['rejtrials'][0, 0].T[0], # Flags 20 and 22, can be updated here directly
                # Target and eye end-points
                'TarX': ii_sess['targ'][0, 0][:, 0].T,
                'TarY': ii_sess['targ'][0, 0][:, 1].T,
                'isaccX': ii_sess['i_sacc_raw'][0, 0][:, 0].T,
                'isaccY': ii_sess['i_sacc_raw'][0, 0][:, 1].T,
                'fsaccX': ii_sess['f_sacc_raw'][0, 0][:, 0].T,
                'fsaccY': ii_sess['f_sacc_raw'][0, 0][:, 1].T,
                # errors added
                'isacc_err': ii_sess['i_sacc_err'][0, 0].T[0], # raw euclidean distance error wrt target for isacc
                'fsacc_err': ii_sess['f_sacc_err'][0, 0].T[0], # raw euclidean distance error wrt target for fsacc
                'nsacc': ii_sess['n_sacc'][0, 0].T[0],
                # Reaction times added
                'isacc_rt': ii_sess['i_sacc_rt'][0, 0].T[0], # RT for isacc wrt response epoch start
                'fsacc_rt': ii_sess['f_sacc_rt'][0, 0].T[0], # RT for fsacc wrt response epoch start
                # Velocities
                'isacc_peakvel': ii_sess['i_sacc_peakvel'][0, 0].T[0],
                'fsacc_peakvel': ii_sess['f_sacc_peakvel'][0, 0].T[0],
                # Add timereports for each epoch
                'initdur': idur[~np.isnan(idur)].flatten(),
                'sampledur': sdur[~np.isnan(sdur)].flatten(),
                'delay1dur': d1dur[~np.isnan(d1dur)].flatten(),
                'delay2dur': d2dur[~np.isnan(d2dur)].flatten(),
                'respdur': rdur[~np.isnan(rdur)].flatten(),
                'feedbackdur': fdur[~np.isnan(fdur)].flatten(),
                'itidur': itidur[~np.isnan(itidur)].flatten(),
                'trialdur': trdur[~np.isnan(trdur)].flatten(),
            }
            session_dfs.append(pd.DataFrame(sess_data))
            this_sess_df = pd.DataFrame(sess_data)
    master_df = pd.concat(session_dfs, ignore_index=True)

    # Add a column for TMS condition
    master_df['TMS_condition'] = ''
    master_df.loc[(master_df['istms'] == 0) & (master_df['instimVF'] == 1), 'TMS_condition'] = 'NoTMS intoPF'
    master_df.loc[(master_df['istms'] == 0) & (master_df['instimVF'] == 0), 'TMS_condition'] = 'NoTMS outPF'
    master_df.loc[(master_df['istms'] == 1) & (master_df['instimVF'] == 1), 'TMS_condition'] = 'TMS intoPF'
    master_df.loc[(master_df['istms'] == 1) & (master_df['instimVF'] == 0), 'TMS_condition'] = 'TMS outPF'
    master_df = master_df.reset_index(drop = True)
    master_df.to_csv(p[df_name], index = False)