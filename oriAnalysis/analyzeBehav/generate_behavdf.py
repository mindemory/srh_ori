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
if hostname == 'zod.psych.nyu.edu' or hostname == 'zod':
    p['datd'] = '/d/DATD/datd/TMS_ori'
else:
    p['datd'] = '/Users/mrugankdake/Documents/Clayspace/EEG_TMS/datd/TMS_ori'

p['data'] = os.path.join(p['datd'], 'data', 'ori_data')
p['analysis'] = os.path.join(p['datd'], 'analysis')
p['meta'] = os.path.join(p['analysis'], 'meta_analysis')

if not os.path.exists(p['meta']):
    os.makedirs(p['meta'])

# summary_df = pd.read_csv(os.path.join(p['analysis'], 'V1ori_TMS_meta.csv'))
# All_metadata = {row['Subject ID']: row for _, row in summary_df.iterrows()}

mdf_fname = 'master_df.csv'
df_name = 'df_fname'
p[df_name] = os.path.join(p['meta'], mdf_fname)

if os.path.exists(p[df_name]):
    print('Loading existing dataframe! If this is not desired, delete the current master_df.csv')
    master_df = pd.read_csv(p[df_name]) 
else:
    print('Creating a new dataframe...')
    # Find subjects and days that have been run so far
    sub_dirs = [d for d in os.listdir(os.path.join(p['analysis'])) if d.startswith('sub')]
    sub_dirs = ['sub99']
    subjIDs = sorted([int(ss[-2:]) for ss in sub_dirs])
    
    print(f"We have {len(subjIDs)} subjects so far: {subjIDs}")
    print()

    master_df = pd.DataFrame()
    # Create a master dataframe
    for ss in subjIDs:
        temp_df = pd.read_csv(os.path.join(p['analysis'], f'sub{ss:02}', f'sub{ss:02}_behav.csv'))
        

        master_df = pd.concat([master_df, temp_df])
        # master_df.append(temp_df)
        # Extract meta-data for this subject
        # metadata = All_metadata[ss]
        # subdir_name = f'sub{ss:02}'
        # subject_row = summary_df[summary_df['Subject ID'] == ss].iloc[0]

        # subjdir = os.path.join(p['ori_data'], subdir_name)
        # phosphene_data_path = os.path.join(p['data'], 'phosphene_data', subdir_name)

        # # Load timestructure for this subject
        # timeStructFname = os.path.join(subjdir, 'timeStruct.mat')
        # timeStruct = loadmat(timeStructFname)['timeStruct']

        # # Get list of all days this subject was run on
        # day_dirs = sorted([d for d in os.listdir(subjdir) if d.startswith('day')])
        # days = []

        # for dd in day_dirs:
        #     th_day = int(dd[-2:])
        #     print(f'Running subject {ss}, day {th_day}...')
        #     daydir = os.path.join(subjdir, dd)

        #     # Extract times for this day
        #     idur = timeStruct['initDuration'][0, 0][th_day-1, :, :]
        #     sdur = timeStruct['sampleDuration'][0, 0][th_day-1, :, :]
        #     d1dur = timeStruct['delay1Duration'][0, 0][th_day-1, :, :]
        #     d2dur = timeStruct['delay2Duration'][0, 0][th_day-1, :, :]
        #     rdur = timeStruct['respDuration'][0, 0][th_day-1, :, :]
        #     fdur = timeStruct['feedbackDuration'][0, 0][th_day-1, :, :]
        #     itidur = timeStruct['itiDuration'][0, 0][th_day-1, :, :]
        #     trdur = timeStruct['trialDuration'][0, 0][th_day-1, :, :]

    # Save the master dataframe
    # master_df = pd.concat(master_df)
    master_df.to_csv(p[df_name], index=False)

