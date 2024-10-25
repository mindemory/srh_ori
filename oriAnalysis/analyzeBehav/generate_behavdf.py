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

summary_df = pd.read_csv(os.path.join(p['analysis'], 'V1ori_TMS_meta.csv'))
All_metadata = {row['Subject ID']: row for _, row in summary_df.iterrows()}

mdf_fname = 'master_df.csv'
df_name = 'df_fname'
p[df_name] = os.path.join(p['meta'], mdf_fname)


print('Creating a new dataframe...')
# Find subjects and days that have been run so far
sub_dirs = [d for d in os.listdir(os.path.join(p['analysis'])) if d.startswith('sub')]
# sub_dirs = ['sub01', 'sub02', 'sub05', 'sub07', 'sub08', 'sub11', 'sub12']
subjIDs = sorted([int(ss[-2:]) for ss in sub_dirs])

print(f"We have {len(subjIDs)} subjects so far: {subjIDs}")
print()

master_df = pd.DataFrame()
# Create a master dataframe
for ss in subjIDs:
    temp_df = pd.read_csv(os.path.join(p['analysis'], f'sub{ss:02}', f'sub{ss:02}_behav.csv'))
    temp_df['subjInitials'] = summary_df[summary_df['Subject ID'] == ss].iloc[0]['Subject Initials']
    temp_df['hemisphereStimulated'] = summary_df[summary_df['Subject ID'] == ss].iloc[0]['Hemisphere stimulated']
    temp_df['PT'] = summary_df[summary_df['Subject ID'] == ss].iloc[0]['PT']

    if ss == 7:
        # Drop rows for day == 2 and block == 7, 8, 12, 13
        temp_df = temp_df[~((temp_df['day'] == 2) & (temp_df['block'].isin([7, 8, 12, 13])))]
    
    master_df = pd.concat([master_df, temp_df])

