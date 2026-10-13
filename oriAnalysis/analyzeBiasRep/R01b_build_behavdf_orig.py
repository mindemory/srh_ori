"""
R01b_build_behavdf_orig.py

Builds the original study's (Study 1) master dataframe exactly as
analyzeBehav/generate_behavdf.py does, from the per-subject A03 outputs
(analysis/subXX/subXX_behav.csv) and V1ori_TMS_meta.csv:
  - subject 27 excluded; subject 07 day 2 blocks 7, 8, 12, 13 dropped
  - initOffset, errOri_scaled, abserrorOri, labels, global 3-SD LargeError
  - oriArray rebuilt from the oriArray_* columns
hemisphereStimulated comes from the metadata sheet (as in the paper). The
hemisphere implied by where the in-PF stimulus was drawn is stored as
hemisphereStimulated_pos so the two can be compared.

Usage: python R01b_build_behavdf_orig.py [orig_root]
orig_root defaults to $ORIG_ROOT or '/Users/mrugank/Documents/TMS ori/orig_data'.
Writes <orig_root>/analysis/master_df_orig.pkl.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = (sys.argv[1] if len(sys.argv) > 1 else
        os.environ.get('ORIG_ROOT', '/Users/mrugank/Documents/TMS ori/orig_data'))
ANALYSIS = os.path.join(ROOT, 'analysis')

summary_df = pd.read_csv(os.path.join(ANALYSIS, 'V1ori_TMS_meta.csv'))
sub_dirs = [d for d in os.listdir(ANALYSIS) if d.startswith('sub')]
subjIDs = sorted([int(ss[-2:]) for ss in sub_dirs])
subjIDs = [ss for ss in subjIDs if ss != 27]
print(f"We have {len(subjIDs)} subjects so far: {subjIDs}")

master_df = pd.DataFrame()
for ss in subjIDs:
    temp_df = pd.read_csv(os.path.join(ANALYSIS, f'sub{ss:02}', f'sub{ss:02}_behav.csv'))
    meta = summary_df[summary_df['Subject ID'] == ss].iloc[0]
    temp_df['subjInitials'] = meta['Subject Initials']
    temp_df['hemisphereStimulated'] = meta['Hemisphere stimulated']
    temp_df['PT'] = meta['PT']
    if ss == 7:
        # Drop rows for day == 2 and block == 7, 8, 12, 13
        temp_df = temp_df[~((temp_df['day'] == 2) & (temp_df['block'].isin([7, 8, 12, 13])))]
    master_df = pd.concat([master_df, temp_df])

oriArrayCols = [col for col in master_df.columns if col.startswith('oriArray_')]
oriSteps = master_df[oriArrayCols].notna().sum(axis=1)
master_df = pd.concat([master_df, oriSteps.rename('oriSteps')], axis=1)
initOffset = master_df['initOri'] - master_df['targOriActual']
master_df['initOffset'] = (initOffset + 90) % 180 - 90
errTemp = master_df['finalOri'] - master_df['targOriActual']
master_df['errOri_scaled'] = (errTemp + 90) % 180 - 90
master_df['abserrorOri'] = np.abs(master_df['errOri_scaled'])
master_df['stimPF_label'] = master_df['stimPF'].apply(lambda x: 'outPF' if x == 0 else 'inPF')
master_df['TMScond_label'] = master_df['TMScond'].apply(lambda x: 'No TMS' if x == 1 else 'Early TMS' if x == 2 else 'Middle TMS')
master_df['TMSstatus'] = master_df['TMScond'].apply(lambda x: 'TMS' if x != 1 else 'No TMS')

# --- Outlier Detection (Global 3SD) ---
global_mean_err = master_df['errOri_scaled'].mean()
global_sd_err = master_df['errOri_scaled'].std()
lower_z = global_mean_err - 3 * global_sd_err
upper_z = global_mean_err + 3 * global_sd_err
master_df['LargeError'] = 0
master_df.loc[(master_df['errOri_scaled'] < lower_z) | (master_df['errOri_scaled'] > upper_z), 'LargeError'] = 1
print(f"Outlier Detection: Global Mean={global_mean_err:.2f}, SD={global_sd_err:.2f}, Bounds=[{lower_z:.2f}, {upper_z:.2f}]")
print(f"Total LargeErrors: {master_df['LargeError'].sum()}/{len(master_df)} ({(master_df['LargeError'].sum()/len(master_df))*100:.1f}%)")

master_df['oriArray'] = master_df[oriArrayCols].apply(lambda row: row.dropna().to_numpy(), axis=1)
master_df = master_df.drop(columns=oriArrayCols).reset_index(drop=True)

# --- Additions for the geometry analyses (not in generate_behavdf.py) ---
# Screen centre recovered from the stored position, eccentricity and polar angle.
master_df['xCenter'] = master_df['targX'] - master_df['rho'] * np.cos(np.radians(master_df['polang']))
master_df['yCenter'] = master_df['targY'] - master_df['rho'] * np.sin(np.radians(master_df['polang']))
pos = (master_df[master_df['stimPF'] == 1].groupby('subjID')
       .apply(lambda d: 'Left' if (d['targX'] > d['xCenter']).all() else ('Right' if (d['targX'] < d['xCenter']).all() else 'Mixed')))
master_df['hemisphereStimulated_pos'] = master_df['subjID'].map(pos)
mism = (master_df.groupby('subjID')[['hemisphereStimulated', 'hemisphereStimulated_pos']].first()
        .query('hemisphereStimulated != hemisphereStimulated_pos'))
print('Metadata hemisphere vs stimulus position mismatches:', 'none' if mism.empty else mism.to_dict('index'))

master_df.to_pickle(os.path.join(ANALYSIS, 'master_df_orig.pkl'))
print(f"Saved {os.path.join(ANALYSIS, 'master_df_orig.pkl')} ({len(master_df)} trials)")
