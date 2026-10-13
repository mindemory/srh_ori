"""
R01_build_behavdf.py

Builds the trial-level master dataframe for the bias-replication study
(ori_task_biasreplication.m). It reproduces, step for step, what the original
study's pipeline did:

  1. A03_BehavAnalysis.m  -> one row per trial from each block's matFile plus
                             the matching taskMap_biasrep_* entry
  2. generate_behavdf.py  -> initOffset, errOri_scaled, abserrorOri, labels,
                             global 3-SD LargeError flag, oriArray / oriSteps

Differences from the original that are forced by the data:
  - No V1ori_TMS_meta.csv exists for this cohort, so hemisphereStimulated is
    derived from where the in-PF stimulus was drawn (in-PF in the right visual
    field -> left hemisphere stimulated), which is how the PF was defined.
  - Subject 27 / subject 07 block exclusions from the original are not applied
    (they were specific to that cohort).

Usage:
    python R01_build_behavdf.py [data_root]
data_root defaults to $BIASREP_ROOT or '/Users/mrugank/Documents/TMS ori/bias_rep'.
Writes <data_root>/analysis/master_df_biasrep.pkl (with oriArray) and .csv.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat

DATA_ROOT = (sys.argv[1] if len(sys.argv) > 1 else
             os.environ.get('BIASREP_ROOT', '/Users/mrugank/Documents/TMS ori/bias_rep'))
ORI_DIR = os.path.join(DATA_ROOT, 'data', 'ori_biasreplication_data')
PHOS_DIR = os.path.join(DATA_ROOT, 'data', 'phosphene_data')
OUT_DIR = os.path.join(DATA_ROOT, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)


def _load(fname):
    return loadmat(fname, squeeze_me=True, struct_as_record=False)


# ---------------------------------------------------------------------------
# Step 1: A03_BehavAnalysis.m equivalent
# ---------------------------------------------------------------------------
rows = []
sub_dirs = sorted(d for d in os.listdir(ORI_DIR) if d.startswith('sub'))
for sub in sub_dirs:
    subjID = int(sub[3:])
    day_dirs = sorted(d for d in os.listdir(os.path.join(ORI_DIR, sub)) if d.startswith('day'))
    for didx, day in enumerate(day_dirs, start=1):
        taskMap = _load(os.path.join(PHOS_DIR, sub, f'taskMap_biasrep_{sub}_day{didx:02d}.mat'))['taskMap']
        block_dirs = sorted(d for d in os.listdir(os.path.join(ORI_DIR, sub, day)) if d.startswith('block'))
        for bidx, block in enumerate(block_dirs, start=1):
            mats = sorted(glob.glob(os.path.join(ORI_DIR, sub, day, block, 'matFile*.mat')))
            matFile = _load(mats[0])['matFile']
            tMap = taskMap[bidx - 1]
            if int(matFile.parameters.TMScond) != int(tMap.TMScond):
                raise ValueError(f'{sub} {day} {block}: matFile TMScond != taskMap TMScond')

            xC, yC = float(matFile.screen.xCenter), float(matFile.screen.yCenter)
            timeReport = np.atleast_1d(matFile.timeReport)
            respReport = np.atleast_1d(matFile.respReport)
            targX = np.atleast_1d(tMap.targX).astype(float)
            targY = np.atleast_1d(tMap.targY).astype(float)
            stimVF = np.atleast_1d(tMap.stimVF)
            targOri = np.atleast_1d(tMap.targOri)
            targOri_ = np.atleast_1d(tMap.targOri_)

            for tr in range(len(respReport)):
                polang = np.degrees(np.arctan2(targY[tr] - yC, targX[tr] - xC))
                row = {
                    'subjID': subjID, 'day': didx, 'block': bidx,
                    'TMScond': int(tMap.TMScond),
                    'stimPF': int(stimVF[tr]),
                    'targOriCategory': float(targOri[tr]),
                    'targOriActual': float(targOri_[tr]),
                    'stimSize': float(tMap.stimSize),
                    'targX': targX[tr], 'targY': targY[tr],
                    'xCenter': xC, 'yCenter': yC,
                    'rho': np.hypot(targX[tr] - xC, targY[tr] - yC),
                    'polang': polang + 360 if polang < 0 else polang,
                }
                for f in timeReport[tr]._fieldnames:
                    row[f] = float(getattr(timeReport[tr], f))
                rr = respReport[tr]
                row['initOri'] = float(rr.initOri)
                row['finalOri'] = float(rr.finalOri)
                row['RT'] = float(rr.RT)
                row['errorOri'] = float(rr.errorOri)
                arr = np.atleast_1d(np.asarray(rr.oriArray, dtype=float))
                row['oriArray'] = arr[~np.isnan(arr)]
                rows.append(row)

master_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Step 2: generate_behavdf.py equivalent
# ---------------------------------------------------------------------------
# Hemisphere stimulated: in-PF stimulus in the right visual field means the
# left hemisphere was stimulated (and vice versa).
inpf_side = (master_df[master_df['stimPF'] == 1]
             .groupby('subjID')
             .apply(lambda d: 'Left' if (d['targX'] > d['xCenter']).all()
                    else ('Right' if (d['targX'] < d['xCenter']).all() else 'Mixed')))
if (inpf_side == 'Mixed').any():
    raise ValueError(f'In-PF stimulus switches sides for: {list(inpf_side[inpf_side == "Mixed"].index)}')
master_df['hemisphereStimulated'] = master_df['subjID'].map(inpf_side)

master_df['oriSteps'] = master_df['oriArray'].apply(len)
initOffset = master_df['initOri'] - master_df['targOriActual']
master_df['initOffset'] = (initOffset + 90) % 180 - 90
errTemp = master_df['finalOri'] - master_df['targOriActual']
master_df['errOri_scaled'] = (errTemp + 90) % 180 - 90
master_df['abserrorOri'] = np.abs(master_df['errOri_scaled'])
master_df['stimPF_label'] = master_df['stimPF'].apply(lambda x: 'outPF' if x == 0 else 'inPF')
master_df['TMScond_label'] = master_df['TMScond'].apply(lambda x: 'No TMS' if x == 1 else 'Early TMS' if x == 2 else 'Middle TMS')
master_df['TMSstatus'] = master_df['TMScond'].apply(lambda x: 'TMS' if x != 1 else 'No TMS')

# --- Outlier Detection (Global 3SD), identical rule to generate_behavdf.py ---
global_mean_err = master_df['errOri_scaled'].mean()
global_sd_err = master_df['errOri_scaled'].std()
lower_z = global_mean_err - 3 * global_sd_err
upper_z = global_mean_err + 3 * global_sd_err
master_df['LargeError'] = 0
master_df.loc[(master_df['errOri_scaled'] < lower_z) | (master_df['errOri_scaled'] > upper_z), 'LargeError'] = 1

print(f"Subjects ({master_df['subjID'].nunique()}): {sorted(master_df['subjID'].unique())}")
print(f"Trials: {len(master_df)}")
print(f"Outlier Detection: Global Mean={global_mean_err:.2f}, SD={global_sd_err:.2f}, Bounds=[{lower_z:.2f}, {upper_z:.2f}]")
print(f"Total LargeErrors: {master_df['LargeError'].sum()}/{len(master_df)} ({master_df['LargeError'].mean() * 100:.1f}%)")

master_df.to_pickle(os.path.join(OUT_DIR, 'master_df_biasrep.pkl'))
master_df.drop(columns=['oriArray']).to_csv(os.path.join(OUT_DIR, 'master_df_biasrep.csv'), index=False)
print(f"Saved to {OUT_DIR}")
