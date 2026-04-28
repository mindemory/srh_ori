import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import generate_behavdf as gm

print("Loaded master_df.")

def get_hesitation_score(ori_array):
    """
    Quantifies motor conflict / hesitation.
    0.0 = Pure directional movement.
    1.0 = Maximum wiggling (50/50 directional split).
    """
    if len(ori_array) < 2: return 0.0
    diffs = np.diff(ori_array)
    diffs[diffs > 90] -= 180
    diffs[diffs < -90] += 180
    
    cw_dist = np.sum(np.abs(diffs[diffs < 0]))
    ccw_dist = np.sum(diffs[diffs > 0])
    total_dist = cw_dist + ccw_dist
    
    if total_dist == 0: return 0.0
    
    cw_tendency = cw_dist / total_dist
    # Map 0.5 (max indecision) to 1.0, and 0.0/1.0 (certainty) to 0.0
    hesitation = 1.0 - 2.0 * np.abs(cw_tendency - 0.5)
    return hesitation

df = gm.master_df.copy()

print("Preprocessing and cleaning data...")
n_orig = len(df)
df = df[df['LargeError'] == 0]
df = df[df['RT'] < 3.0]
df = df[df['oriSteps'] > 0]
print(f"Dropped {n_orig - len(df)} bad trials (Outliers/Timeouts).")

print("Calculating Motor Hesitation (~5 seconds)...")
df['Hesitation'] = df['oriArray'].apply(get_hesitation_score)
df['InitOffset'] = (df['initOri'] - df['targOriActual'] + 90) % 180 - 90

# Radial vs Tangential mapping logic
def get_ori_type(row):
    ori = row['targOriCategory']
    hemi = row['hemisphereStimulated']
    pf   = row['stimPF_label']
    target_vf = ('right' if pf == 'inPF' else 'left') if hemi == 'Left' else ('left' if pf == 'inPF' else 'right')
    if target_vf == 'right':
        return 'Radial' if ori in [60, 120] else 'Tangential'
    else:
        return 'Radial' if ori in [30, 150] else 'Tangential'

df['OriType'] = df.apply(get_ori_type, axis=1)

df['TMS_Group'] = df['TMScond_label'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
def make_tms_cond(row):
    if row['TMS_Group'] in ['No TMS', 'NoTMS']: return 'NoTMS'
    return f"TMS_{row['stimPF_label']}"

df['TMS_Condition'] = df.apply(make_tms_cond, axis=1)
df['Full_Condition'] = df['TMS_Condition'] + '_' + df['OriType']

bin_edges = np.arange(-91, 92, 1)
df['Offset_Bin'] = pd.cut(df['InitOffset'], bins=bin_edges)

# Subject-level aggregation (1-degree buckets)
sub_stats = df.groupby(
    ['subjID', 'Full_Condition', 'Offset_Bin'], observed=True
).agg(
    Mean_Hesitation=('Hesitation', 'mean')
).reset_index()

# 提取 bin 的中间点
sub_stats['Bin_Mid'] = sub_stats['Offset_Bin'].apply(lambda x: x.mid).astype(float)

# Population aggregation
pop_stats = sub_stats.groupby(
    ['Full_Condition', 'Bin_Mid'], observed=True
).agg(
    Pop_Mean=('Mean_Hesitation', 'mean'),
    Pop_SEM=('Mean_Hesitation', lambda x: np.std(x) / np.sqrt(len(x)))
).reset_index().dropna(subset=['Pop_Mean'])

# -------------------------------------------------------
# Visualization: Smoothed Mountain Plot
# -------------------------------------------------------
SIGMA = 3.0  # Smoothing kernel width in degrees
condition_styles = {
    'NoTMS_Radial':        {'color': 'black',   'marker': 'o', 'label': 'NoTMS Radial',     'alpha': 1.0},
    'NoTMS_Tangential':    {'color': 'black',   'marker': 's', 'label': 'NoTMS Tangential', 'alpha': 0.6},
    'TMS_inPF_Radial':     {'color': 'red',     'marker': 'o', 'label': 'TMS inPF Radial',  'alpha': 1.0},
    'TMS_inPF_Tangential': {'color': 'red',     'marker': 's', 'label': 'TMS inPF Tangential','alpha': 0.6},
    'TMS_outPF_Radial':    {'color': 'blue',    'marker': 'o', 'label': 'TMS outPF Radial', 'alpha': 1.0},
    'TMS_outPF_Tangential':{'color': 'blue',    'marker': 's', 'label': 'TMS outPF Tangential','alpha': 0.6},
}

fig, ax = plt.subplots(figsize=(12, 8))

for cond, style in condition_styles.items():
    cond_df = pop_stats[pop_stats['Full_Condition'] == cond].sort_values('Bin_Mid')
    if len(cond_df) < 10: continue

    x = cond_df['Bin_Mid'].values
    y_mean = gaussian_filter1d(cond_df['Pop_Mean'].values, sigma=SIGMA)
    y_sem = gaussian_filter1d(cond_df['Pop_SEM'].values, sigma=SIGMA)

    # Plot smooth line
    ax.plot(x, y_mean, color=style['color'], 
            linestyle='-' if 'Radial' in cond else '--', 
            linewidth=3, alpha=style['alpha'],
            label=style['label'])
    
    # Plot shaded SEM ribbon
    ax.fill_between(x, y_mean - y_sem, y_mean + y_sem,
                    color=style['color'], alpha=0.15, lw=0)

ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='0°')
ax.axhline(0, color='gray', alpha=0.3)

ax.set_xlabel("Initial Offset (°)", fontsize=14)
ax.set_ylabel("Hesitation Score", fontsize=14)
ax.set_title("Motor Hesitation", fontsize=18, fontweight='bold')

ax.set_ylim(-0.02, 0.45) 
ax.set_xlim(-95, 95)
ax.legend(loc='upper right', fontsize=10, ncol=2, frameon=False)
ax.grid(False)

plt.tight_layout()
print("Launching motor hesitation GUI...")
plt.show()
