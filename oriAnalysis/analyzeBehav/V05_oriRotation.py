import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import generate_behavdf as gm

print("Loaded master_df.")

def get_cw_tendency_continuous(ori_array):
    """
    Returns the fraction of total distance moved in the Clockwise direction.
    1.0 = Pure CW rotation, 0.0 = Pure CCW rotation.
    """
    if len(ori_array) < 2: return 0.5
    diffs = np.diff(ori_array)
    diffs[diffs > 90] -= 180
    diffs[diffs < -90] += 180
    
    cw_dist = np.sum(np.abs(diffs[diffs < 0]))
    ccw_dist = np.sum(diffs[diffs > 0])
    total_dist = cw_dist + ccw_dist
    
    if total_dist == 0: return 0.5
    return cw_dist / total_dist

df = gm.master_df.copy()

print("Preprocessing and cleaning data...")
# --- FILTERING "BAD" TRIALS ---
n_orig = len(df)
df = df[df['LargeError'] == 0]
df = df[df['RT'] < 3.0]
df = df[df['oriSteps'] > 0]
print(f"Dropped {n_orig - len(df)} bad trials total (Outliers/Timeouts).")

print("Unwrapping arrays (~5 seconds)...")
df['CW_Tendency'] = df['oriArray'].apply(get_cw_tendency_continuous)
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

bin_edges = np.arange(-95, 96, 5)
df['Offset_Bin'] = pd.cut(df['InitOffset'], bins=bin_edges)

# Subject-level aggregation
sub_stats = df.groupby(
    ['subjID', 'Full_Condition', 'Offset_Bin'], observed=True
).agg(
    Mean_Tendency=('CW_Tendency', 'mean'),
    Bin_Mid=('InitOffset', 'mean')
).reset_index()

# Population aggregation
pop_stats = sub_stats.groupby(
    ['Full_Condition', 'Offset_Bin'], observed=True
).agg(
    Pop_Mean=('Mean_Tendency', 'mean'),
    Pop_SEM=('Mean_Tendency', lambda x: np.std(x) / np.sqrt(len(x))),
    Final_Bin_Mid=('Bin_Mid', 'mean')
).reset_index().dropna(subset=['Pop_Mean'])

# -------------------------------------------------------
# Visualization
# -------------------------------------------------------
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
    cond_df = pop_stats[pop_stats['Full_Condition'] == cond]
    if len(cond_df) == 0: continue

    ax.errorbar(
        cond_df['Final_Bin_Mid'], cond_df['Pop_Mean'],
        yerr=cond_df['Pop_SEM'],
        color=style['color'], marker=style['marker'],
        linestyle='-', linewidth=0.8, markersize=8,
        capsize=3, elinewidth=1.2, alpha=style['alpha'],
        label=style['label']
    )

ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='green', linestyle='-', alpha=0.3)

ax.set_xlabel("Initial Offset Relative to Target (°)", fontsize=13)
ax.set_ylabel("Mean CW Motor Preference (0.0 to 1.0)", fontsize=13)
ax.set_title("Population Continuous Psychometric: Motor preference vs. Offset\n(Aggregates Binary Choices + Subthreshold Wiggles)", fontsize=16, fontweight='bold')

ax.set_ylim(-0.02, 1.02)
ax.set_xlim(-95, 95)
ax.legend(loc='upper right', fontsize=9, ncol=2, frameon=False)
ax.grid(False)

plt.tight_layout()
print("Launching continuous motor preference GUI...")
plt.show()
