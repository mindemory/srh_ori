import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

import generate_behavdf as gm

print("Data loaded. Starting SAT analysis...")
df = gm.master_df.copy()
df.rename(columns={'TMScond_label': 'TMS'}, inplace=True)

# Filter basic outliers and invalid trials
df = df[df['LargeError'] == 0]
df = df[df['RT'] < 3.0]
df = df[df['oriSteps'] > 0]
df = df.dropna(subset=['RT', 'errorOri', 'abserrorOri'])

sns.set_theme(style="ticks", context="talk")
fig, axes_2d = plt.subplots(2, 2, figsize=(14, 12))
axes = axes_2d.flatten()

# 0. RT Distribution KDE (Split by TMS)
sns.kdeplot(data=df, x='RT', hue='TMS', palette={'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}, ax=axes[0], fill=False, linewidth=2, common_norm=False)
axes[0].set_title('RT distribution', fontweight='bold', pad=15)
axes[0].set_xlabel('Reaction Time (s)'); axes[0].set_ylabel('Density')

# 1. Compile Subject-level deciles for CI Bootstrapping (Split by TMS)
decile_data = []
n_bins = 25  # Finer bins for more detail
df['abs_initOffset'] = df['initOffset'].abs()

for (subj, tms), subj_df in df.groupby(['subjID', 'TMS']):
    if len(subj_df) < n_bins: continue
    try:
        subj_df = subj_df.copy()
        
        # RT deciles and 3-bin smoothing
        subj_df['RT_Decile'] = pd.qcut(subj_df['RT'], q=n_bins, labels=False, duplicates='drop')
        subj_agg_rt = subj_df.groupby('RT_Decile').agg({'RT': 'mean', 'abserrorOri': 'mean'}).reset_index()
        subj_agg_rt.update(subj_agg_rt[['RT', 'abserrorOri']].rolling(window=3, center=True).mean())
        
        # InitOffset deciles and 3-bin smoothing
        subj_df['Offset_Decile'] = pd.qcut(subj_df['abs_initOffset'], q=n_bins, labels=False, duplicates='drop')
        subj_agg_off = subj_df.groupby('Offset_Decile').agg({'abs_initOffset': 'mean', 'RT': 'mean', 'abserrorOri': 'mean'}).reset_index()
        subj_agg_off.update(subj_agg_off[['abs_initOffset', 'RT', 'abserrorOri']].rolling(window=3, center=True).mean())
        
        # Add labels
        subj_agg_rt['subjID'] = subj; subj_agg_rt['TMS'] = tms
        subj_agg_off['subjID'] = subj; subj_agg_off['TMS'] = tms
        
        decile_data.append((subj_agg_rt, subj_agg_off))
    except ValueError: pass

# Concatenate and Aligned Plotting Data
df_rt_all = pd.concat([d[0] for d in decile_data])
df_off_all = pd.concat([d[1] for d in decile_data])

# Use Group Means for the X-axis to align points for a smoother plot
df_rt_all['Group_RT'] = df_rt_all.groupby(['RT_Decile', 'TMS'])['RT'].transform('mean')
df_off_all['Group_Offset'] = df_off_all.groupby(['Offset_Decile', 'TMS'])['abs_initOffset'].transform('mean')

tms_colors = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}

# Plot 1: Angular Error vs Reaction Time 
sns.lineplot(data=df_rt_all, x='Group_RT', y='abserrorOri', hue='TMS', palette=tms_colors, errorbar=('ci', 95), marker=None, ax=axes[1], linewidth=3)
axes[1].set_title('Angular Error vs Reaction Time', fontweight='bold', pad=15)
axes[1].set_xlabel('Reaction Time (s)'); axes[1].set_ylabel('Angular Error (°)')

# Plot 2: RT vs Absolute Init Offset
sns.lineplot(data=df_off_all, x='Group_Offset', y='RT', hue='TMS', palette=tms_colors, errorbar=('ci', 95), marker=None, ax=axes[2], linewidth=3)
axes[2].set_title('Reaction Time vs. Init Offset', fontweight='bold', pad=15)
axes[2].set_xlabel('Absolute Init Offset (°)'); axes[2].set_ylabel('Reaction Time (s)')

# Plot 3: Angular Error vs Absolute Init Offset
sns.lineplot(data=df_off_all, x='Group_Offset', y='abserrorOri', hue='TMS', palette=tms_colors, errorbar=('ci', 95), marker=None, ax=axes[3], linewidth=3)
axes[3].set_title('Angular Error vs. Init Offset', fontweight='bold', pad=15)
axes[3].set_xlabel('Absolute Init Offset (°)'); axes[3].set_ylabel('Angular Error (°)')

for ax in axes: ax.legend(fontsize=8, frameon=False)
sns.despine(offset=10)
plt.tight_layout(pad=2)

print("Displaying 2x2 visualization grid with 30 combined bins...")
plt.show()
