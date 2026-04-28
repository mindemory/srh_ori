import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

import generate_behavdf as gm

print("Data loaded. Starting SAT analysis...")
df = gm.master_df.copy()

# Filter basic outliers and invalid trials
df = df[df['LargeError'] == 0]
df = df[df['RT'] < 3.0]
df = df[df['oriSteps'] > 0]
df = df.dropna(subset=['RT', 'errorOri', 'abserrorOri'])

# Define TMS Condition
df['TMS_Group'] = df['TMScond_label'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
def make_tms_cond(row):
    if row['TMS_Group'] in ['No TMS', 'NoTMS']: return 'NoTMS'
    return f"TMS {row['stimPF_label']}"

df['TMS_Condition'] = df.apply(make_tms_cond, axis=1)

sns.set_theme(style="ticks", context="talk")
fig, axes_2d = plt.subplots(2, 2, figsize=(14, 12))
axes = axes_2d.flatten()

# 0. RT Distribution Histogram (Pooled)
sns.histplot(df['RT'], bins=30, ax=axes[0], color='black', alpha=0.3, element="step", fill=True)
axes[0].set_title('Overall RT Distribution', fontweight='bold', pad=15)
axes[0].set_xlabel('Reaction Time (s)')
axes[0].set_ylabel('Count')

# 1. Compile Subject-level deciles for CI Bootstrapping (Pooled)
decile_data = []
n_bins = 30
df['abs_initOffset'] = df['initOffset'].abs()

for subj, subj_df in df.groupby('subjID'):
    if len(subj_df) < n_bins:
        continue
    try:
        subj_df = subj_df.copy()
        
        # RT deciles
        subj_df['RT_Decile'] = pd.qcut(subj_df['RT'], q=n_bins, labels=False, duplicates='drop')
        subj_agg_rt = subj_df.groupby('RT_Decile').agg({
            'RT': 'median',
            'abserrorOri': 'mean'
        }).reset_index()
        subj_agg_rt['subjID'] = subj
        
        # InitOffset deciles
        subj_df['Offset_Decile'] = pd.qcut(subj_df['abs_initOffset'], q=n_bins, labels=False, duplicates='drop')
        subj_agg_off = subj_df.groupby('Offset_Decile').agg({
            'abs_initOffset': 'median',
            'RT': 'mean',
            'abserrorOri': 'mean'
        }).reset_index()
        subj_agg_off['subjID'] = subj
        
        # Merge decile stats
        subj_combined = pd.merge(subj_agg_rt, subj_agg_off[['Offset_Decile', 'abs_initOffset', 'RT', 'abserrorOri']], left_index=True, right_index=True, suffixes=('', '_off'))
        decile_data.append(subj_combined)
    except ValueError:
        pass

df_deciles = pd.concat(decile_data)

# Normalize the X-axis mapping per pooled decile for Seaborn
df_deciles['Group_Mean_RT'] = df_deciles.groupby('RT_Decile')['RT'].transform('mean')
df_deciles['Group_Mean_Offset'] = df_deciles.groupby('Offset_Decile')['abs_initOffset'].transform('mean')

# Plot 1: Speed-Accuracy Tradeoff 
sns.lineplot(
    data=df_deciles, x='Group_Mean_RT', y='abserrorOri', 
    color='black', errorbar=('ci', 95), marker='o', ax=axes[1], linewidth=2.5
)
axes[1].set_title('Overall Speed-Accuracy Tradeoff', fontweight='bold', pad=15)
axes[1].set_xlabel('Reaction Time (s)')
axes[1].set_ylabel('MAE (°)')

# Plot 2: RT vs Absolute Init Offset
sns.lineplot(
    data=df_deciles, x='Group_Mean_Offset', y='RT_off', 
    color='black', errorbar=('ci', 95), marker='o', ax=axes[2], linewidth=2.5
)
axes[2].set_title('Reaction Time vs. Init Offset', fontweight='bold', pad=15)
axes[2].set_xlabel('Absolute Init Offset (°)')
axes[2].set_ylabel('Reaction Time (s)')

# Plot 3: MAE vs Absolute Init Offset
sns.lineplot(
    data=df_deciles, x='Group_Mean_Offset', y='abserrorOri_off', 
    color='black', errorbar=('ci', 95), marker='o', ax=axes[3], linewidth=2.5
)
axes[3].set_title('Accuracy vs. Init Offset', fontweight='bold', pad=15)
axes[3].set_xlabel('Absolute Init Offset (°)')
axes[3].set_ylabel('MAE (°)')

sns.despine(offset=10)
plt.tight_layout(pad=2)

print("Displaying 2x2 visualization grid with 30 combined bins...")
plt.show()
