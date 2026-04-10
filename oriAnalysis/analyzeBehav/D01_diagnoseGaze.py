"""
D01_diagnoseGaze.py - Diagnostic script for gaze data investigation.

Goals:
1. Compare Mean vs Median traces (to check for outlier influence).
2. Plot Sample Count (N) over time (to check for data loss/dropout).
3. Visualize distribution of position at specific timepoints.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import generate_behavdf as gm
from GazeAnalyzer import GazeAnalyzer

# Initialize
datd_path = gm.p['datd']
analyzer = GazeAnalyzer(datd_path=datd_path, master_df=gm.master_df)

# Load Subject 01 NoTMS
subj_id = 1
print(f"\n=== Loading Subject {subj_id} ===")
gaze_data = analyzer.load_gaze_data(subj_id)
behav_df = analyzer.get_behavioral_df(subj_id)
filtered_df = analyzer.filter_trials(behav_df, TMScond=1)

print("\n=== Extracting Traces ===")
traces_df = analyzer.extract_traces(
    df=filtered_df,
    gaze_data=gaze_data,
    gaze_threshold=1.5,
    downsample=10,
    baseline_window=(-1.0, 0.0)
)

print("\n=== Computing Diagnostics ===")
# Group by TimeOnly to see global issues first, then by OriCat
# We'll stick to 'Time' and 'OriCat'
columns = ['X', 'Y', 'Pupil_Z']

# Custom aggregation: Mean, Median, Count, Brow-Forsythe (Variance)
diagnostic_aggs = {}

for col in columns:
    # Aggregation
    agg = traces_df.groupby(['Time', 'OriCat'])[col].agg(['mean', 'median', 'count', 'std']).reset_index()
    diagnostic_aggs[col] = agg

print("\n=== Generating Diagnostic Plots ===")
out_dir = os.path.join(datd_path, 'figures', 'gazeDiagnostics')
os.makedirs(out_dir, exist_ok=True)

# 1. Plot Mean vs Median for X and Y (collapsed across Ori to see general stability, or separate?)
# Let's show separate OriCats in separate subplots if noisy, or just one illustrative plot
# We'll plot X and Y for ALL orientations first (average marginal)
global_agg = traces_df.groupby('Time')[['X', 'Y']].agg(['mean', 'median', 'count']).reset_index()

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
fig.suptitle(f"Subject {subj_id:02d} | Global Diagnostics (All Orientations)", fontsize=16)

# Plot X Mean vs Median
ax = axes[0]
ax.plot(global_agg['Time'], global_agg['X']['mean'], 'b-', lw=2, label='Mean X')
ax.plot(global_agg['Time'], global_agg['X']['median'], 'b--', lw=1.5, label='Median X')
ax.fill_between(global_agg['Time'], 
                global_agg['X']['mean'] - global_agg['X']['mean'].std(),
                global_agg['X']['mean'] + global_agg['X']['mean'].std(),
                color='b', alpha=0.1, label='Global Std Dev (not SEM)')
ax.set_ylabel('X Position (deg)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot Y Mean vs Median
ax = axes[1]
ax.plot(global_agg['Time'], global_agg['Y']['mean'], 'g-', lw=2, label='Mean Y')
ax.plot(global_agg['Time'], global_agg['Y']['median'], 'g--', lw=1.5, label='Median Y')
ax.set_ylabel('Y Position (deg)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot Sample Count (N)
ax = axes[2]
ax.plot(global_agg['Time'], global_agg['X']['count'], 'k-', lw=2, label='Valid Samples (N)')
ax.set_ylabel('Count (N)')
ax.set_ylim(0, global_agg['X']['count'].max() * 1.1)
ax.axhline(global_agg['X']['count'].max(), color='r', ls=':', label='Max Possible')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)')

# Events
for ax in axes:
    try:
        analyzer._add_event_lines(ax, ax.get_ylim())
    except: pass

plt.tight_layout()
save_path = os.path.join(out_dir, 'D01_Subj01_MeanMedian_Count.png')
print(f"Saving {save_path}")
plt.savefig(save_path, dpi=300)
plt.close()

# 2. Check for "Weird" Trials (Outliers)
# Identify trials with max deviation > 1.2 deg (since threshold is 1.5)
outlier_trials = []
for trial_id in traces_df['Trial'].unique():
    t_data = traces_df[traces_df['Trial'] == trial_id]
    if t_data['X'].abs().max() > 1.2 or t_data['Y'].abs().max() > 1.2:
        outlier_trials.append(trial_id)

print(f"\nFound {len(outlier_trials)} potential outlier trials (max dev > 1.2 deg)")
print(f"Outlier Trial IDs: {outlier_trials[:20]}...")

# Plot Top 5 Outliers
if outlier_trials:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Top Outlier Trials (Max Abs Dev > 1.2 deg)", fontsize=16)
    
    for i, trial in enumerate(outlier_trials[:5]):
        t_data = traces_df[traces_df['Trial'] == trial]
        axes[0].plot(t_data['Time'], t_data['X'], label=f"T{trial}")
        axes[1].plot(t_data['Time'], t_data['Y'], label=f"T{trial}")
        
    axes[0].set_ylabel('X (deg)')
    axes[1].set_ylabel('Y (deg)')
    axes[0].legend()
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'D01_Subj01_OutlierExamples.png')
    print(f"Saving {save_path}")
    plt.savefig(save_path)
    plt.close()

print("\n=== Done ===")
