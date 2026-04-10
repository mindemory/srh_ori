"""
E01_gazeDiscovery.py - Demo script for gaze trace analysis.

Uses the GazeAnalyzer class to analyze gaze data for Subject 01, NoTMS condition.
Plots 50 randomly selected individual trials to visualize data variability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import generate_behavdf as gm
from GazeAnalyzer import GazeAnalyzer

# Initialize analyzer
datd_path = gm.p['datd']
analyzer = GazeAnalyzer(datd_path=datd_path, master_df=gm.master_df)

# Load gaze data for Subject 01
subj_id = 1
print(f"\n=== Loading gaze data for Subject {subj_id} ===")
gaze_data = analyzer.load_gaze_data(subj_id)
print(f"Total trials loaded: {len(gaze_data['X'])}")

# Get behavioral data and filter for NoTMS
behav_df = analyzer.get_behavioral_df(subj_id)
print(f"Behavioral trials: {len(behav_df)}")

# Filter for NoTMS only
filtered_df = analyzer.filter_trials(behav_df, TMScond=1)
print(f"NoTMS trials: {len(filtered_df)}")

# Extract traces
print("\n=== Extracting traces ===")
traces_df = analyzer.extract_traces(
    df=filtered_df,
    gaze_data=gaze_data,
    gaze_threshold=1.5,
    downsample=10,
    baseline_window=(-1.0, 0.0)
)
print(f"Extracted {len(traces_df)} samples from {traces_df['Trial'].nunique()} trials")

# Select 50 random trials
print("\n=== Selecting 50 random trials ===")
unique_trials = traces_df['Trial'].unique()
np.random.seed(42)  # For reproducibility
n_trials = min(50, len(unique_trials))
selected_trials = np.random.choice(unique_trials, size=n_trials, replace=False)
print(f"Selected trials: {sorted(selected_trials)}")

# Filter traces to only selected trials
selected_traces = traces_df[traces_df['Trial'].isin(selected_trials)].copy()
print(f"Samples for selected trials: {len(selected_traces)}")

# Verify data integrity
print("\n=== Data Integrity Check ===")
for col in ['X', 'Y', 'Pupil_Z']:
    n_nan = selected_traces[col].isna().sum()
    d_min = selected_traces[col].min() 
    d_max = selected_traces[col].max()
    print(f"{col}: NaNs={n_nan}, Range=[{d_min:.4f}, {d_max:.4f}]")

# Plot individual trials
print("\n=== Generating individual trial plots ===")
out_dir = os.path.join(datd_path, 'figures', 'gazeDecoding')
os.makedirs(out_dir, exist_ok=True)

# Create figure with 3 subplots (X, Y, Pupil)
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f"Subject {subj_id:02d} - NoTMS | 50 Random Individual Trials", fontsize=14)

# Color palette - use continuous map for many trials
colors = plt.cm.viridis(np.linspace(0, 1, n_trials))

metrics = [('X', 'X Position (deg)'), ('Y', 'Y Position (deg)'), ('Pupil_Z', 'Pupil (z-score)')]

for ax_idx, (metric, ylabel) in enumerate(metrics):
    ax = axes[ax_idx]
    
    # Autoscale Y axis
    ax.autoscale(enable=True, axis='y')
    
    for i, trial in enumerate(sorted(selected_trials)):
        trial_data = selected_traces[selected_traces['Trial'] == trial]
        ax.plot(trial_data['Time'], trial_data[metric], 
                color=colors[i], alpha=0.4, lw=1, label=f"Trial {trial}" if i < 10 else "")
    
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    
    # Add event lines
    try:
        analyzer._add_event_lines(ax, ax.get_ylim())
    except:
        pass
    
    # Only show legend for first few trials to avoid clutter, or omit
    if ax_idx == 0:
         ax.text(0.02, 0.95, f"Displaying {n_trials} trials", transform=ax.transAxes)

axes[-1].set_xlabel("Time (s)")

# Set xlim based on Response (Event 4) start
if 4 in analyzer.mean_event_times:
    resp_start = analyzer.mean_event_times[4]
    print(f"Mean Response Start: {resp_start:.4f}s")
    axes[-1].set_xlim(-1.0, resp_start)
else:
    print("Warning: Response event (4) not found in mean_event_times.")
    axes[-1].set_xlim(-1.0, 4.5) 

plt.tight_layout()
save_path = os.path.join(out_dir, 'E01_Subj01_NoTMS_50RandomTrials.png')
print(f"Saving to {save_path}")
plt.savefig(save_path, dpi=300)
plt.close()

print("\n=== Done ===")

