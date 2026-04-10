"""
E02_gazeAllSubjects.py - Group-level gaze trace analysis.

Analyzes gaze data for ALL subjects (NoTMS only).
- Computes mean X, Y, Pupil within each subject
- Then computes mean ± SEM across subjects
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import generate_behavdf as gm
from GazeAnalyzer import GazeAnalyzer

# Initialize analyzer
datd_path = gm.p['datd']
analyzer = GazeAnalyzer(datd_path=datd_path, master_df=gm.master_df)

# Get all subject IDs from behavioral data
all_subj_ids = gm.master_df['subjID'].unique()
print(f"\n=== Processing {len(all_subj_ids)} subjects: {sorted(all_subj_ids)} ===")

# Collect subject-level means
subject_means = []  # List of DataFrames, one per subject

for subj_id in sorted(all_subj_ids):
    print(f"\n--- Subject {subj_id} ---")
    
    try:
        # Load gaze data
        gaze_data = analyzer.load_gaze_data(subj_id)
        
        if len(gaze_data['X']) == 0:
            print(f"  No gaze data found, skipping.")
            continue
            
        # Get behavioral data and filter for NoTMS
        behav_df = analyzer.get_behavioral_df(subj_id)
        filtered_df = analyzer.filter_trials(behav_df, TMScond=1)
        
        if len(filtered_df) == 0:
            print(f"  No NoTMS trials, skipping.")
            continue
            
        print(f"  NoTMS trials: {len(filtered_df)}")
        
        # Extract traces
        traces_df = analyzer.extract_traces(
            df=filtered_df,
            gaze_data=gaze_data,
            gaze_threshold=1.5,
            downsample=10,
            baseline_window=(-1.0, 0.0)
        )
        
        if len(traces_df) == 0:
            print(f"  No valid traces, skipping.")
            continue
            
        print(f"  Extracted {len(traces_df)} samples")
        
        # Compute within-subject mean (mean across trials at each time point)
        subj_mean = traces_df.groupby(['Time', 'OriCat']).agg({
            'X': 'mean',
            'Y': 'mean',
            'Pupil_Z': 'mean'
        }).reset_index()
        
        subj_mean['SubjID'] = subj_id
        subject_means.append(subj_mean)
        
    except Exception as e:
        print(f"  Error processing subject {subj_id}: {e}")
        continue

print(f"\n=== Successfully processed {len(subject_means)} subjects ===")

# Combine all subject means
all_means_df = pd.concat(subject_means, ignore_index=True)
print(f"Total data points: {len(all_means_df)}")

# Compute across-subject mean and SEM
# Group by Time and OriCat, compute mean and SEM across subjects
def agg_across_subjects(df, col):
    agg = df.groupby(['Time', 'OriCat']).agg(
        mean=(col, 'mean'),
        sem=(col, lambda x: x.std() / np.sqrt(len(x)))
    ).reset_index()
    return agg

agg_x = agg_across_subjects(all_means_df, 'X')
agg_y = agg_across_subjects(all_means_df, 'Y')
agg_pup = agg_across_subjects(all_means_df, 'Pupil_Z')

aggregates = {'X': agg_x, 'Y': agg_y, 'Pupil_Z': agg_pup}

# Plot
print("\n=== Generating group-level plots ===")
out_dir = os.path.join(datd_path, 'figures', 'gazeDecoding')
os.makedirs(out_dir, exist_ok=True)

# Use last analyzer's event times (should be similar across subjects)
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

ori_palette = {30: '#1f77b4', 60: '#2ca02c', 120: '#ff7f0e', 150: '#d62728'}
ylims = {'X': (-0.5, 0.5), 'Y': (-0.5, 0.5), 'Pupil_Z': None}
columns = ['X', 'Y', 'Pupil_Z']
labels = ['X (deg)', 'Y (deg)', 'Pupil (Z)']

for ax, col, label in zip(axes, columns, labels):
    agg = aggregates[col]
    
    for ori in sorted(agg['OriCat'].unique()):
        color = ori_palette.get(ori, 'gray')
        ori_data = agg[agg['OriCat'] == ori].sort_values('Time')
        
        t = ori_data['Time'].values
        m = ori_data['mean'].values
        sem = ori_data['sem'].values
        
        ax.plot(t, m, color=color, linewidth=1.5, label=f'{ori}°')
        ax.fill_between(t, m - sem, m + sem, color=color, alpha=0.15)
    
    if ylims.get(col) is not None:
        ax.set_ylim(ylims[col])
    
    ax.set_ylabel(label)
    ax.set_title(f"Mean {col} Trace - All Subjects (NoTMS)")
    
    # Add event lines
    ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
    ax.text(0.02, ax.get_ylim()[1], 'S', fontsize=8, color='k', verticalalignment='top')

# Add legend
axes[0].legend(loc='upper right', fontsize='small', title='OriCat')

# Set x-axis limits
x_min = -1.3
x_max = 4.3
for ax in axes:
    ax.set_xlim(x_min, x_max)

axes[2].set_xlabel("Time from Sample Onset (s)")

# Add subtitle with N
fig.suptitle(f"Group-Level Gaze Traces (N={len(subject_means)} subjects, NoTMS only)", fontsize=14, y=0.995)

plt.tight_layout()
save_path = os.path.join(out_dir, 'E02_AllSubjects_NoTMS_Traces.png')
print(f"Saving to {save_path}")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print("\n=== Done ===")
