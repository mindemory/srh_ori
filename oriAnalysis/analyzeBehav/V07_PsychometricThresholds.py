import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- FILTERING FOR RADIAL ONLY ---
df = df[df['OriType'] == 'Radial'].copy()

# Define Hemifield based on target PF relative to stimulated hemisphere
def get_hemifield(row):
    hemi = row['hemisphereStimulated']
    pf = row['stimPF_label']
    if hemi == 'Left':
        return 'Right' if pf == 'inPF' else 'Left'
    else:
        return 'Left' if pf == 'inPF' else 'Right'

df['Hemifield'] = df.apply(get_hemifield, axis=1)
df['TMS_Status'] = df['TMS_Group'].map({'No TMS': 'NoTMS', 'TMS': 'TMS'})

# Diagnostic print
print("\n--- DATA DIAGNOSTICS ---")
print(f"Total trials post-cleaning: {len(df)}")
notms_radial = df[(df['TMS_Status'] == 'NoTMS')]
print(f"NoTMS Radial trials: {len(notms_radial)}")
if not notms_radial.empty:
    print(f"NoTMS Hemifield distribution:\n{notms_radial['Hemifield'].value_counts()}")
    print(f"NoTMS Subject distribution (first 5):\n{notms_radial['subjID'].value_counts().head()}")

# Sliding window aggregation (30° window to ensure density, 5° steps)
window_size = 30
step_size = 5
centers = np.arange(-90, 91, step_size)

sub_records = []
groups = df.groupby(['subjID', 'Hemifield', 'TMS_Status'])

for (subj, hemi, tms_stat), grp in groups:
    # Filter for valid trials
    valid_grp = grp.dropna(subset=['InitOffset', 'CW_Tendency', 'errorOri'])
    if valid_grp.empty: continue
    
    for c in centers:
        low = c - window_size/2
        high = c + window_size/2
        win_trials = valid_grp[(valid_grp['InitOffset'] >= low) & (valid_grp['InitOffset'] < high)]
        
        if not win_trials.empty:
            sub_records.append({
                'subjID': int(subj),
                'Hemifield': hemi,
                'TMS_Status': tms_stat,
                'Bin_Mid': c,
                'Mean_Tendency': win_trials['CW_Tendency'].mean(),
                'Mean_Error': win_trials['errorOri'].mean(),
                'N': len(win_trials)
            })

sub_stats_hemi = pd.DataFrame(sub_records)
print(f"Aggregated Data Shape: {sub_stats_hemi.shape}")
if not sub_stats_hemi.empty:
    print(f"Aggregated Status Counts:\n{sub_stats_hemi['TMS_Status'].value_counts()}")
else:
    print("WARNING: Aggregated dataframe is empty!")

# -------------------------------------------------------
# Visualization: Individual Subject Hemifield Baseline
# -------------------------------------------------------
sns.set_theme(style="white", context="paper")
output_dir = os.path.join(gm.p['datd'], 'Figures', 'motorcertainty')
os.makedirs(output_dir, exist_ok=True)

if sub_stats_hemi.empty:
    print("CRITICAL: Nothing to plot. Exiting.")
    sys.exit()

subjects = sorted(sub_stats_hemi['subjID'].unique())
n_subs = len(subjects)
cols = 4
rows = int(np.ceil(n_subs / cols))

fig, axes = plt.subplots(rows, cols, figsize=(18, 4*rows), sharex=True, sharey=True)
axes = axes.flatten()

color_map = {'Left': '#1f77b4', 'Right': '#d62728'}

for i, subj in enumerate(subjects):
    ax = axes[i]
    ax2 = ax.twinx()  # Right y-axis for Signed Error
    
    subj_data = sub_stats_hemi[(sub_stats_hemi['subjID'] == subj) & (sub_stats_hemi['TMS_Status'] == 'NoTMS')]
    
    for hemi in ['Left', 'Right']:
        plot_data = subj_data[subj_data['Hemifield'] == hemi].sort_values('Bin_Mid')
        if not plot_data.empty:
            label_tendency = hemi + " (CW)" if i == 0 else "_nolegend_"
            label_error = hemi + " (Err)" if i == 0 else "_nolegend_"
            
            # 1. Plot CW Tendency (Left Y-Axis)
            ax.plot(
                plot_data['Bin_Mid'], plot_data['Mean_Tendency'], 
                'o-', color=color_map[hemi], lw=2.0, markersize=5, alpha=0.9, label=label_tendency
            )
            
            # 2. Plot Signed Error (Right Y-Axis)
            ax2.plot(
                plot_data['Bin_Mid'], plot_data['Mean_Error'], 
                's--', color=color_map[hemi], lw=1.5, markersize=4, alpha=0.4, label=label_error
            )

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_title(f"Subject {subj}", fontsize=12, fontweight='bold')
    
    # Left Axis limits
    ax.set_ylim(-0.05, 1.05)
    
    # Right Axis limits - dynamic but centered at 0
    if not subj_data.empty and not subj_data['Mean_Error'].isna().all():
        max_err = subj_data['Mean_Error'].abs().max()
        # Add a 10% buffer, ensure minimum scale isn't too tiny
        max_err = max(max_err * 1.1, 10)
        ax2.set_ylim(-max_err, max_err)
    
    ax.set_xlim(-95, 95)    
    # Labels
    if i % cols == 0:
        ax.set_ylabel("CW Tendency")
    if (i + 1) % cols == 0 or i == n_subs - 1:
        ax2.set_ylabel("Signed Error (°)")
        
    if i >= (rows - 1) * cols:
        ax.set_xlabel("Init Offset (°)")

# Legend placement
handles1, labels1 = axes[0].get_legend_handles_labels()
handles2, labels2 = fig.axes[1].get_legend_handles_labels() 
fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05), frameon=True, fontsize=12)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
save_path = os.path.join(output_dir, 'Fig7_Individual_Radial_Hemi_Baseline.png')
print(f"\nSaving figure to {save_path}")
plt.savefig(save_path, dpi=300)
plt.close()

print("Done!")
