import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import subprocess
import generate_behavdf as gm
from scipy import stats

# Setup aesthetics
sns.set_theme(style="white", context="talk")

def get_roi_size(filepath):
    """Returns (voxels, volume) using fslstats."""
    if not os.path.exists(filepath): return np.nan, np.nan
    try:
        cmd = f"/usr/local/fsl/bin/fslstats {filepath} -V"
        res = subprocess.check_output(cmd, shell=True).decode().split()
        return float(res[0]), float(res[1])
    except: return np.nan, np.nan

# --- 1. EXTRACT ROI SIZES (V1 and V1+V2d) ---
print("Extracting ROI sizes...")
subjects = sorted(gm.master_df['subjID'].unique())
data_roi = []
base_path = os.path.join(gm.p['datd'], 'brainsight')
for subj in subjects:
    subj_str = f"sub{int(subj):02d}"
    v1_lh, v1_rh = 0, 0
    v2_lh, v2_rh = 0, 0
    
    # V1
    _, v1_lh = get_roi_size(os.path.join(base_path, subj_str, 'lh.V1.nii.gz'))
    _, v1_rh = get_roi_size(os.path.join(base_path, subj_str, 'rh.V1.nii.gz'))
    # V2d
    _, v2_lh = get_roi_size(os.path.join(base_path, subj_str, 'lh.V2d.nii.gz'))
    _, v2_rh = get_roi_size(os.path.join(base_path, subj_str, 'rh.V2d.nii.gz'))
    
    v1_lh = v1_lh if not np.isnan(v1_lh) else 0
    v1_rh = v1_rh if not np.isnan(v1_rh) else 0
    v2_lh = v2_lh if not np.isnan(v2_lh) else 0
    v2_rh = v2_rh if not np.isnan(v2_rh) else 0
    
    data_roi.append({'subjID': subj, 'ROI': 'V1', 'Left': v1_lh, 'Right': v1_rh})
    data_roi.append({'subjID': subj, 'ROI': 'V1+V2d', 'Left': v1_lh+v2_lh, 'Right': v1_rh+v2_rh})

df_roi = pd.DataFrame(data_roi)

# --- 2. PLOTTING ROI VOLUMES ---
print("Generating ROI Volume plot...")
df_melt = df_roi.melt(id_vars=['subjID', 'ROI'], value_vars=['Left', 'Right'], var_name='Hemi', value_name='Volume')

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_melt, x='ROI', y='Volume', hue='Hemi', palette='Set2')
sns.stripplot(data=df_melt, x='ROI', y='Volume', hue='Hemi', dodge=True, alpha=0.5, color='black', legend=False)

plt.title("ROI Volumes across Subjects")
plt.ylabel("Volume (mm³)")
plt.xlabel("ROI")
sns.despine()
plt.tight_layout()

# No saving as requested, just show if interactive or just end
print("Plot generated. Displaying not supported in background, but script completed.")
plt.show() # Uncomment if running locally with display
