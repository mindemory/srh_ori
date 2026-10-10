"""
efield_v1_TMS_vs_Other.py

For each subject with a completed simulation:
  - Load lh/rh E.magn surface overlays
  - Map lh/rh V1 NIfTI masks onto the surface
  - Use hemisphereStimulated from gm.master_df (same source as V10_paperFigures)
  - TMS V1   = V1 of the directly stimulated hemisphere
  - Other V1 = V1 of the opposite hemisphere
  - Stimulated Left  → TMS V1 = lh.V1,  Other V1 = rh.V1
  - Stimulated Right → TMS V1 = rh.V1,  Other V1 = lh.V1
  - Compute mean E-field within each V1 per subject
  - Plot as paired boxplot with individual dots using V10 color scheme (Middle TMS colors)
"""

import nibabel as nib
from nilearn import surface
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import glob
from scipy import stats
import sys
sys.path.insert(0, '/Users/mrugank/Documents/localhyper/srh_ori/oriAnalysis/analyzeBehav')
import generate_behavdf as gm

# ---- Paths ----
SIMNIBS_DIR = '/d/DATD/datd/TMS_ori/SIMNIBS_output/'
V1_DIR = '/d/DATD/datd/TMS_ori/brainsight/'
OUTPUT_DIR = '/d/DATD/datd/TMS_ori/Figures/efield/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- V10 Color Scheme (Middle TMS = blue) ----
COL_BOX   = '#90CAF9'  # light blue box fill
COL_STRIP = '#1565C0'  # deep blue for dots and edges

# ---- Get per-subject hemisphere from master_df (same source as V10) ----
df_meta = gm.master_df[['subjID', 'hemisphereStimulated']].drop_duplicates()
hemi_map = {f"sub{int(r['subjID']):02d}": r['hemisphereStimulated'] for _, r in df_meta.iterrows()}

print("Hemisphere stimulation map:")
for k, v in sorted(hemi_map.items()):
    print(f"  {k}: {v}")

# ---- Collect E-field values per subject ----
results = []

subs_with_overlays = sorted(glob.glob(os.path.join(SIMNIBS_DIR, 'sub*/simstandard/subject_overlays/')))

for overlay_dir in subs_with_overlays:
    parts = overlay_dir.split('/')
    sub_id = [p for p in parts if p.startswith('sub') and p[3:].isdigit()][0]

    # Check V1 mask exists
    v1_lh_path = os.path.join(V1_DIR, sub_id, 'lh.V1.nii.gz')
    v1_rh_path = os.path.join(V1_DIR, sub_id, 'rh.V1.nii.gz')
    if not os.path.exists(v1_lh_path) or not os.path.exists(v1_rh_path):
        print(f"Skipping {sub_id}: V1 masks not found.")
        continue

    # Check hemisphere info
    if sub_id not in hemi_map:
        print(f"Skipping {sub_id}: not in master_df.")
        continue
    hemi_stim = hemi_map[sub_id]
    if hemi_stim not in ['Left', 'Right']:
        print(f"Skipping {sub_id}: hemisphere = '{hemi_stim}' (unclear).")
        continue

    # Find E.magn files
    try:
        lh_efield_path = glob.glob(os.path.join(overlay_dir, 'lh.*.central.E.magn'))[0]
        rh_efield_path = glob.glob(os.path.join(overlay_dir, 'rh.*.central.E.magn'))[0]
        lh_surf_path   = os.path.join(overlay_dir, 'lh.central')
        rh_surf_path   = os.path.join(overlay_dir, 'rh.central')
    except IndexError:
        print(f"Skipping {sub_id}: E.magn files not found.")
        continue

    print(f"Processing {sub_id} (stimulated: {hemi_stim})...")

    # Load surface geometry
    coords_lh, faces_lh = nib.freesurfer.read_geometry(lh_surf_path)
    coords_rh, faces_rh = nib.freesurfer.read_geometry(rh_surf_path)

    # Load E-field values
    efield_lh = nib.freesurfer.read_morph_data(lh_efield_path)
    efield_rh = nib.freesurfer.read_morph_data(rh_efield_path)

    # Map V1 masks to surface
    v1_lh_img = nib.load(v1_lh_path)
    v1_rh_img = nib.load(v1_rh_path)
    v1_lh_surf = surface.vol_to_surf(v1_lh_img, [coords_lh, faces_lh])
    v1_rh_surf = surface.vol_to_surf(v1_rh_img, [coords_rh, faces_rh])

    # Binarize V1 masks
    v1_lh_mask = v1_lh_surf > 0.1
    v1_rh_mask = v1_rh_surf > 0.1

    # Assign TMS vs Other based on directly stimulated hemisphere
    # TMS V1   = V1 of the stimulated hemi (more direct field)
    # Other V1 = V1 of the opposite hemi
    if hemi_stim == 'Left':
        tms_efield   = efield_lh[v1_lh_mask]
        other_efield = efield_rh[v1_rh_mask]
    else:  # Right
        tms_efield   = efield_rh[v1_rh_mask]
        other_efield = efield_lh[v1_lh_mask]

    if len(tms_efield) == 0 or len(other_efield) == 0:
        print(f"  Warning: empty V1 mask for {sub_id}. Skipping.")
        continue

    # Compute max E-field within each V1 per subject
    tms_val   = np.max(tms_efield)
    other_val = np.max(other_efield)

    print(f"  TMS V1 max E: {tms_val:.3f} V/m  |  Other V1 max E: {other_val:.3f} V/m")
    results.append({'sub_id': sub_id, 'TMS': tms_val, 'Other': other_val})

df = pd.DataFrame(results)
print(f"\nTotal subjects: {len(df)}")
print(df[['sub_id', 'TMS', 'Other']].to_string(index=False))

# ---- Plot ----
fig, ax = plt.subplots(figsize=(5, 6))

bw = 0.06
gap = 0.04
pos_tms   = 0.0
pos_other = bw + gap

rng = np.random.default_rng(42)

for pos, col, vals, label, al, sal in [
    (pos_tms,   COL_BOX, df['TMS'].values,   'TMS',   1.0, 0.7),
    (pos_other, COL_BOX, df['Other'].values, 'Other', 0.65, 0.35),
]:
    ax.boxplot(
        [vals], positions=[pos], widths=bw, patch_artist=True,
        showfliers=False, manage_ticks=False,
        boxprops=dict(facecolor=col, alpha=al, linewidth=0.8, edgecolor=COL_STRIP),
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color=COL_STRIP, alpha=al, linewidth=0.8),
        capprops=dict(color=COL_STRIP, alpha=al, linewidth=0.8),
    )
    jitter = rng.uniform(-bw * 0.28, bw * 0.28, len(vals))
    ax.scatter(pos + jitter, vals, color=COL_STRIP, alpha=sal, s=30, zorder=3)

# Connect paired dots with thin lines
for i in range(len(df)):
    ax.plot([pos_tms, pos_other], [df['TMS'].iloc[i], df['Other'].iloc[i]],
            color=COL_STRIP, alpha=0.2, linewidth=0.8, zorder=2)

# Stats: paired t-test and effect size (Cohen's d)
t_stat, p_val = stats.ttest_rel(df['TMS'].values, df['Other'].values)
diff = df['TMS'].values - df['Other'].values
cohen_d = np.mean(diff) / np.std(diff, ddof=1)
print(f"\nPaired t-test: t={t_stat:.3f}, p={p_val:.2e}")
print(f"Effect size (Cohen's d): {cohen_d:.3f}")

# Significance bar
ymax = max(df['TMS'].max(), df['Other'].max()) * 1.1
sig_y = ymax * 1.05
sig_str = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
ax.plot([pos_tms, pos_tms, pos_other, pos_other], [sig_y, sig_y*1.02, sig_y*1.02, sig_y], color='k', lw=1)
ax.text((pos_tms + pos_other) / 2, sig_y * 1.025, sig_str, ha='center', va='bottom', fontsize=12)

ax.set_xticks([pos_tms, pos_other])
ax.set_xticklabels(['TMS V1', 'Other V1'], fontsize=12)
ax.set_ylabel('Max E-field in V1 (V/m)', fontsize=12)
ax.set_title(f'V1 Max E-field: TMS Hemisphere vs Other\n(N={len(df)} subjects)', fontsize=11)
ax.set_xlim(pos_tms - 0.05, pos_other + 0.05)
ax.set_ylim(0, 2.0)

handles = [
    mpatches.Patch(facecolor=COL_BOX, alpha=1.0, edgecolor=COL_STRIP, lw=0.8, label='TMS Hemi'),
    mpatches.Patch(facecolor=COL_BOX, alpha=0.65, edgecolor=COL_STRIP, lw=0.8, label='Other Hemi'),
]
ax.legend(handles=handles, frameon=False, fontsize=9, loc='upper right')

import seaborn as sns
sns.despine(ax=ax, offset=5)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'efield_V1_TMS_vs_Other.svg')
plt.savefig(out_path, bbox_inches='tight')
print(f"\nFigure saved to: {out_path}")
