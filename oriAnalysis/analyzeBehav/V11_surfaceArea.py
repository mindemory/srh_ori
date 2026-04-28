import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import subprocess
import generate_behavdf as gm
from scipy import stats

# Setup aesthetics
sns.set_theme(style="whitegrid", palette="muted")
output_dir = os.path.join(gm.p['datd'], 'Figures', 'V11_SurfaceArea')
os.makedirs(output_dir, exist_ok=True)

# HARD-CODED PHOSPHENE AREAS (from V12_tempphosphene.py)
PHOSPHENE_AREAS = {
    1: 336374.74, 2: 167536.34, 3: 117065.06, 5: 251938.71, 6: 181427.12,
    7: 263543.54, 8: 48461.33, 9: 302323.34, 11: 109957.43, 12: 122747.67,
    14: 231613.78, 16: 315987.92, 17: 41137.00, 18: 263938.35, 19: 321687.83,
    20: 134379.74, 21: 99538.24, 22: 163113.10, 23: 31812.30, 24: 76675.30,
    25: 106969.62, 26: 76766.58, 28: 24483.74
}

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
    data_roi.append({'subjID': subj, 'ROI': 'V1V2d', 'Left': v1_lh+v2_lh, 'Right': v1_rh+v2_rh})

df_roi = pd.DataFrame(data_roi)

# --- 2. BEHAVIORAL PREPROCESSING ---
print("Preprocessing behavioral data...")
df_behav = gm.master_df.copy()
df_behav = df_behav[df_behav['LargeError'] == 0].copy()
df_behav.rename(columns={'TMScond_label': 'TMS', 'stimPF_label': 'Hemifield', 'targOriCategory': 'OriCat', 'hemisphereStimulated': 'HemiStim'}, inplace=True)

# Feature engineering
df_behav['OriCat45'] = df_behav['OriCat'].replace({30: 'Acute', 60: 'Acute', 120: 'Obtuse', 150: 'Obtuse'})
def classify_diag(row):
    hemi, in_pf, is_acute = row['HemiStim'], row['Hemifield'] == 'inPF', row['OriCat45'] == 'Acute'
    if hemi == 'Left': return 'Radial' if (in_pf and not is_acute) or (not in_pf and is_acute) else 'Tangential'
    else: return 'Radial' if (in_pf and is_acute) or (not in_pf and not is_acute) else 'Tangential'

df_behav['DiagType'] = df_behav.apply(classify_diag, axis=1)
df_behav['AbsError'] = np.abs(df_behav['errOri_scaled'])
df_behav['BiasTowardsDiagonal'] = df_behav['errOri_scaled']
df_behav.loc[df_behav['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1

# --- 3. SIZE MAPPING & PHOSPHENE INTEGRATION ---
print("Merging data...")
# Get phosphene size
df_phos = pd.DataFrame(list(PHOSPHENE_AREAS.items()), columns=['subjID', 'PhospheneArea'])

# Pivot ROI data
df_roi_piv = df_roi.pivot(index='subjID', columns='ROI', values=['Left', 'Right'])
df_roi_piv.columns = [f'{c[0]}_{c[1]}' for c in df_roi_piv.columns]
df_roi_piv = df_roi_piv.reset_index()

# Map ROI to HemiStim
df_hemistim = df_behav.groupby('subjID')['HemiStim'].first().reset_index()
df_final_struct = pd.merge(df_roi_piv, df_hemistim, on='subjID')
df_final_struct = pd.merge(df_final_struct, df_phos, on='subjID')

# Assign inPF/outPF ROI Volumes
for roi in ['V1', 'V1V2d']:
    df_final_struct[f'{roi}_inPF'] = np.where(df_final_struct['HemiStim'] == 'Right', df_final_struct[f'Left_{roi}'], df_final_struct[f'Right_{roi}'])
    df_final_struct[f'{roi}_outPF'] = np.where(df_final_struct['HemiStim'] == 'Right', df_final_struct[f'Right_{roi}'], df_final_struct[f'Left_{roi}'])

# --- 4. PRECISION & RADIAL SENSITIVITY ---
import modeller as md
mixture_stats = []
for tms in ['No TMS', 'Early TMS', 'Middle TMS']:
    for hemi in ['inPF', 'outPF']:
        for subj in subjects:
            mask = (df_behav['TMS'] == tms) & (df_behav['Hemifield'] == hemi) & (df_behav['subjID'] == subj)
            subset = df_behav[mask]
            if len(subset) < 10: continue
            try:
                res = md.fit_model(subset['errOri_scaled'].values, 'mixtureModel')
                mixture_stats.append({'subjID': subj, 'TMS': tms, 'Hemifield': hemi, 'Precision': res[1]})
            except: pass
df_precision = pd.DataFrame(mixture_stats)

df_basic = df_behav.groupby(['subjID', 'TMS', 'Hemifield'], observed=False)[['AbsError', 'BiasTowardsDiagonal']].mean().reset_index()
df_diag_agg = df_behav.groupby(['subjID', 'TMS', 'Hemifield', 'DiagType'], observed=False)['AbsError'].mean().reset_index()

rsm_list = []
for subj in subjects:
    for tms in ['No TMS', 'Early TMS', 'Middle TMS']:
        for hemi in ['inPF', 'outPF']:
            sub = df_diag_agg[(df_diag_agg['subjID'] == subj) & (df_diag_agg['TMS'] == tms) & (df_diag_agg['Hemifield'] == hemi)]
            if len(sub) < 2: continue
            rad = sub[sub['DiagType'] == 'Radial']['AbsError'].values[0]
            tan = sub[sub['DiagType'] == 'Tangential']['AbsError'].values[0]
            rsm = (rad - tan) / (rad + tan + 1e-12)
            rsm_list.append({'subjID': subj, 'TMS': tms, 'Hemifield': hemi, 'RadialIndex': rsm})
df_rsm = pd.DataFrame(rsm_list)

# --- 5. FIGURE 5: ROI Volumes vs Phosphene Area ---
print("Generating Figure 5...")
sns.set_style("ticks")
fig5, axes5 = plt.subplots(2, 2, figsize=(14, 12))
roi_cols = [('V1_inPF', 'V1 inPF'), ('V1_outPF', 'V1 outPF'), ('V1V2d_inPF', 'V1+V2d inPF'), ('V1V2d_outPF', 'V1+V2d outPF')]
for i, (col, label) in enumerate(roi_cols):
    ax = axes5[i//2, i%2]
    sns.regplot(data=df_final_struct, x='PhospheneArea', y=col, ax=ax, color='teal', scatter_kws={'alpha':0.6})
    r, p = stats.spearmanr(df_final_struct['PhospheneArea'], df_final_struct[col])
    ax.text(0.05, 0.9, f"r={r:.2f}, p={p:.3f}", transform=ax.transAxes, fontweight='bold')
    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.set_xlabel("Phosphene Area (px²)", fontsize=12)
    ax.set_ylabel("Volume (mm³)", fontsize=12)
    sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure5_Structural_vs_PhospheneSize.png'), dpi=300)

# --- 6. FIGURE 6: Behavioral Contrasts vs Phosphene Area ---
print("Generating Figure 6...")
# Merge behavior and structure
df_piv_abs = df_basic.pivot_table(index=['subjID', 'TMS'], columns='Hemifield', values='AbsError', observed=False).reset_index()
df_piv_bias = df_basic.pivot_table(index=['subjID', 'TMS'], columns='Hemifield', values='BiasTowardsDiagonal', observed=False).reset_index()
df_piv_prec = df_precision.pivot_table(index=['subjID', 'TMS'], columns='Hemifield', values='Precision', observed=False).reset_index()
df_piv_rsm = df_rsm.pivot_table(index=['subjID', 'TMS'], columns='Hemifield', values='RadialIndex', observed=False).reset_index()

for df_p, met in zip([df_piv_abs, df_piv_bias, df_piv_prec, df_piv_rsm], ['AbsError', 'Bias', 'Precision', 'RadialIndex']):
    df_p[f'{met}_Diff'] = df_p['inPF'] - df_p['outPF']

df_behav_final = pd.merge(df_piv_abs[['subjID', 'TMS', 'AbsError_Diff']], df_piv_bias[['subjID', 'TMS', 'Bias_Diff']], on=['subjID', 'TMS'])
df_behav_final = pd.merge(df_behav_final, df_piv_prec[['subjID', 'TMS', 'Precision_Diff']], on=['subjID', 'TMS'])
df_behav_final = pd.merge(df_behav_final, df_piv_rsm[['subjID', 'TMS', 'RadialIndex_Diff']], on=['subjID', 'TMS'])
df_behav_final = pd.merge(df_behav_final, df_phos, on='subjID')

fig6, axes6 = plt.subplots(2, 2, figsize=(14, 12))
met_cols = [('AbsError_Diff', 'Δ Absolute Error (in-out)'), ('Bias_Diff', 'Δ Diagonal Bias (in-out)'), ('Precision_Diff', 'Δ Precision (in-out)'), ('RadialIndex_Diff', 'Δ Radial Sensitivity (in-out)')]
tms_colors = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}

for i, (col, label) in enumerate(met_cols):
    ax = axes6[i//2, i%2]
    for tms in ['No TMS', 'Early TMS', 'Middle TMS']:
        sub = df_behav_final[df_behav_final['TMS'] == tms].dropna()
        sns.regplot(data=sub, x='PhospheneArea', y=col, ax=ax, color=tms_colors[tms], label=tms, scatter_kws={'alpha':0.4, 's':20})
        r, p = stats.spearmanr(sub['PhospheneArea'], sub[col])
        y_pos = 0.95 - (['No TMS', 'Early TMS', 'Middle TMS'].index(tms) * 0.05)
        ax.text(0.05, y_pos, f"{tms}: r={r:.2f}, p={p:.3f}", transform=ax.transAxes, color=tms_colors[tms], fontsize=9)
    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.set_xlabel("Phosphene Area (px²)", fontsize=12)
    ax.set_ylabel("Δ Performance Metric", fontsize=12)
    sns.despine(ax=ax)
    if i == 0: ax.legend(frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure6_Behavioral_vs_PhospheneSize.png'), dpi=300)
print("\nFigures 5 and 6 generated successfully with clean aesthetics.")
