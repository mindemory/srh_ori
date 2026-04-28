import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import generate_behavdf as gm
from scipy import stats
import warnings
import os
import modeller as md

warnings.filterwarnings('ignore')
sns.set_theme(style="white", context="notebook")
# Output Dir
output_dir = os.path.join(gm.p['datd'], 'Figures', 'paperFigures')
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
df = gm.master_df.copy()
df.reset_index(drop=True, inplace=True)

# User Request (V3 Refinement): Remove LargeError trials
print(f"Total trials before filtering: {len(df)}")
df = df[df['LargeError'] == 0].copy()
print(f"Total trials after dropping LargeError=1: {len(df)}")

# --- Feature Engineering ---
df.rename(columns={
    'TMScond_label': 'TMS',        
    'stimPF_label': 'Hemifield',   
    'targOriCategory': 'OriCat',
    'hemisphereStimulated': 'HemiStim'
}, inplace=True)

# Metrics
df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1
df['AbsError'] = np.abs(df['errOri_scaled'])
df['AbsInitOffset'] = np.abs(df['initOffset'])

# --- GROUPING LOGIC ---
# TMS Group: No TMS vs TMS (Early/Middle combined)
df['TMS_Group'] = df['TMS'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
# Enforce order
df['TMS_Group'] = pd.Categorical(df['TMS_Group'], categories=['No TMS', 'TMS'], ordered=True)

# Offset Group: Small (0-25) vs Large (45-90). Intermediate (25-45) are ignored.
df['OffsetGroup'] = pd.cut(df['AbsInitOffset'], bins=[0, 25.1, 44.9, 90.1], labels=['Small Offset', 'Medium Offset', 'Large Offset'], include_lowest=True)
df['OffsetGroup'] = df['OffsetGroup'].cat.remove_categories(['Medium Offset'])
df['OffsetGroup3'] = pd.cut(df['AbsInitOffset'], bins=[0, 30, 60, 90], labels=['Small Offset', 'Medium Offset', 'Large Offset'], include_lowest=True)

# Type Conversion for others
df['Hemifield'] = pd.Categorical(df['Hemifield'], categories=['outPF', 'inPF'], ordered=True)
df['OriCat'] = df['OriCat'].astype(int)

# Ori Group: 30/150 vs 60/120
df['OriGroup'] = df['OriCat'].replace({
    30: '30_150', 150: '30_150',
    60: '60_120', 120: '60_120'
})

# OriCat45: 30+60 (diagonal ~45°) vs 120+150 (diagonal ~135°)
df['OriCat45'] = df['OriCat'].replace({
    30: 'Acute (30+60)', 60: 'Acute (30+60)',
    120: 'Obtuse (120+150)', 150: 'Obtuse (120+150)'
})

# DiagType: Radial vs Tangential
def classify_diag(row):
    hemi_stim = row['HemiStim']
    in_pf = row['Hemifield'] == 'inPF'
    is_acute = row['OriCat45'] == 'Acute (30+60)'
    
    if hemi_stim == 'Left':
        return 'Radial' if (in_pf and not is_acute) or (not in_pf and is_acute) else 'Tangential'
    else: # Right TMS
        return 'Radial' if (in_pf and is_acute) or (not in_pf and not is_acute) else 'Tangential'

df['DiagType'] = df.apply(classify_diag, axis=1)

# --- ORIENTATION BINNING ---
# Bins: 0-180 in 5 deg steps
ori_bins = np.arange(0, 185, 5)
df['OriBin'] = pd.cut(df['targOriActual'], bins=ori_bins, include_lowest=True)
df['OriBinCenter'] = df['OriBin'].apply(lambda x: x.mid).astype(float)


# --- Reusable Plotting Functions (Updated for 2 Cols) ---

def run_permutation_test(vals, n_perms=10000):
    """One-sample sign-flipping permutation test against 0."""
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0: return 1.0
    
    observed_mean = np.mean(vals)
    null_dist = []
    for _ in range(n_perms):
        # Flip signs randomly
        flipped = vals * np.random.choice([-1, 1], size=len(vals))
        null_dist.append(np.mean(flipped))
    
    null_dist = np.array(null_dist)
    p_val = np.mean(np.abs(null_dist) >= np.abs(observed_mean))
    return p_val

def get_sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

def plot_contrast_grid(data, contrast_cols, fig_title, filename, bin_col=None, bins=None):
    """
    Generic function to plot contrast indices (A-B)/(A+B).
    For Diagonal Bias, we use Abs(bias) before normalization as requested.
    """
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
        ('Precision', 'Precision (K)')
    ]
    
    tms_groups = ['No TMS', 'Early TMS', 'Middle TMS']
    A_val, B_val = contrast_cols
    
    if A_val in ['inPF', 'outPF']:
        pivot_col = 'Hemifield'
    else:
        pivot_col = 'DiagType'

    # Pre-calculate Subject Means
    group_cols = ['subjID', 'TMS', pivot_col]
    if bin_col: group_cols.append(bin_col)
    
    df_subj = data.groupby(group_cols, observed=False)[['AbsError', 'RT', 'BiasTowardsDiagonal']].mean().reset_index()
    
    # Mixture Model fitting
    print("  Fitting Mixture Models...")
    mixture_stats = []
    subjects = data['subjID'].unique()
    for tms in tms_groups:
        for val in [A_val, B_val]:
            search_bins = bins if bins is not None else [None]
            for b in search_bins:
                for subj in subjects:
                    mask = (data['TMS'] == tms) & (data[pivot_col] == val) & (data['subjID'] == subj)
                    if bin_col and b is not None: mask &= (data[bin_col] == b)
                    subset = data[mask]
                    if len(subset) < 10: continue
                    try:
                        res = md.fit_model(subset['errOri_scaled'].dropna().values, 'mixtureModel')
                        item = {'TMS': tms, pivot_col: val, 'subjID': subj, 'Precision': res[1]}
                        if bin_col: item[bin_col] = b
                        mixture_stats.append(item)
                    except: pass
    
    df_mix = pd.DataFrame(mixture_stats)
    df_all = pd.merge(df_subj, df_mix, on=group_cols, how='outer') if not df_mix.empty else df_subj

    num_bins = len(bins) if bins is not None else 1
    n_rows, n_cols = 2, 2 * num_bins
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 10), squeeze=False)
    fig.suptitle(fig_title, fontsize=16)

    for i_met, (met, label) in enumerate(metrics):
        if met not in df_all.columns: continue
        row_base, col_base = (i_met // 2), (i_met % 2)
        
        for c_bin in range(num_bins):
            ax = axes[row_base, col_base + 2*c_bin]
            b_val = bins[c_bin] if bins is not None else None
            
            subset_all = df_all
            if bin_col: subset_all = subset_all[subset_all[bin_col] == b_val]
            
            piv = subset_all.pivot_table(index=['subjID', 'TMS'], columns=pivot_col, values=met, observed=False).reset_index()
            if A_val not in piv.columns or B_val not in piv.columns: continue
                
            # ABS(BIAS) logic as requested
            if met == 'BiasTowardsDiagonal':
                a_data, b_data = np.abs(piv[A_val]), np.abs(piv[B_val])
            else:
                a_data, b_data = piv[A_val], piv[B_val]
                
            piv['ContrastIndex'] = (a_data - b_data) / (a_data + b_data + 1e-12)
            piv = piv.dropna(subset=['ContrastIndex'])
            
            # Fixed alignment: no dodge, consistent palettes
            sns.boxplot(
                data=piv, x='TMS', y='ContrastIndex',
                palette={'No TMS': '#E0E0E0', 'Early TMS': '#FFCDD2', 'Middle TMS': '#BBDEFB'},
                order=tms_groups, ax=ax, showfliers=False
            )
            sns.stripplot(
                data=piv, x='TMS', y='ContrastIndex',
                palette={'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'},
                order=tms_groups, ax=ax, alpha=0.5, size=4, jitter=True, dodge=False
            )
            
            ax.axhline(0, color='k', lw=1, ls='--')
            
            for i_tms, tms in enumerate(tms_groups):
                vals = piv[piv['TMS'] == tms]['ContrastIndex'].values
                if len(vals) > 5:
                    p_val = run_permutation_test(vals)
                    sig = get_sig_label(p_val)
                    y_max = piv['ContrastIndex'].max()
                    y_pos = y_max + (piv['ContrastIndex'].max() - piv['ContrastIndex'].min()) * 0.15
                    ax.text(i_tms, y_pos, f"{sig}\np={p_val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

            ax.set_title(f"{label}" + (f"\n({b_val})" if b_val else ""), fontsize=12)
            ax.set_ylabel("Contrast Index", fontsize=10)
            ax.set_xlabel("")
            ax.set_xticklabels(['No', 'Early', 'Middle'], fontsize=9)

    sns.despine()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()
def plot_scatter_in_vs_out(data, fig_title, filename, custom_limits=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
        ('Precision', 'Precision (K)')
    ]
    
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']
    
    # 1. Subject Means for regular metrics
    df_means = data.groupby(['subjID', 'TMS', 'Hemifield'], observed=False)[['AbsError', 'RT', 'BiasTowardsDiagonal']].mean().reset_index()
    
    # 2. Precision from mixture models
    print("  Fitting Mixture Models (Scatter)...")
    mixture_stats = []
    hemis = ['inPF', 'outPF']
    subjects = data['subjID'].unique()
    
    for tms in tms_conds:
        for hemi in hemis:
            for subj in subjects:
                subset = data[
                    (data['TMS'] == tms) & 
                    (data['Hemifield'] == hemi) &
                    (data['subjID'] == subj)
                ]
                if len(subset) < 10: continue
                angles = subset['errOri_scaled'].dropna().values
                try:
                    res = md.fit_model(angles, 'mixtureModel')
                    g, K = res
                    if g < 0 or g > 1 or K < 0 or K > 50: g, K = np.nan, np.nan
                except:
                    g, K = np.nan, np.nan
                
                if not np.isnan(K):
                    mixture_stats.append({
                        'TMS': tms, 'Hemifield': hemi, 'subjID': subj,
                        'Precision': K
                    })
    df_mix = pd.DataFrame(mixture_stats)
    
    # Merge
    if not df_mix.empty:
        df_all = pd.merge(df_means, df_mix, on=['subjID', 'TMS', 'Hemifield'], how='outer')
    else:
        df_all = df_means
        
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    fig.suptitle(fig_title, fontsize=16)
    
    palette = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}
    
    for i, (met, label) in enumerate(metrics):
        ax = axes[i]
        
        if met not in df_all.columns: continue
        
        piv = df_all.pivot_table(index=['subjID', 'TMS'], columns='Hemifield', values=met, observed=False).reset_index()
        piv.columns.name = None
        
        if 'inPF' not in piv.columns or 'outPF' not in piv.columns:
            continue
            
        piv = piv.dropna(subset=['inPF', 'outPF'])
        if piv.empty: continue
        
        for tms in tms_conds:
            sub = piv[piv['TMS'] == tms]
            if sub.empty: continue
            
            # Individual dots
            ax.scatter(sub['outPF'], sub['inPF'], color=palette[tms], alpha=0.6, s=20)
            
            # Mean crosshair
            mean_out = sub['outPF'].mean()
            mean_in = sub['inPF'].mean()
            sem_out = sub['outPF'].sem()
            sem_in = sub['inPF'].sem()
            
            ax.errorbar(mean_out, mean_in, xerr=sem_out, yerr=sem_in, 
                        fmt='o', color=palette[tms], markersize=12, 
                        markeredgecolor='white', label=tms)
            
        # Identity line
        all_vals = piv[['inPF', 'outPF']].values.flatten()
        if len(all_vals) == 0: continue
        
        min_v = np.nanmin(all_vals)
        max_v = np.nanmax(all_vals)
        margin = (max_v - min_v) * 0.1
        min_val = min_v - margin
        max_val = max_v + margin
        
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, zorder=0)
        
        ax.set_aspect('equal', 'box')
        
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        ax.set_title(label)
        ax.set_xlabel(f"outPF")
        ax.set_ylabel(f"inPF")
        
        if i == 0:
            ax.legend(title='TMS Group', loc='best')

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_scatter_diag_contrast_in_vs_out(data, fig_title, filename):
    """
    Plots (Radial - Tangential) / (Radial + Tangential) for inPF vs outPF.
    Each point is a subject, colored by TMS condition.
    """
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
        ('Precision', 'Precision (K)')
    ]
    
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']
    
    # 1. Subject Means
    df_means = data.groupby(['subjID', 'TMS', 'Hemifield', 'DiagType'], observed=False)[['AbsError', 'RT', 'BiasTowardsDiagonal']].mean().reset_index()
    
    # 2. Mixture Models
    print("  Fitting Mixture Models (Scatter)...")
    mixture_stats = []
    subjects = data['subjID'].unique()
    for tms in tms_conds:
        for hemi in ['inPF', 'outPF']:
            for diag in ['Radial', 'Tangential']:
                for subj in subjects:
                    mask = (data['TMS'] == tms) & (data['Hemifield'] == hemi) & (data['DiagType'] == diag) & (data['subjID'] == subj)
                    subset = data[mask]
                    if len(subset) < 10: continue
                    try:
                        res = md.fit_model(subset['errOri_scaled'].dropna().values, 'mixtureModel')
                        mixture_stats.append({'TMS': tms, 'Hemifield': hemi, 'DiagType': diag, 'subjID': subj, 'Precision': res[1]})
                    except: pass
    df_mix = pd.DataFrame(mixture_stats)
    df_all = pd.merge(df_means, df_mix, on=['subjID', 'TMS', 'Hemifield', 'DiagType'], how='outer') if not df_mix.empty else df_means
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    fig.suptitle(fig_title, fontsize=16)
    palette = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}
    
    for i, (met, label) in enumerate(metrics):
        ax = axes[i]
        if met not in df_all.columns: continue
        
        # Calculate Diagonal Contrast Index for each Subject/TMS/Hemi
        piv_diag = df_all.pivot_table(index=['subjID', 'TMS', 'Hemifield'], columns='DiagType', values=met, observed=False).reset_index()
        if 'Radial' not in piv_diag.columns or 'Tangential' not in piv_diag.columns: continue
        
        if met == 'BiasTowardsDiagonal':
            a_val, b_val = np.abs(piv_diag['Radial']), np.abs(piv_diag['Tangential'])
        else:
            a_val, b_val = piv_diag['Radial'], piv_diag['Tangential']
            
        piv_diag['DiagContrast'] = (a_val - b_val) / (a_val + b_val + 1e-12)
        piv_diag = piv_diag.dropna(subset=['DiagContrast'])
        
        # Now pivot Hemifield so inPF and outPF are columns
        piv_hemi = piv_diag.pivot_table(index=['subjID', 'TMS'], columns='Hemifield', values='DiagContrast', observed=False).reset_index()
        if 'inPF' not in piv_hemi.columns or 'outPF' not in piv_hemi.columns: continue
        
        piv_hemi = piv_hemi.dropna(subset=['inPF', 'outPF'])
        
        for tms in tms_conds:
            sub = piv_hemi[piv_hemi['TMS'] == tms]
            if sub.empty: continue
            
            ax.scatter(sub['outPF'], sub['inPF'], color=palette[tms], alpha=0.6, s=30, edgecolor='white', linewidth=0.5)
            
            # Mean crosshair
            m_out, m_in = sub['outPF'].mean(), sub['inPF'].mean()
            s_out, s_in = sub['outPF'].sem(), sub['inPF'].sem()
            ax.errorbar(m_out, m_in, xerr=s_out, yerr=s_in, fmt='o', color=palette[tms], markersize=12, markeredgecolor='white', label=tms)
            
        # Identity line and Zero lines
        all_vals = piv_hemi[['inPF', 'outPF']].values.flatten()
        if len(all_vals) > 0:
            limit = np.max(np.abs(all_vals)) * 1.2
            ax.plot([-limit, limit], [-limit, limit], 'k--', alpha=0.3)
            ax.axhline(0, color='k', alpha=0.1, lw=1)
            ax.axvline(0, color='k', alpha=0.1, lw=1)
            ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
            
        ax.set_aspect('equal', 'box')
        ax.set_title(f"Diag Contrast Index: {label}")
        ax.set_xlabel("outPF (Radial vs Tangential)")
        ax.set_ylabel("inPF (Radial vs Tangential)")
        if i == 1: ax.legend(title='TMS Group', loc='best', fontsize='small')

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

hemi_palette = {'inPF': '#e91e63', 'outPF': '#ff9800'}
ori_palette = {30: '#1f77b4', 60: '#2ca02c', 120: '#ff7f0e', 150: '#d62728'}
ori_group_palette = {'30_150': '#009688', '60_120': '#9c27b0'}
ori_cat45_palette = {'Acute (30+60)': '#3f51b5', 'Obtuse (120+150)': '#e91e63'}
diag_palette = {'Radial': '#E65100', 'Tangential': '#00B8D4'}

# Fig 1: Hemifield Contrast Index (inPF vs outPF)
plot_contrast_grid(
    df, contrast_cols=('inPF', 'outPF'),
    fig_title="1. Selective TMS Effect: (inPF - outPF) / (inPF + outPF)",
    filename="Fig1_ContrastIndex_Hemifield.png"
)

# Fig 1A: Scatterplot (Keep as is, it already shows individual points and means)
plot_scatter_in_vs_out(
    df,
    fig_title="1A. Individual Performance: outPF vs inPF",
    filename="Fig1A_Scatter_in_vs_out.png"
)

# Fig 2: Diagonal Contrast Index (Radial vs Tangential)
plot_contrast_grid(
    df, contrast_cols=('Radial', 'Tangential'),
    fig_title="2. Diagonal Selective Effect: (Radial - Tangential) / (Radial + Tangential)",
    filename="Fig2_ContrastIndex_DiagType.png"
)

# Fig 2A: Diagonal Contrast Scatter (inPF vs outPF)
plot_scatter_diag_contrast_in_vs_out(
    df,
    fig_title="2A. Diagonal Contrast Index: outPF vs inPF",
    filename="Fig2A_Scatter_diag_contrast_in_vs_out.png"
)

# Fig 3: Diagonal Contrast Index Split by Offset
plot_contrast_grid(
    df, contrast_cols=('Radial', 'Tangential'),
    fig_title="3. Diagonal Selective Effect | Split by Offset",
    filename="Fig3_ContrastIndex_DiagType_SplitOffset.png",
    bin_col='OffsetGroup',
    bins=['Small Offset', 'Large Offset']
)

# Fig 3A/3B: Diagonal Contrast Scatter (inPF vs outPF)
plot_scatter_diag_contrast_in_vs_out(
    df[df['OffsetGroup'] == 'Small Offset'],
    fig_title="3A. Diagonal Contrast Index: outPF vs inPF (Small Offset)",
    filename="Fig3A_Scatter_diag_contrast_in_vs_out_SmallOffset.png"
)

plot_scatter_diag_contrast_in_vs_out(
    df[df['OffsetGroup'] == 'Large Offset'],
    fig_title="3B. Diagonal Contrast Index: outPF vs inPF (Large Offset)",
    filename="Fig3B_Scatter_diag_contrast_in_vs_out_LargeOffset.png"
)

# ---------------------------------------------------------------------------
# Fig 4: Model RDMs and RSA Boxplots
# ---------------------------------------------------------------------------
def get_upper_tri(mat):
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]

CENTERS_4CAT = np.array([30, 60, 120, 150])

def build_model_rdms(x_sorted):
    x_cat   = CENTERS_4CAT[np.argmin(np.abs(CENTERS_4CAT - x_sorted[:, None]), axis=1)]
    x_broad = np.where(x_sorted < 90, 45, 135)
    H1 = get_upper_tri(np.abs(x_sorted[:, None] - x_sorted[None, :]))
    H2 = get_upper_tri(np.abs(x_cat[:, None]    - x_cat[None, :]))
    H3 = get_upper_tri(np.abs(x_broad[:, None]  - x_broad[None, :]))
    return H1, H2, H3

def compute_partial_rsa_with_perms(x_ori, y_resp, min_pairs=20):
    if len(x_ori) < 10: return None
    sort_idx = np.argsort(x_ori)
    x_s = x_ori[sort_idx]
    y_s = y_resp[sort_idx]
    H1, H2, H3 = build_model_rdms(x_s)
    
    # Compute Partial RSA
    # Target data RDM
    data_vec = get_upper_tri(np.abs(y_s[:, None] - y_s[None, :]))
    ry = stats.rankdata(data_vec)
    ry_cent = ry - ry.mean()
    ry_norm = np.linalg.norm(ry_cent) + 1e-12
    
    rh1 = stats.rankdata(H1)
    rh2 = stats.rankdata(H2)
    rh3 = stats.rankdata(H3)
    
    ones = np.ones(len(rh1))
    
    # Check for variance in models
    std2 = np.std(H2)
    std3 = np.std(H3)
    
    # Default zeros
    r1, r2, r3 = 0, 0, 0

    if std3 < 1e-8:
        # H3 is constant
        r3 = 0
        if std2 < 1e-8:
            # H2 and H3 are constant, just compute raw r1
            rh1_cent = rh1 - rh1.mean()
            r1 = np.dot(ry_cent, rh1_cent) / (ry_norm * (np.linalg.norm(rh1_cent) + 1e-12))
            r2 = 0
        else:
            # H3 constant, partial r2 | H3 is just raw r2
            rh2_cent = rh2 - rh2.mean()
            r2 = np.dot(ry_cent, rh2_cent) / (ry_norm * (np.linalg.norm(rh2_cent) + 1e-12))
            # Partial r1 | H2
            X2 = np.column_stack([ones, rh2])
            res_x1 = rh1 - X2 @ (np.linalg.pinv(X2) @ rh1)
            res_y1 = ry - X2 @ (np.linalg.pinv(X2) @ ry)
            r1 = np.dot(res_x1 - res_x1.mean(), res_y1 - res_y1.mean()) / (np.linalg.norm(res_x1 - res_x1.mean()) * np.linalg.norm(res_y1 - res_y1.mean()) + 1e-12)
    else:
        # H3 has variance
        r3 = np.dot(ry_cent, rh3 - rh3.mean()) / (ry_norm * (np.linalg.norm(rh3 - rh3.mean()) + 1e-12))
        if std2 < 1e-8:
            r2 = 0
            # Partial r1 | H3
            X = np.column_stack([ones, rh3])
            res_x1 = rh1 - X @ (np.linalg.pinv(X) @ rh1)
            res_y = ry - X @ (np.linalg.pinv(X) @ ry)
            r1 = np.dot(res_x1 - res_x1.mean(), res_y - res_y.mean()) / (np.linalg.norm(res_x1 - res_x1.mean()) * np.linalg.norm(res_y - res_y.mean()) + 1e-12)
        else:
            # Standard 3-model partial RSA
            X2 = np.column_stack([ones, rh3])
            res_x2 = rh2 - X2 @ (np.linalg.pinv(X2) @ rh2)
            res_y2 = ry - X2 @ (np.linalg.pinv(X2) @ ry)
            r2 = np.dot(res_x2 - res_x2.mean(), res_y2 - res_y2.mean()) / (np.linalg.norm(res_x2 - res_x2.mean()) * np.linalg.norm(res_y2 - res_y2.mean()) + 1e-12)
            
            X3 = np.column_stack([ones, rh2, rh3])
            res_x1 = rh1 - X3 @ (np.linalg.pinv(X3) @ rh1)
            res_y3 = ry - X3 @ (np.linalg.pinv(X3) @ ry)
            r1 = np.dot(res_x1 - res_x1.mean(), res_y3 - res_y3.mean()) / (np.linalg.norm(res_x1 - res_x1.mean()) * np.linalg.norm(res_y3 - res_y3.mean()) + 1e-12)
    
    return {'r3': r3, 'r2': r2, 'r1': r1}

def plot_model_rdms_fig4(data, fig_title, filename):
    print(f"Generating Figure: {fig_title}...")
    x_plot = np.linspace(0, 179, 180)
    x_cat = CENTERS_4CAT[np.argmin(np.abs(CENTERS_4CAT - x_plot[:, None]), axis=1)]
    x_broad = np.where(x_plot < 90, 45, 135)
    
    H1_full = np.abs(x_plot[:, None] - x_plot[None, :])
    H2_full = np.abs(x_cat[:, None] - x_cat[None, :])
    H3_full = np.abs(x_broad[:, None] - x_broad[None, :])

    sns.set_style("ticks")
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # ROW 1: Models
    models = [(H3_full, 'H3: 2-Category'), (H2_full, 'H2: 4-Category'), (H1_full, 'H1: Continuous')]
    for col_idx, (mat, title) in enumerate(models):
        ax = axs[0, col_idx]
        im = ax.imshow(mat, cmap='RdBu_r', origin='lower', extent=[0, 180, 0, 180])
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 90, 180]); ax.set_yticks([0, 90, 180])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ROW 2: RSA Hemifield Difference Boxplots (Offset < 45)
    print("  Computing RSA Hemifield Difference (Offset < 45)...")
    rsa_stats = []
    tms_groups = ['No TMS', 'Early TMS', 'Middle TMS']
    subjects = data['subjID'].unique()
    sub_data = data[np.abs(data['initOffset']) < 45]

    for tms in tms_groups:
        for subj in subjects:
            res_dict = {'subjID': subj, 'TMS': tms}
            valid_subj = True
            for hemi in ['inPF', 'outPF']:
                subset = sub_data[(sub_data['TMS'] == tms) & (sub_data['Hemifield'] == hemi) & (sub_data['subjID'] == subj)]
                if len(subset) < 10: valid_subj = False; break
                res = compute_partial_rsa_with_perms(subset['targOriActual'].values, subset['finalOri'].values)
                res_dict[f'r1_{hemi}'], res_dict[f'r2_{hemi}'], res_dict[f'r3_{hemi}'] = res['r1'], res['r2'], res['r3']
            
            if valid_subj:
                for r_key in ['r1', 'r2', 'r3']:
                    res_dict[f'{r_key}_Diff'] = res_dict[f'{r_key}_inPF'] - res_dict[f'{r_key}_outPF']
                rsa_stats.append(res_dict)
                    
    df_rsa = pd.DataFrame(rsa_stats)
    metrics = [('r3_Diff', 'Δ H3 RSA'), ('r2_Diff', 'Δ H2|H3 RSA'), ('r1_Diff', 'Δ H1|H2,H3 RSA')]
    tms_palette = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}
    
    for col_idx, (met, title) in enumerate(metrics):
        ax = axs[1, col_idx]
        if df_rsa.empty: continue
        sns.boxplot(data=df_rsa, x='TMS', y=met, palette=tms_palette, ax=ax, boxprops={'facecolor': 'none'}, showfliers=False, order=tms_groups)
        sns.stripplot(data=df_rsa, x='TMS', y=met, palette=tms_palette, alpha=0.6, ax=ax, order=tms_groups, size=5, jitter=True)
        ax.axhline(0, color='k', lw=1, ls='--')
        for i_tms, tms in enumerate(tms_groups):
            vals = df_rsa[df_rsa['TMS'] == tms][met].dropna().values
            if len(vals) > 5:
                p_val = run_permutation_test(vals)
                sig = get_sig_label(p_val)
                ax.text(i_tms, ax.get_ylim()[1]*0.9, f"{sig}\np={p_val:.3f}", ha='center', fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel("RSA Hemifield Diff (in - out)")
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_scatter_rsa_fig4A(data, fig_title, filename):
    print(f"Generating Figure: {fig_title}...")
    print("  Computing RSA Hemifield Scatter (Offset < 45)...")
    rsa_stats = []
    tms_groups = ['No TMS', 'Early TMS', 'Middle TMS']
    subjects = data['subjID'].unique()
    sub_data = data[np.abs(data['initOffset']) < 45]
    
    for tms in tms_groups:
        for subj in subjects:
            res_dict = {'subjID': subj, 'TMS': tms}
            valid_subj = True
            for hemi in ['inPF', 'outPF']:
                subset = sub_data[(sub_data['TMS'] == tms) & (sub_data['Hemifield'] == hemi) & (sub_data['subjID'] == subj)]
                if len(subset) < 10: 
                    valid_subj = False; break
                res = compute_partial_rsa_with_perms(subset['targOriActual'].values, subset['finalOri'].values)
                if not res: 
                    valid_subj = False; break
                res_dict[f'r1_{hemi}'] = res['r1']
                res_dict[f'r2_{hemi}'] = res['r2']
                res_dict[f'r3_{hemi}'] = res['r3']
            if valid_subj: rsa_stats.append(res_dict)
                    
    df_rsa = pd.DataFrame(rsa_stats)
    if df_rsa.empty: return

    metrics = [('r3', 'H3 RSA'), ('r2', 'H2|H3 RSA'), ('r1', 'H1|H2,H3 RSA')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tms_palette = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}
    
    for i, (met_base, label) in enumerate(metrics):
        ax = axes[i]
        in_col, out_col = f'{met_base}_inPF', f'{met_base}_outPF'
        
        for tms in tms_groups:
            sub = df_rsa[df_rsa['TMS'] == tms].dropna(subset=[in_col, out_col])
            if sub.empty: continue
            ax.scatter(sub[out_col], sub[in_col], color=tms_palette[tms], alpha=0.6, s=30, edgecolor='white', linewidth=0.5)
            ax.errorbar(sub[out_col].mean(), sub[in_col].mean(), xerr=sub[out_col].sem(), yerr=sub[in_col].sem(), fmt='o', color=tms_palette[tms], markersize=12, markeredgecolor='white', label=tms)
            
        all_v = df_rsa[[in_col, out_col]].values.flatten()
        # Robust limit: 95th percentile or max, but capped at something reasonable
        v_limit = np.nanmax(np.abs(all_v))
        limit = min(v_limit, 0.4) * 1.1 # Cap at 0.4 to avoid outlier stretching
        
        ax.plot([-limit, limit], [-limit, limit], 'k--', alpha=0.3)
        ax.axhline(0, color='k', alpha=0.1); ax.axvline(0, color='k', alpha=0.1)
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        ax.set_aspect('equal', 'box')
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel("outPF RSA (Correlation)"); ax.set_ylabel("inPF RSA (Correlation)")
        if i == 0: ax.legend(frameon=False)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Fig 4: Model RDMs and RSA Hemifield Contrast
plot_model_rdms_fig4(df, fig_title="4. RSA Hemifield Contrast (Offset < 45)", filename="Fig4_RSA_HemifieldContrast.png")

# Fig 4A: RSA Hemifield Scatter
plot_scatter_rsa_fig4A(df, fig_title="4A. RSA Hemifield Scatter (Offset < 45)", filename="Fig4A_RSA_HemifieldScatter.png")
