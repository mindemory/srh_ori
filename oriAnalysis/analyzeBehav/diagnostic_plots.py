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
output_dir = os.path.join(gm.p['datd'], 'Figures', 'paperFigures', 'diagnostics')
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
df = gm.master_df.copy()
df = df[df['LargeError'] == 0].copy()

df.rename(columns={
    'TMScond_label': 'TMS',        
    'stimPF_label': 'Hemifield',   
    'targOriCategory': 'OriCat',
    'hemisphereStimulated': 'HemiStim'
}, inplace=True)

df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1
df['AbsError'] = np.abs(df['errOri_scaled'])
df['AbsInitOffset'] = np.abs(df['initOffset'])

df['OffsetGroup'] = pd.cut(df['AbsInitOffset'], bins=[0, 25.1, 44.9, 90.1], labels=['Small Offset', 'Medium Offset', 'Large Offset'], include_lowest=True)
df['OffsetGroup'] = df['OffsetGroup'].cat.remove_categories(['Medium Offset'])

df['OriCat45'] = df['OriCat'].replace({
    30: 'Acute (30+60)', 60: 'Acute (30+60)',
    120: 'Obtuse (120+150)', 150: 'Obtuse (120+150)'
})

def classify_diag(row):
    hemi_stim = row['HemiStim']
    in_pf = row['Hemifield'] == 'inPF'
    is_acute = row['OriCat45'] == 'Acute (30+60)'
    if hemi_stim == 'Left':
        return 'Radial' if (in_pf and not is_acute) or (not in_pf and is_acute) else 'Tangential'
    else:
        return 'Radial' if (in_pf and is_acute) or (not in_pf and not is_acute) else 'Tangential'

df['DiagType'] = df.apply(classify_diag, axis=1)

def plot_diagnostic_grid(data, contrast_cols, fig_title, filename, bin_col=None, bins=None):
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
        ('Precision', 'Precision (K)')
    ]
    tms_groups = ['No TMS', 'Early TMS', 'Middle TMS']
    A_val, B_val = contrast_cols
    pivot_col = 'Hemifield' if A_val == 'inPF' else 'DiagType'

    # Subject aggregation
    group_cols = ['subjID', 'TMS', pivot_col]
    if bin_col: group_cols.append(bin_col)
    df_subj = data.groupby(group_cols, observed=False)[['AbsError', 'RT', 'BiasTowardsDiagonal']].mean().reset_index()

    # Mixture Model
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
                        mixture_stats.append({'TMS': tms, pivot_col: val, 'subjID': subj, 'Precision': res[1], bin_col: b if bin_col else None})
                    except: pass
    
    df_all = pd.merge(df_subj, pd.DataFrame(mixture_stats), on=group_cols, how='outer') if mixture_stats else df_subj

    num_bins = len(bins) if bins is not None else 1
    fig, axes = plt.subplots(4, num_bins * 3, figsize=(6 * num_bins * 3, 20), squeeze=False)
    fig.suptitle(fig_title, fontsize=16)

    for i_met, (met, label) in enumerate(metrics):
        for i_tms, tms in enumerate(tms_groups):
            for c_bin in range(num_bins):
                b_val = bins[c_bin] if bins is not None else None
                ax_col = i_tms * num_bins + c_bin
                ax = axes[i_met, ax_col]
                
                subset = df_all[(df_all['TMS'] == tms)]
                if bin_col: subset = subset[subset[bin_col] == b_val]
                
                piv = subset.pivot_table(index='subjID', columns=pivot_col, values=met, observed=False).reset_index()
                if A_val not in piv.columns or B_val not in piv.columns: continue
                
                piv['Diff'] = piv[A_val] - piv[B_val]
                piv = piv.dropna(subset=['Diff', B_val])
                
                sns.regplot(data=piv, x=B_val, y='Diff', ax=ax, color='black', scatter_kws={'alpha':0.6})
                ax.set_title(f"{label}\n{tms}" + (f" ({b_val})" if b_val else ""))
                ax.set_xlabel(f"Baseline ({B_val})")
                ax.set_ylabel(f"Difference ({A_val} - {B_val})")
                
                # Correlation
                if len(piv) > 3:
                    r, p = stats.pearsonr(piv[B_val], piv['Diff'])
                    ax.annotate(f"r={r:.2f}, p={p:.3f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10, fontweight='bold')

    sns.despine()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, filename)
    print(f"Saving diagnostic to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

# 1. Hemifield Diagnostic (Overall)
plot_diagnostic_grid(df, ('inPF', 'outPF'), "Diagnostic: (inPF - outPF) vs outPF (Overall)", "Diag_Fig1_Hemi_Overall.png")

# 2. Hemifield Diagnostic faceted by DiagType (Fig 2 Context)
plot_diagnostic_grid(df, ('inPF', 'outPF'), "Diagnostic: (inPF - outPF) vs outPF | faceted by DiagType", "Diag_Fig2_Hemi_by_DiagType.png",
                     bin_col='DiagType', bins=['Radial', 'Tangential'])

# 3. Hemifield Diagnostic faceted by DiagType & Offset (Fig 3 Context)
# Combine DiagType and OffsetGroup for faceting
df['Diag_Offset'] = df['DiagType'].astype(str) + " | " + df['OffsetGroup'].astype(str)
unique_combos = [
    'Radial | Small Offset', 'Tangential | Small Offset',
    'Radial | Large Offset', 'Tangential | Large Offset'
]
plot_diagnostic_grid(df, ('inPF', 'outPF'), "Diagnostic: (inPF - outPF) vs outPF | faceted by DiagType & Offset", "Diag_Fig3_Hemi_by_DiagType_Offset.png",
                     bin_col='Diag_Offset', bins=unique_combos)
