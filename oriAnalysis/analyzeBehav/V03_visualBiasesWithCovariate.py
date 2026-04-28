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
output_dir = os.path.join(gm.p['datd'], 'Figures', 'visualbiases_covariate')
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

# Offset Group: Small (0-45) vs Large (45-90)
# 'include_lowest=True' makes the first interval inclusive (0 is included)
# bins=[0, 45, 90] -> (0, 45], (45, 90]
df['OffsetGroup'] = pd.cut(df['AbsInitOffset'], bins=[0, 45, 90], labels=['Small Offset', 'Large Offset'], include_lowest=True)

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

# DiagType: nearDiag vs offDiag
def classify_diag(row):
    hemi_stim = row['HemiStim']
    in_pf = row['Hemifield'] == 'inPF'
    is_acute = row['OriCat45'] == 'Acute (30+60)'
    
    if hemi_stim == 'Left':
        return 'nearDiag' if (in_pf and not is_acute) or (not in_pf and is_acute) else 'offDiag'
    else: # Right TMS
        return 'nearDiag' if (in_pf and is_acute) or (not in_pf and not is_acute) else 'offDiag'

df['DiagType'] = df.apply(classify_diag, axis=1)

# --- ORIENTATION BINNING ---
# Bins: 0-180 in 5 deg steps
ori_bins = np.arange(0, 185, 5)
df['OriBin'] = pd.cut(df['targOriActual'], bins=ori_bins, include_lowest=True)
df['OriBinCenter'] = df['OriBin'].apply(lambda x: x.mid).astype(float)


# =============================================================================
# --- COVARIATE REGRESSION: Regress out initOffset effect from NoTMS data ---
# Strategy:
#   1. Pool ALL NoTMS trials across all subjects (inPF + outPF combined)
#   2. Fit a single quadratic polynomial of initOffset -> metric on that pool
#   3. For every trial (all subjects, all conditions) compute:
#        residual = raw_metric - model_prediction(initOffset) + grand_mean
#      This removes the group-level offset-driven trend from every trial.
# =============================================================================

METRICS_TO_RESIDUALIZE = [
    'AbsError',
    'errOri_scaled',
    'BiasTowardsDiagonal',
    'RT'
]

print("\n--- Fitting group-level quadratic (NoTMS pooled, inPF+outPF) and residualizing ---")

notms_pool = df[df['TMS_Group'] == 'No TMS'].copy()
x_all      = df['initOffset'].values   # initOffset for every trial (signed, -90..+90)

fit_polys_global = {}   # stored so plotting block can rebuild them for Fig 0

for met in METRICS_TO_RESIDUALIZE:
    resid_col = f"{met}_resid"

    # Fit on pooled NoTMS data
    clean  = notms_pool[['initOffset', met]].dropna()
    coeffs = np.polyfit(clean['initOffset'].values, clean[met].values, 2)
    poly   = np.poly1d(coeffs)
    fit_polys_global[met] = poly

    anchor = float(poly(0))   # model prediction at 0° offset — the interpretable baseline

    # Residual anchored at poly(0):
    #   resid = raw - model(offset) + model(0)
    # A trial at 0° offset has resid == raw (no correction needed).
    # Trials with non-zero offset are adjusted relative to that 0° baseline.
    df[resid_col] = df[met].values - poly(x_all) + anchor

    print(f"  {met}: coeffs={coeffs.round(5)}, "
          f"poly(0)={anchor:.3f} (anchor), "
          f"fit on {len(clean)} NoTMS trials")

print("Covariate regression complete.\n")



# --- Reusable Plotting Functions (Updated for 2 Cols) ---

def plot_ori_bins(data, palette, fig_title, filename):
    print(f"Generating Figure: {fig_title}...")
    
    # Pre-aggregate per subject to get Means/Counts per bin
    # This ensures we plot the distribution of SUBJECT Means, not trial-level variance
    print("Aggregating per subject/bin...")
    
    # We need trial counts, so we can use 'size'
    agg_funcs = {
        'BiasTowardsDiagonal_resid': 'mean',
        'AbsError_resid': 'mean',
        'subjID': 'size' # This becomes the Trial Count
    }
    
    df_agg = data.groupby(['subjID', 'TMS_Group', 'OffsetGroup', 'OriBinCenter'], observed=False).agg(agg_funcs).rename(columns={'subjID': 'TrialCount'}).reset_index()

    
    # Metrics to plot (residualized)
    metrics = [
        ('TrialCount', 'Trial Count'),
        ('BiasTowardsDiagonal_resid', 'Diag Attraction (resid)'),
        ('AbsError_resid', 'Abs Error (resid)')
    ]
    
    offset_groups = ['Small Offset', 'Large Offset']
    
    # 36 Bins for 0-180 in 5 deg steps (0-5, 5-10, ... 175-180)
    # Centers: 2.5, 7.5, ..., 177.5
    all_centers = np.arange(2.5, 180, 5) 
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(fig_title, fontsize=16)
    
    for r, (met, label) in enumerate(metrics):
        for c, off_group in enumerate(offset_groups):
            ax = axes[r, c]
            
            subset = df_agg[df_agg['OffsetGroup'] == off_group]
            if subset.empty: continue
            
            # Barplot with SEM, Fixed Order
            sns.barplot(
                data=subset, x='OriBinCenter', y=met, hue='TMS_Group',
                palette={'No TMS': 'k', 'TMS': 'r'}, # Fixed palette for TMS
                errorbar='se', ax=ax, capsize=0.1, errwidth=1,
                order=all_centers # Ensure full x-axis even if bins empty
            )
            
            ax.set_title(f"{label} | {off_group}")
            
            if r == 2: ax.set_xlabel("Orientation (deg)")
            else: ax.set_xlabel("")
            
            if c == 0: ax.set_ylabel(label)
            else: ax.set_ylabel("")
            
            # Ticks at 0, 15, 30, ... 165, (180?)
            # Bin 0 (0-5) -> Index 0. Label "0"
            # Bin 3 (15-20) -> Index 3. Label "15"
            # ...
            # Bin 33 (165-170) -> Index 33. Label "165"
            # Bin 36 (180) -> Index 36. (Out of range, assume ends at 180)
            
            tick_indices = np.arange(0, 37, 3) # 0, 3, 6, ... 36
            tick_labels = [str(i*5) for i in tick_indices]
            
            # Only set ticks that fall within the range of bars (0-35)
            # We can force the axis limits if needed, usually barplot auto-scales 0..N-1
            # Index 36 (180) corresponds to the Right Edge of the last bar
            
            # Set ticks
            valid_indices = [i for i in tick_indices if i < len(all_centers)]
            valid_labels = [tick_labels[i] for i, idx in enumerate(tick_indices) if idx < len(all_centers)]
            
            ax.set_xticks(valid_indices)
            ax.set_xticklabels(valid_labels)
            
            if met == 'BiasTowardsDiagonal_resid':
                ax.axhline(0, color='k', lw=1, ls='--')

            # Legend only in top left
            if r == 0 and c == 0:
                ax.legend(loc='upper right', title='TMS Group')
            elif ax.get_legend():
                ax.get_legend().remove()
                
    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_adjustment_grid(data, hue_col, palette, fig_title, filename, hue_order=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError_resid', 'Abs Error (resid)'), 
        ('errOri_scaled_resid', 'Signed Bias (resid)'), 
        ('RT_resid', 'Reaction Time (resid)'), 
        ('BiasTowardsDiagonal_resid', 'Diag Attraction (resid)')
    ]
    tms_conds = ['No TMS', 'TMS'] # New Groups
    
    if hue_order: unique_hues = hue_order
    else: unique_hues = np.sort(data[hue_col].unique())

    if isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(unique_hues))
        hue_map = dict(zip(unique_hues, colors))
    else:
        hue_map = palette

    # 4 Rows x 2 Columns (No TMS, TMS)
    fig, axes = plt.subplots(4, 2, figsize=(10, 18))
    fig.suptitle(fig_title, fontsize=16)

    for r, (met, label) in enumerate(metrics):
        for c, tms in enumerate(tms_conds):
            ax = axes[r, c]
            subset = data[data['TMS_Group'] == tms]
            
            for hue_val in unique_hues:
                hue_data = subset[subset[hue_col] == hue_val]
                if len(hue_data) < 5: continue
                
                clean = hue_data[['initOffset', met]].dropna()
                if len(clean) < 5: continue

                # Quadratic Fit
                try:
                    coeffs = np.polyfit(clean['initOffset'], clean[met], 2)
                    p_poly = np.poly1d(coeffs)
                    y_pred = p_poly(clean['initOffset'])
                    y_true = clean[met]
                    
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - (ss_res / ss_tot)
                    lbl = f"{hue_val}: $R^2$={r2:.3f}"
                except:
                    lbl = f"{hue_val}"

                sns.regplot(
                    data=hue_data, x='initOffset', y=met,
                    ax=ax, color=hue_map[hue_val],
                    scatter_kws={'alpha': 0.1, 's': 5},
                    line_kws={'alpha': 0.8, 'label': lbl},
                    ci=None, order=2
                )
            
            ax.set_title(f"{label} | {tms}")
            if r == 3: ax.set_xlabel("Signed Init Offset (deg)")
            else: ax.set_xlabel("")
            if c == 0: ax.set_ylabel(label)
            else: ax.set_ylabel("")
            
            ax.legend(fontsize='x-small', loc='best')

            if met == 'AbsError_resid': ax.set_ylim(0, 20)
            elif met == 'errOri_scaled_resid': ax.set_ylim(-20, 20)
            elif met == 'RT_resid': ax.set_ylim(0.5, 3)
            elif met == 'BiasTowardsDiagonal_resid': ax.set_ylim(-20, 20)
    
    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close() 

def plot_binned_grid(data, hue_col, palette, fig_title, filename, hue_order=None, custom_limits=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError_resid', 'Abs Error (resid)'), 
        ('RT_resid', 'Reaction Time (resid)'), 
        ('BiasTowardsDiagonal_resid', 'Diag Attraction (resid)'),
        ('Precision', 'Precision (K)')
    ]
    
    tms_conds = ['No TMS', 'TMS']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 2 Rows, 2 Cols
    axes = axes.flatten()
    fig.suptitle(fig_title, fontsize=16)

    print("Fitting Mixture Models (Combined)...")
    mixture_stats = []
    
    hues = data[hue_col].unique()
    subjects = data['subjID'].unique()
    
    for tms in tms_conds:
        for h in hues:
            for subj in subjects:
                subset = data[
                    (data['TMS_Group'] == tms) & 
                    (data[hue_col] == h) &
                    (data['subjID'] == subj)
                ]
                
                if len(subset) < 10: continue
                
                angles = subset['errOri_scaled_resid'].dropna().values
                try:
                    res = md.fit_model(angles, 'mixtureModel')
                    g, K = res
                    if g < 0 or g > 1 or K < 0 or K > 50: g, K = np.nan, np.nan
                except:
                    g, K = np.nan, np.nan
                
                if not np.isnan(K):
                    mixture_stats.append({
                        'TMS_Group': tms, hue_col: h, 'subjID': subj,
                        'Precision': K, 'GuessRate': g
                    })
    
    df_mix = pd.DataFrame(mixture_stats)

    for i, (met, label) in enumerate(metrics):
        ax = axes[i]
        
        if met in ['Precision', 'GuessRate']:
            plot_data = df_mix
        else:
            plot_data = data
            
        if plot_data.empty: continue

        sns.barplot(
            data=plot_data, x='TMS_Group', y=met, hue=hue_col,
            palette=palette, errorbar='se', ax=ax,
            capsize=0.1, errwidth=1.5, hue_order=hue_order
        )
        
        ax.set_title(f"{label}")
        
        if i >= 2: 
            ax.set_xlabel("TMS Condition")
        else: 
            ax.set_xlabel("")
            ax.set_xticklabels([])
        
        ax.set_ylabel(label)

        if custom_limits and met in custom_limits:
            ax.set_ylim(custom_limits[met])
        else:
            if met == 'AbsError_resid': ax.set_ylim(0, 12)
            elif met == 'errOri_scaled_resid': ax.set_ylim(-7, 7); ax.axhline(0, color='k', lw=0.5)
            elif met == 'RT_resid': ax.set_ylim(1.0, 2.5)
            elif met == 'BiasTowardsDiagonal_resid': ax.set_ylim(-0.5, 4); ax.axhline(0, color='k', lw=0.5)
            elif met == 'Precision': ax.set_ylim(0, 25) 
            elif met == 'GuessRate': ax.set_ylim(-0.1, 0.1)

        if met == 'BiasTowardsDiagonal_resid':
            ax.axhline(0, color='k', lw=0.5)

        if i == 0:
            ax.legend(fontsize='x-small', loc='upper right', title=hue_col)
        elif ax.get_legend():
            ax.get_legend().remove()

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close() 

def plot_binned_grid_hemi(data, hue_col, palette, fig_title, filename, hue_order=None, custom_limits=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError_resid', 'Abs Error (resid)'), 
        ('RT_resid', 'Reaction Time (resid)'), 
        ('BiasTowardsDiagonal_resid', 'Diag Attraction (resid)'),
        ('Precision', 'Precision (K)')
    ]
    
    def make_tms_hemi_cat(row):
        if row['TMS_Group'] == 'No TMS':
            return 'NoTMS'
        else:
            return f"{row['HemiStim']}TMS_{row['Hemifield']}"
    
    data = data.copy()
    data['TMS_Hemi'] = data.apply(make_tms_hemi_cat, axis=1)
    
    x_order = ['NoTMS', 'LeftTMS_inPF', 'LeftTMS_outPF', 'RightTMS_inPF', 'RightTMS_outPF']
    data['TMS_Hemi'] = pd.Categorical(data['TMS_Hemi'], categories=x_order, ordered=True)
    
    hemi_counts = data.groupby('HemiStim')['subjID'].nunique()
    print(f"Subjects per hemisphere: {dict(hemi_counts)}")
    
    print("\nFitting Mixture Models...")
    mixture_stats = []
    
    hues = data[hue_col].unique()
    subjects = data['subjID'].unique()
    
    for tms_hemi in x_order:
        for h in hues:
            for subj in subjects:
                subset = data[
                    (data['TMS_Hemi'] == tms_hemi) & 
                    (data[hue_col] == h) &
                    (data['subjID'] == subj)
                ]
                
                if len(subset) < 10: continue
                
                angles = subset['errOri_scaled_resid'].dropna().values
                try:
                    res = md.fit_model(angles, 'mixtureModel')
                    g, K = res
                    if g < 0 or g > 1 or K < 0 or K > 50: g, K = np.nan, np.nan
                except:
                    g, K = np.nan, np.nan
                
                if not np.isnan(K):
                    mixture_stats.append({
                        'TMS_Hemi': tms_hemi, hue_col: h, 'subjID': subj,
                        'Precision': K, 'GuessRate': g
                    })
    
    df_mix = pd.DataFrame(mixture_stats) if mixture_stats else pd.DataFrame()
    if not df_mix.empty:
        df_mix['TMS_Hemi'] = pd.Categorical(df_mix['TMS_Hemi'], categories=x_order, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.suptitle(f"{fig_title}", fontsize=16)
    
    for i, (met, label) in enumerate(metrics):
        ax = axes[i]
        
        if met in ['Precision', 'GuessRate']:
            if df_mix.empty:
                continue
            plot_data = df_mix
        else:
            plot_data = data
            
        if plot_data.empty: continue

        sns.barplot(
            data=plot_data, x='TMS_Hemi', y=met, hue=hue_col,
            palette=palette, errorbar='se', ax=ax,
            capsize=0.1, errwidth=1.5, hue_order=hue_order, order=x_order
        )
        
        ax.set_title(f"{label}")
        
        if i >= 2: 
            ax.set_xlabel("TMS Condition x Hemifield")
            ax.set_xticklabels(['NoTMS', 'L-in', 'L-out', 'R-in', 'R-out'], rotation=45, ha='right')
        else: 
            ax.set_xlabel("")
            ax.set_xticklabels([])
        
        ax.set_ylabel(label)

        if custom_limits and met in custom_limits:
            ax.set_ylim(custom_limits[met])
        else:
            if met == 'AbsError_resid': ax.set_ylim(0, 12)
            elif met == 'RT_resid': ax.set_ylim(1.0, 2.5)
            elif met == 'BiasTowardsDiagonal_resid': ax.set_ylim(-0.5, 4); ax.axhline(0, color='k', lw=0.5)
            elif met == 'Precision': ax.set_ylim(0, 25) 

        if met == 'BiasTowardsDiagonal_resid':
            ax.axhline(0, color='k', lw=0.5)

        if i == 0:
            ax.legend(fontsize='small', loc='upper right', title=hue_col)
        elif ax.get_legend():
            ax.get_legend().remove()

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_binned_grid_inpf_outpf(data, hue_col, palette, fig_title, filename, hue_order=None, custom_limits=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError_resid', 'Abs Error (resid)'), 
        ('RT_resid', 'Reaction Time (resid)'), 
        ('BiasTowardsDiagonal_resid', 'Diag Attraction (resid)'),
        ('Precision', 'Precision (K)')
    ]
    
    def make_tms_inpf_cat(row):
        if row['TMS_Group'] == 'No TMS':
            return 'NoTMS'
        else:
            return f"TMS_{row['Hemifield']}"
            
    data = data.copy()
    data['TMS_Condition'] = data.apply(make_tms_inpf_cat, axis=1)
    
    x_order = ['NoTMS', 'TMS_inPF', 'TMS_outPF']
    data['TMS_Condition'] = pd.Categorical(data['TMS_Condition'], categories=x_order, ordered=True)
    
    print("\nFitting Mixture Models...")
    mixture_stats = []
    
    hues = data[hue_col].unique()
    subjects = data['subjID'].unique()
    
    for cond in x_order:
        for h in hues:
            for subj in subjects:
                subset = data[
                    (data['TMS_Condition'] == cond) & 
                    (data[hue_col] == h) &
                    (data['subjID'] == subj)
                ]
                
                if len(subset) < 10: continue
                
                angles = subset['errOri_scaled_resid'].dropna().values
                try:
                    res = md.fit_model(angles, 'mixtureModel')
                    g, K = res
                    if g < 0 or g > 1 or K < 0 or K > 50: g, K = np.nan, np.nan
                except:
                    g, K = np.nan, np.nan
                
                if not np.isnan(K):
                    mixture_stats.append({
                        'TMS_Condition': cond, hue_col: h, 'subjID': subj,
                        'Precision': K, 'GuessRate': g
                    })
                    
    df_mix = pd.DataFrame(mixture_stats) if mixture_stats else pd.DataFrame()
    if not df_mix.empty:
        df_mix['TMS_Condition'] = pd.Categorical(df_mix['TMS_Condition'], categories=x_order, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.suptitle(f"{fig_title}", fontsize=16)
    
    for i, (met, label) in enumerate(metrics):
        ax = axes[i]
        
        if met in ['Precision', 'GuessRate']:
            if df_mix.empty:
                continue
            plot_data = df_mix
        else:
            plot_data = data
            
        if plot_data.empty: continue

        sns.barplot(
            data=plot_data, x='TMS_Condition', y=met, hue=hue_col,
            palette=palette, errorbar='se', ax=ax,
            capsize=0.1, errwidth=1.5, hue_order=hue_order, order=x_order
        )
        
        ax.set_title(f"{label}")
        
        if i >= 2: 
            ax.set_xlabel("TMS Condition x Hemifield")
            #ax.set_xticklabels(['NoTMS', 'TMS inPF', 'TMS outPF'], rotation=45, ha='right')
        else: 
            ax.set_xlabel("")
            ax.set_xticklabels([])
        
        ax.set_ylabel(label)

        if custom_limits and met in custom_limits:
            ax.set_ylim(custom_limits[met])
        else:
            if met == 'AbsError_resid': ax.set_ylim(0, 12)
            elif met == 'RT_resid': ax.set_ylim(1.0, 2.5)
            elif met == 'BiasTowardsDiagonal_resid': ax.set_ylim(-0.5, 4); ax.axhline(0, color='k', lw=0.5)
            elif met == 'Precision': ax.set_ylim(0, 25) 

        if met == 'BiasTowardsDiagonal_resid':
            ax.axhline(0, color='k', lw=0.5)

        if i == 0:
            ax.legend(fontsize='small', loc='upper right', title=hue_col)
        elif ax.get_legend():
            ax.get_legend().remove()

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_binned_grid_inpf_outpf_split(data, hue_col, palette, fig_title, filename, hue_order=None, bin_col='OffsetGroup', custom_limits=None, offset_groups=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError_resid', 'Abs Error (resid)'), 
        ('RT_resid', 'Reaction Time (resid)'), 
        ('BiasTowardsDiagonal_resid', 'Diag Attraction (resid)'),
        ('Precision', 'Precision (K)')
    ]
    
    if offset_groups is None:
        if data[bin_col].dtype.name == 'category':
            offset_groups = list(data[bin_col].cat.categories)
        else:
            offset_groups = list(data[bin_col].unique())
    
    def make_tms_inpf_cat(row):
        if row['TMS_Group'] == 'No TMS':
            return 'NoTMS'
        else:
            return f"TMS_{row['Hemifield']}"
            
    data = data.copy()
    data['TMS_Condition'] = data.apply(make_tms_inpf_cat, axis=1)
    
    x_order = ['NoTMS', 'TMS_inPF', 'TMS_outPF']
    data['TMS_Condition'] = pd.Categorical(data['TMS_Condition'], categories=x_order, ordered=True)
    
    print("\nFitting Mixture Models...")
    mixture_stats = []
    
    bins = data[bin_col].unique()
    hues = data[hue_col].unique()
    subjects = data['subjID'].unique()
    
    for cond in x_order:
        for b in bins:
            for h in hues:
                for subj in subjects:
                    subset = data[
                        (data['TMS_Condition'] == cond) & 
                        (data[bin_col] == b) & 
                        (data[hue_col] == h) &
                        (data['subjID'] == subj)
                    ]
                    
                    if len(subset) < 10: continue
                    
                    angles = subset['errOri_scaled_resid'].dropna().values
                    try:
                        res = md.fit_model(angles, 'mixtureModel')
                        g, K = res
                        if g < 0 or g > 1 or K < 0 or K > 50: g, K = np.nan, np.nan
                    except:
                        g, K = np.nan, np.nan
                    
                    if not np.isnan(K):
                        mixture_stats.append({
                            'TMS_Condition': cond, bin_col: b, hue_col: h, 'subjID': subj,
                            'Precision': K, 'GuessRate': g
                        })
                        
    df_mix = pd.DataFrame(mixture_stats) if mixture_stats else pd.DataFrame()
    if not df_mix.empty:
        df_mix['TMS_Condition'] = pd.Categorical(df_mix['TMS_Condition'], categories=x_order, ordered=True)

    num_cols = len(offset_groups)
    fig, axes = plt.subplots(4, num_cols, figsize=(6*num_cols, 16), squeeze=False)
    fig.suptitle(f"{fig_title}", fontsize=16)
    
    for r, (met, label) in enumerate(metrics):
        for c, off_group in enumerate(offset_groups):
            ax = axes[r, c]
            
            if met in ['Precision', 'GuessRate']:
                if df_mix.empty:
                    continue
                plot_data = df_mix[df_mix[bin_col] == off_group]
            else:
                plot_data = data[data[bin_col] == off_group]
                
            if plot_data.empty: continue

            sns.barplot(
                data=plot_data, x='TMS_Condition', y=met, hue=hue_col,
                palette=palette, errorbar='se', ax=ax,
                capsize=0.1, errwidth=1.5, hue_order=hue_order, order=x_order
            )
            
            ax.set_title(f"{label} | {off_group}")
            
            if r == 3: 
                ax.set_xlabel("TMS Condition x Hemifield")
            else: 
                ax.set_xlabel("")
                ax.set_xticklabels([])
            
            if c == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("")

            if custom_limits and met in custom_limits:
                ax.set_ylim(custom_limits[met])
            else:
                if met == 'AbsError_resid': ax.set_ylim(0, 12)
                elif met == 'RT_resid': ax.set_ylim(1.0, 2.5)
                elif met == 'BiasTowardsDiagonal_resid': ax.set_ylim(-0.5, 4); ax.axhline(0, color='k', lw=0.5)
                elif met == 'Precision': ax.set_ylim(0, 25) 

            if met == 'BiasTowardsDiagonal_resid':
                ax.axhline(0, color='k', lw=0.5)

            if r == 0 and c == 0:
                ax.legend(fontsize='small', loc='upper right', title=hue_col)
            elif ax.get_legend():
                ax.get_legend().remove()

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_diagonal_bias_grid(data, fig_title, filename, custom_limits=None):
    print(f"Generating Figure: {fig_title}...")
    
    ori_groups = [
        ([30, 60], 'Acute (30° + 60°)')
        # Removed obtuse here or both are here, let's keep both
    ]
    ori_groups = [
        ([30, 60], 'Acute (30° + 60°)'),
        ([120, 150], 'Obtuse (120° + 150°)')
    ]
    
    data = data.copy()
    
    def make_vf(row):
        hemi_stim = row['HemiStim']
        hemifield = row['Hemifield']
        if hemi_stim == 'Left':
            return 'RightVF' if hemifield == 'inPF' else 'LeftVF'
        else:
            return 'LeftVF' if hemifield == 'inPF' else 'RightVF'
    
    data['VisualField'] = data.apply(make_vf, axis=1)
    
    def make_ipsi_contra(row):
        if row['TMS_Group'] == 'No TMS':
            return 'NoTMS'
        hemi_stim = row['HemiStim']
        vf = row['VisualField']
        if (hemi_stim == 'Left' and vf == 'LeftVF') or (hemi_stim == 'Right' and vf == 'RightVF'):
            return 'IpsiTMS'
        else:
            return 'ContraTMS'
    
    data['TMS_Type'] = data.apply(make_ipsi_contra, axis=1)
    
    hemi_counts = data.groupby('HemiStim')['subjID'].nunique()
    print(f"Subjects per hemisphere: {dict(hemi_counts)}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(fig_title, fontsize=16)
    
    leftvf_color = '#2196F3'
    rightvf_color = '#4CAF50'
    vf_palette = {'LeftVF': leftvf_color, 'RightVF': rightvf_color}
    
    x_order = ['NoTMS', 'IpsiTMS', 'ContraTMS']

    for i, (ori_cats, ori_label) in enumerate(ori_groups):
        ax = axes[i]
        
        plot_data = data[data['OriCat'].isin(ori_cats)]
        
        if plot_data.empty:
            continue

        plot_data['TMS_Type'] = pd.Categorical(plot_data['TMS_Type'], categories=x_order, ordered=True)
        
        sns.barplot(
            data=plot_data, x='TMS_Type', y='BiasTowardsDiagonal_resid', hue='VisualField',
            palette=vf_palette, errorbar='se', ax=ax,
            capsize=0.1, errwidth=1.5, hue_order=['LeftVF', 'RightVF'], order=x_order
        )
        
        ax.set_title(f"{ori_label}")
        ax.set_xlabel("TMS Condition")
        
        if i == 0: 
            ax.set_ylabel("Diagonal Attraction (resid)")
        else: 
            ax.set_ylabel("")

        if custom_limits and 'BiasTowardsDiagonal_resid' in custom_limits:
            ax.set_ylim(custom_limits['BiasTowardsDiagonal_resid'])
        else:
            ax.set_ylim(-1, 6)
        
        ax.axhline(0, color='k', lw=0.5)
        
        if i == 0:
            ax.legend(fontsize='small', loc='upper right', title='Visual Field')
        elif ax.get_legend():
            ax.get_legend().remove()

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_polar_diff_oricat(data, fig_title, filename):
    print(f"Generating Figure: {fig_title}...")
    
    # 1. Format and map orientations
    # For Right TMS subjects, mirror horizontal orientations: 180 - OriCat
    def map_ori(row):
        return 180 - row['OriCat'] if row['HemiStim'] == 'Right' else row['OriCat']
    
    df['OriMapped'] = df.apply(map_ori, axis=1)
    
    metrics = ['AbsError_resid', 'RT_resid', 'BiasTowardsDiagonal_resid']
    
    # 2. Aggregate including TMS_Group
    agg_means = df.groupby(['subjID', 'TMS_Group', 'OriMapped', 'Hemifield'], observed=True)[metrics].mean().reset_index()
    
    subjs = df['subjID'].unique()
    ori_cats = [30, 60, 120, 150]
    hemis = ['inPF', 'outPF']
    tms_groups = ['TMS', 'No TMS']
    
    prec_records = []
    print("  Fitting Mixture Models for Precision (Polar Diff)...")
    for subj in subjs:
        for tms_grp in tms_groups:
            for ori in ori_cats:
                for hemi in hemis:
                    sub_df = df[(df['subjID'] == subj) & (df['TMS_Group'] == tms_grp) & 
                                (df['OriMapped'] == ori) & (df['Hemifield'] == hemi)]
                    if len(sub_df) < 5: continue
                    angles = sub_df['errOri_scaled_resid'].dropna().values
                    try:
                        res = md.fit_model(angles, 'mixtureModel')
                        g, K = res
                        if g < 0 or g > 1 or K < 0 or K > 50: K = np.nan
                    except:
                        K = np.nan
                    if not np.isnan(K):
                        prec_records.append({'subjID': subj, 'TMS_Group': tms_grp, 'OriMapped': ori, 'Hemifield': hemi, 'Precision': K})
                    
    df_prec = pd.DataFrame(prec_records)
    if not df_prec.empty:
        agg = pd.merge(agg_means, df_prec, on=['subjID', 'TMS_Group', 'OriMapped', 'Hemifield'], how='left')
    else:
        agg = agg_means
        agg['Precision'] = np.nan
        
    # 3. Calculate Difference inPF - outPF
    diff_records = []
    for subj in subjs:
        for tms_grp in tms_groups:
            for ori in ori_cats:
                sub = agg[(agg['subjID'] == subj) & (agg['TMS_Group'] == tms_grp) & (agg['OriMapped'] == ori)]
                if len(sub) == 2:
                    in_val = sub[sub['Hemifield'] == 'inPF'].iloc[0]
                    out_val = sub[sub['Hemifield'] == 'outPF'].iloc[0]
                    rec = {'subjID': subj, 'TMS_Group': tms_grp, 'OriMapped': ori}
                    for m in metrics + ['Precision']:
                        if m in in_val and not pd.isna(in_val[m]) and not pd.isna(out_val[m]):
                            rec[m] = in_val[m] - out_val[m]
                    diff_records.append(rec)
                
    df_diff = pd.DataFrame(diff_records)
    group_stats = df_diff.groupby(['TMS_Group', 'OriMapped']).agg(['mean', 'sem'])
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), subplot_kw={'projection': 'polar'})
    fig.suptitle(fig_title, fontsize=16)
    fig.text(0.5, 0.93, "(Right TMS subjects mirrored to standardize geometry: '30°' always means tilting outward from midline)", 
             ha='center', fontsize=10, style='italic', color='grey')
    
    plot_metrics = [
        ('AbsError_resid', 'Δ Abs Error (resid)\n(inPF - outPF)', axes[0,0]),
        ('RT_resid', 'Δ Reaction Time (resid)\n(inPF - outPF)', axes[0,1]),
        ('BiasTowardsDiagonal_resid', 'Δ Diag Attraction (resid)\n(inPF - outPF)', axes[1,0]),
        ('Precision', 'Δ Precision K\n(inPF - outPF)', axes[1,1])
    ]
    
    thetas = np.deg2rad(ori_cats)
    sort_idx = np.argsort(thetas)
    thetas_sorted = thetas[sort_idx]
    
    palette = {'TMS': '#e91e63', 'No TMS': '#757575'}
    
    for (m, label, ax) in plot_metrics:
        if m not in group_stats.columns.levels[0]: continue
        
        # Calculate dynamic limit including both groups
        max_abs = 0
        for tms_grp in tms_groups:
            if tms_grp not in group_stats.index.get_level_values('TMS_Group'): continue
            grp_data = group_stats.xs(tms_grp, level='TMS_Group')[m]
            if not all(o in grp_data.index for o in ori_cats): continue
                
            means_plot = grp_data['mean'].loc[ori_cats].values[sort_idx]
            sems_plot  = grp_data['sem'].loc[ori_cats].values[sort_idx]
            cur_max = np.nanmax(np.abs(means_plot) + sems_plot)
            if not pd.isna(cur_max) and cur_max > max_abs:
                max_abs = cur_max
                
        max_abs = max_abs * 1.5 if max_abs > 0 else 1
        
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rorigin(-max_abs)
        ax.set_ylim(-max_abs, max_abs)
        
        # Zero line
        ax.plot(np.linspace(0, np.pi, 100), np.zeros(100), 'k--', alpha=0.4, lw=1)
        
        for tms_grp in tms_groups:
            if tms_grp not in group_stats.index.get_level_values('TMS_Group'): continue
            grp_data = group_stats.xs(tms_grp, level='TMS_Group')[m]
            if not all(o in grp_data.index for o in ori_cats): continue
            
            means_plot = grp_data['mean'].loc[ori_cats].values[sort_idx]
            sems_plot  = grp_data['sem'].loc[ori_cats].values[sort_idx]
            
            color = palette[tms_grp]
            ls = '-' if tms_grp == 'TMS' else '--'
            alpha = 1.0 if tms_grp == 'TMS' else 0.7
            
            ax.plot(thetas_sorted, means_plot, 'o', ls=ls, color=color, alpha=alpha, linewidth=2, markersize=8, label=tms_grp)
            ax.errorbar(thetas_sorted, means_plot, yerr=sems_plot, fmt='none', ecolor=color, alpha=alpha, capsize=4, elinewidth=1.5)
        
        ax.set_title(label, pad=15, fontsize=12, fontweight='bold')
        ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
        
        ax.set_yticks([0])
        ax.set_yticklabels(['0'], color='gray', fontsize=9)
        
        if ax == axes[0,0]:
            ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), title='Condition')
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

# --- PLOTTING CALLS (DEBUG: Fig 0 + Fig 1) ---

hemi_palette = {'inPF': '#e91e63', 'outPF': '#ff9800'}

# --- Diagnostic: verify residualization did something ---
print("\n=== RESIDUALIZATION DIAGNOSTIC ===")
for met in METRICS_TO_RESIDUALIZE:
    resid_col = f"{met}_resid"
    r_raw  = df[[met, 'initOffset']].dropna().corr().iloc[0, 1]
    r_res  = df[[resid_col, 'initOffset']].dropna().corr().iloc[0, 1]
    print(f"  {met:30s}  r(offset)={r_raw:+.4f}  →  {resid_col}: r(offset)={r_res:+.4f}")
print("==================================\n")

# =============================================================================
# FIG 0: Diagnostic — raw data + quadratic fit, then residuals vs initOffset
# =============================================================================
print("Generating Figure 0: Covariate fit diagnostic...")

metrics_diag = [
    ('AbsError',            'AbsError_resid',            'Abs Error',          (0, 30)),
    ('errOri_scaled',       'errOri_scaled_resid',       'Signed Bias',         (-30, 30)),
    ('BiasTowardsDiagonal', 'BiasTowardsDiagonal_resid', 'Diag Attraction',     (-30, 30)),
    ('RT',                  'RT_resid',                  'Reaction Time (s)',   (0.4, 4.0)),
]

# Re-use the polys already computed in the regression block
fit_polys = fit_polys_global

x_curve = np.linspace(-90, 90, 300)

fig0, axes0 = plt.subplots(2, 4, figsize=(20, 8))
fig0.suptitle("Fig 0: initOffset Covariate Check\n"
              "Top: raw NoTMS data + quadratic fit  |  Bottom: residuals for ALL trials",
              fontsize=13)

for col, (met, resid_col, label, ylim) in enumerate(metrics_diag):

    # --- TOP ROW: raw NoTMS scatter + fit ---
    ax_top = axes0[0, col]
    notms_clean = notms_pool[['initOffset', met]].dropna()

    # Bin mean ± se across subjects for cleaner view
    notms_pool_tmp = notms_pool[['initOffset', met, 'subjID']].dropna().copy()
    notms_pool_tmp['offsetBin'] = pd.cut(
        notms_pool_tmp['initOffset'],
        bins=np.arange(-90, 91, 10), include_lowest=True
    )
    bin_means = notms_pool_tmp.groupby(['offsetBin', 'subjID'], observed=True)[met].mean().reset_index()
    bin_centers = bin_means['offsetBin'].apply(lambda x: x.mid).astype(float)
    bin_means['offsetBinCenter'] = bin_centers

    sns.lineplot(
        data=bin_means, x='offsetBinCenter', y=met,
        errorbar='se', color='steelblue', ax=ax_top,
        err_style='bars', err_kws={'capsize': 3, 'elinewidth': 1.2},
        label='NoTMS mean±SE (10° bins)'
    )

    # Quadratic fit curve
    y_curve = fit_polys[met](x_curve)
    ax_top.plot(x_curve, y_curve, color='crimson', lw=2, label='Quadratic fit')

    ax_top.set_title(f"{label}")
    ax_top.set_xlabel("Signed initOffset (°)")
    ax_top.set_ylabel(label if col == 0 else "")
    ax_top.set_ylim(ylim)
    ax_top.axvline(0, color='k', lw=0.5, ls='--')
    ax_top.legend(fontsize='x-small') if col == 0 else ax_top.get_legend() and ax_top.get_legend().remove()

    # --- BOTTOM ROW: residuals vs initOffset, ALL conditions ---
    ax_bot = axes0[1, col]

    all_clean = df[['initOffset', resid_col, 'TMS_Group']].dropna().copy()
    all_clean['offsetBin'] = pd.cut(
        all_clean['initOffset'],
        bins=np.arange(-90, 91, 10), include_lowest=True
    )
    all_clean['offsetBinCenter'] = all_clean['offsetBin'].apply(lambda x: x.mid).astype(float)

    sns.lineplot(
        data=all_clean, x='offsetBinCenter', y=resid_col,
        hue='TMS_Group', palette={'No TMS': '#444444', 'TMS': '#e53935'},
        errorbar='se', ax=ax_bot,
        err_style='bars', err_kws={'capsize': 3, 'elinewidth': 1.2}
    )
    ax_bot.axhline(0, color='gray', lw=0.8, ls='--', label='zero (model)')
    ax_bot.axvline(0, color='k', lw=0.5, ls='--')

    ax_bot.set_title(f"{label} residuals")
    ax_bot.set_xlabel("Signed initOffset (°)")
    ax_bot.set_ylabel(f"{label} (resid)" if col == 0 else "")
    ax_bot.set_ylim(ylim)
    if col == 0:
        ax_bot.legend(fontsize='x-small')
    elif ax_bot.get_legend():
        ax_bot.get_legend().remove()

sns.despine()
plt.tight_layout()
save_path = os.path.join(output_dir, 'Fig0_CovariateCheck.png')
print(f"Saving to {save_path}")
plt.savefig(save_path, dpi=300)
plt.close()

# =============================================================================
# FIG 1: Metrics by Hemifield
# =============================================================================
ori_palette        = {30: '#1f77b4', 60: '#2ca02c', 120: '#ff7f0e', 150: '#d62728'}
ori_group_palette  = {'30_150': '#009688', '60_120': '#9c27b0'}
ori_cat45_palette  = {'Acute (30+60)': '#3f51b5', 'Obtuse (120+150)': '#e91e63'}
diag_palette       = {'nearDiag': '#E65100', 'offDiag': '#00B8D4'}

# Shared residual limits — anchored at poly(0), so values are in real units
# poly(0) ≈ AbsError:6.61°, RT:1.49s, BiasTowardsDiag:1.85°
resid_limits = {
    'AbsError_resid':            (3,   12),
    'RT_resid':                  (0.9,  2.1),
    'BiasTowardsDiagonal_resid': (-2,   6),
    'Precision':                 (5,   20)
}

plot_binned_grid(
    df, hue_col='Hemifield', palette=hemi_palette,
    fig_title="1. Metrics by Hemifield [covariate corrected]",
    filename="Fig1_Metrics_Hemifield_Combined.png",
    hue_order=['inPF', 'outPF'],
    custom_limits=resid_limits
)

# =============================================================================
# FIG 2: Metrics by OriGroup
# =============================================================================
plot_binned_grid(
    df, hue_col='OriGroup', palette=ori_group_palette,
    fig_title="2. Metrics by OriGroup [covariate corrected]",
    filename="Fig2_Metrics_OriGroup_Combined.png",
    hue_order=['30_150', '60_120'],
    custom_limits=resid_limits
)

# =============================================================================
# FIG 3: Metrics by Hemisphere Stimulated (OriCat45)
# =============================================================================
plot_binned_grid_hemi(
    df, hue_col='OriCat45', palette=ori_cat45_palette,
    fig_title="3. Metrics by Hemisphere Stimulated [covariate corrected]",
    filename="Fig3_Metrics_OriCat45_Hemi_Combined.png",
    hue_order=['Acute (30+60)', 'Obtuse (120+150)'],
    custom_limits=resid_limits
)

# =============================================================================
# FIG 3c: Diagonal Bias Focus
# =============================================================================
plot_diagonal_bias_grid(
    df,
    fig_title="3c. Diagonal Bias Focus | Ori Groups [covariate corrected]",
    filename="Fig3c_DiagonalBias_Combined.png",
    custom_limits={'BiasTowardsDiagonal_resid': (-2, 6)}
)

# =============================================================================
# FIG 4: Metrics by Diagonal Proximity (inPF vs outPF)
# =============================================================================
plot_binned_grid_inpf_outpf(
    df, hue_col='DiagType', palette=diag_palette,
    fig_title="4. Metrics by Diagonal Proximity | Split by inPF vs outPF [covariate corrected]",
    filename="Fig4_Metrics_DiagType_Combined.png",
    hue_order=['nearDiag', 'offDiag'],
    custom_limits=resid_limits
)

print("Done! Main figures saved.\n")

# =============================================================================
# FIG 5: Polar Plot of inPF - outPF Difference by OriCat
# =============================================================================
plot_polar_diff_oricat(
    df, 
    fig_title="5. Within-Subject Difference (TMS_inPF - TMS_outPF) by Grating Orientation",
    filename="Fig05_PolarDiff_OriCat.png"
)

print("Done! Fig05 saved.\n")

# =============================================================================
# FIG 6: Metrics by Diagonal Proximity (inPF vs outPF) SPLIT by Offset Size
# =============================================================================
plot_binned_grid_inpf_outpf_split(
    df, hue_col='DiagType', palette=diag_palette,
    fig_title="6. Metrics by Diagonal Proximity | Split by Offset Size [covariate corrected]",
    filename="Fig6_Metrics_DiagType_OffsetSplit.png",
    hue_order=['nearDiag', 'offDiag'],
    bin_col='OffsetGroup',
    custom_limits=resid_limits
)

print("Done! Fig6 Offset Split saved.\n")


