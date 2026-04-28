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
output_dir = os.path.join(gm.p['datd'], 'Figures', 'visualbiases')
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

def plot_ori_bins(data, palette, fig_title, filename):
    print(f"Generating Figure: {fig_title}...")
    
    # Pre-aggregate per subject to get Means/Counts per bin
    # This ensures we plot the distribution of SUBJECT Means, not trial-level variance
    print("Aggregating per subject/bin...")
    
    # We need trial counts, so we can use 'size'
    agg_funcs = {
        'BiasTowardsDiagonal': 'mean',
        'AbsError': 'mean',
        'subjID': 'size' # This becomes the Trial Count
    }
    
    df_agg = data.groupby(['subjID', 'TMS_Group', 'OffsetGroup', 'OriBinCenter'], observed=False).agg(agg_funcs).rename(columns={'subjID': 'TrialCount'}).reset_index()
    
    # Metrics to plot
    metrics = [
        ('TrialCount', 'Trial Count'),
        ('BiasTowardsDiagonal', 'Diag Attraction'),
        ('AbsError', 'Abs Error')
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
            
            if met == 'BiasTowardsDiagonal':
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
        ('AbsError', 'Abs Error'), 
        ('errOri_scaled', 'Signed Bias'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction')
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

            if met == 'AbsError': ax.set_ylim(0, 20)
            elif met == 'errOri_scaled': ax.set_ylim(-20, 20)
            elif met == 'RT': ax.set_ylim(0.5, 3)
            elif met == 'BiasTowardsDiagonal': ax.set_ylim(-20, 20)
    
    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close() 

def plot_binned_grid(data, hue_col, palette, fig_title, filename, hue_order=None, custom_limits=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
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
                
                angles = subset['errOri_scaled'].dropna().values
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
            if met == 'AbsError': ax.set_ylim(0, 12)
            elif met == 'errOri_scaled': ax.set_ylim(-7, 7); ax.axhline(0, color='k', lw=0.5)
            elif met == 'RT': ax.set_ylim(1.0, 2.5)
            elif met == 'BiasTowardsDiagonal': ax.set_ylim(-0.5, 4); ax.axhline(0, color='k', lw=0.5)
            elif met == 'Precision': ax.set_ylim(0, 25) 
            elif met == 'GuessRate': ax.set_ylim(-0.1, 0.1)

        if met == 'BiasTowardsDiagonal':
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
        ('AbsError', 'Abs Error'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
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
                
                angles = subset['errOri_scaled'].dropna().values
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
            if met == 'AbsError': ax.set_ylim(0, 12)
            elif met == 'RT': ax.set_ylim(1.0, 2.5)
            elif met == 'BiasTowardsDiagonal': ax.set_ylim(-0.5, 4); ax.axhline(0, color='k', lw=0.5)
            elif met == 'Precision': ax.set_ylim(0, 25) 

        if met == 'BiasTowardsDiagonal':
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
        ('AbsError', 'Abs Error'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
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
                
                angles = subset['errOri_scaled'].dropna().values
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
            if met == 'AbsError': ax.set_ylim(0, 12)
            elif met == 'RT': ax.set_ylim(1.0, 2.5)
            elif met == 'BiasTowardsDiagonal': ax.set_ylim(-0.5, 4); ax.axhline(0, color='k', lw=0.5)
            elif met == 'Precision': ax.set_ylim(0, 25) 

        if met == 'BiasTowardsDiagonal':
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
        ('AbsError', 'Abs Error'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
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
                    
                    angles = subset['errOri_scaled'].dropna().values
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
                if met == 'AbsError': ax.set_ylim(0, 12)
                elif met == 'RT': ax.set_ylim(1.0, 2.5)
                elif met == 'BiasTowardsDiagonal': ax.set_ylim(-0.5, 4); ax.axhline(0, color='k', lw=0.5)
                elif met == 'Precision': ax.set_ylim(0, 25) 

            if met == 'BiasTowardsDiagonal':
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
            data=plot_data, x='TMS_Type', y='BiasTowardsDiagonal', hue='VisualField',
            palette=vf_palette, errorbar='se', ax=ax,
            capsize=0.1, errwidth=1.5, hue_order=['LeftVF', 'RightVF'], order=x_order
        )
        
        ax.set_title(f"{ori_label}")
        ax.set_xlabel("TMS Condition")
        
        if i == 0: 
            ax.set_ylabel("Diagonal Attraction")
        else: 
            ax.set_ylabel("")

        if custom_limits and 'BiasTowardsDiagonal' in custom_limits:
            ax.set_ylim(custom_limits['BiasTowardsDiagonal'])
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

# --- PLOTTING CALLS ---

hemi_palette = {'inPF': '#e91e63', 'outPF': '#ff9800'}
ori_palette = {30: '#1f77b4', 60: '#2ca02c', 120: '#ff7f0e', 150: '#d62728'}
ori_group_palette = {'30_150': '#009688', '60_120': '#9c27b0'}
ori_cat45_palette = {'Acute (30+60)': '#3f51b5', 'Obtuse (120+150)': '#e91e63'}
diag_palette = {'Radial': '#E65100', 'Tangential': '#00B8D4'}

# Fig 1: Binned (Hue: Hemi)
limits_fig1 = {
    'AbsError': (4, 8.5),
    'RT': (1.0, 2.0),
    'BiasTowardsDiagonal': (-0.1, 4),
    'Precision': (5, 20)
}
plot_binned_grid(
    df, hue_col='Hemifield', palette=hemi_palette,
    fig_title="1. Metrics by Hemifield (Combined Offsets)",
    filename="Fig1_Metrics_Hemifield_Combined.png",
    hue_order=['inPF', 'outPF'],
    custom_limits=limits_fig1
)

# Fig 2: Binned (Hue: DiagType=Radial/Tangential) split by Hemisphere Stimulated
limits_fig2 = {
    'AbsError': (4, 8.5),
    'RT': (1.0, 2.1),
    'BiasTowardsDiagonal': (-0.1, 5),
    'Precision': (5, 20)
}
plot_binned_grid_hemi(
    df, hue_col='DiagType', palette=diag_palette,
    fig_title="2. Metrics by Hemisphere Stimulated | Radial vs Tangential",
    filename="Fig2_Metrics_DiagType_Hemi_Combined.png",
    hue_order=['Radial', 'Tangential'],
    custom_limits=limits_fig2
)

# Fig 3: Diagonal Bias Focus
plot_diagonal_bias_grid(
    df,
    fig_title="3. Diagonal Bias Focus | Ori Groups (Combined Offsets)",
    filename="Fig3_DiagonalBias_Combined.png",
    custom_limits={'BiasTowardsDiagonal': (-1, 6)}
)


# Fig 4: Binned (Hue: DiagType)
limits_fig4 = {
    'AbsError': (4, 8.5),
    'RT': (1.0, 2.1),
    'BiasTowardsDiagonal': (-0.1, 5),
    'Precision': (5, 20)
}
plot_binned_grid_inpf_outpf(
    df, hue_col='DiagType', palette=diag_palette,
    fig_title="4. Metrics by Diagonal Proximity | Split by inPF vs outPF",
    filename="Fig4_Metrics_DiagType_Combined.png",
    hue_order=['Radial', 'Tangential'],
    custom_limits=limits_fig4
)

# Fig 5: Binned (Hue: DiagType) Split by Offset
plot_binned_grid_inpf_outpf_split(
    df, hue_col='DiagType', palette=diag_palette,
    fig_title="5. Metrics by Diagonal Proximity | Split by inPF vs outPF & Offset",
    filename="Fig5_Metrics_DiagType_SplitOffset.png",
    hue_order=['Radial', 'Tangential'],
    bin_col='OffsetGroup',
    custom_limits=limits_fig4
)

# Fig 6: Radial/Tangential Contrast Index = (R - T) / (R + T) per subject
# ---------------------------------------------------------------------------
# Permutation tests: (1) one-sample sign test vs 0 per bar (Bonferroni x3),
# (2) omnibus within-subject permutation F-test across conditions,
# (3) pairwise post-hoc sign-permutation t-tests (Bonferroni x3) if sig.
# ---------------------------------------------------------------------------

_N_PERM_FIG6 = 5000
_RNG_FIG6 = np.random.default_rng(42)

def _perm_one_sample(vals, n_perm=_N_PERM_FIG6):
    """Sign-permutation test: mean of vals vs 0 (two-tailed)."""
    obs = np.mean(vals)
    null = np.array([
        np.mean(vals * _RNG_FIG6.choice([-1, 1], size=len(vals)))
        for _ in range(n_perm)
    ])
    return obs, np.mean(np.abs(null) >= np.abs(obs))

def _perm_f_fig6(mat, n_perm=_N_PERM_FIG6):
    """Within-subject permutation F-test; mat shape (n_subj, n_cond)."""
    def f_stat(m):
        n_s, n_c = m.shape
        gm = m.mean()
        ss_c = n_s * np.sum((m.mean(axis=0) - gm) ** 2)
        ss_s = n_c * np.sum((m.mean(axis=1) - gm) ** 2)
        ss_e = np.sum((m - gm) ** 2) - ss_c - ss_s
        ms_c = ss_c / (n_c - 1)
        ms_e = ss_e / ((n_s - 1) * (n_c - 1))
        return ms_c / ms_e if ms_e > 0 else np.nan
    obs_f = f_stat(mat)
    null_f = np.zeros(n_perm)
    for k in range(n_perm):
        s = mat.copy()
        for r in range(mat.shape[0]):
            s[r] = _RNG_FIG6.permutation(mat[r])
        null_f[k] = f_stat(s)
    return obs_f, np.mean(null_f >= obs_f)

def _perm_paired_t_fig6(v1, v2, n_perm=_N_PERM_FIG6):
    """Paired sign-permutation t-test (two-tailed)."""
    diff = v1 - v2
    obs_t = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
    null_t = np.zeros(n_perm)
    for k in range(n_perm):
        d = diff * _RNG_FIG6.choice([-1, 1], size=len(diff))
        null_t[k] = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d)))
    return obs_t, np.mean(np.abs(null_t) >= np.abs(obs_t))

def _stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

def _add_bracket(ax, x1, x2, y, label, lw=1.2):
    if not label:
        return
    ylim = ax.get_ylim()
    h = (ylim[1] - ylim[0]) * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c='k', clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.1, label, ha='center', va='bottom', fontsize=11, clip_on=False)

def plot_radial_tangential_index(data, fig_title, filename, custom_limits=None):
    print(f"Generating Figure: {fig_title}...")

    data = data.copy()

    def make_tms_inpf_cat(row):
        if row['TMS_Group'] == 'No TMS':
            return 'NoTMS'
        return f"TMS_{row['Hemifield']}"

    data['TMS_Condition'] = data.apply(make_tms_inpf_cat, axis=1)
    x_order = ['NoTMS', 'TMS_inPF', 'TMS_outPF']
    data['TMS_Condition'] = pd.Categorical(data['TMS_Condition'], categories=x_order, ordered=True)

    # --- Fit mixture models for Precision per subject x condition x DiagType ---
    print("  Fitting Mixture Models for Precision index...")
    mixture_stats = []
    subjects = data['subjID'].unique()
    for cond in x_order:
        for diag in ['Radial', 'Tangential']:
            for subj in subjects:
                subset = data[
                    (data['TMS_Condition'] == cond) &
                    (data['DiagType'] == diag) &
                    (data['subjID'] == subj)
                ]
                if len(subset) < 10:
                    continue
                angles = subset['errOri_scaled'].dropna().values
                try:
                    res = md.fit_model(angles, 'mixtureModel')
                    g, K = res
                    if g < 0 or g > 1 or K < 0 or K > 50:
                        g, K = np.nan, np.nan
                except:
                    g, K = np.nan, np.nan
                if not np.isnan(K):
                    mixture_stats.append({
                        'TMS_Condition': cond, 'DiagType': diag,
                        'subjID': subj, 'Precision': K
                    })

    df_mix = pd.DataFrame(mixture_stats) if mixture_stats else pd.DataFrame()
    if not df_mix.empty:
        df_mix['TMS_Condition'] = pd.Categorical(
            df_mix['TMS_Condition'], categories=x_order, ordered=True
        )

    # --- Per-subject means for non-Precision metrics ---
    subj_means = (
        data.groupby(['subjID', 'TMS_Condition', 'DiagType'])[
            ['AbsError', 'RT', 'BiasTowardsDiagonal']
        ].mean().reset_index()
    )

    # --- Contrast index function ---
    def compute_index(df_in, value_col):
        """Returns (R - T) / (R + T) per subject x condition."""
        pivot = df_in.pivot_table(
            index=['subjID', 'TMS_Condition'],
            columns='DiagType',
            values=value_col
        ).reset_index()
        pivot.columns.name = None
        if 'Radial' not in pivot.columns or 'Tangential' not in pivot.columns:
            return pd.DataFrame()
        pivot = pivot.dropna(subset=['Radial', 'Tangential'])
        denom = pivot['Radial'] + pivot['Tangential']
        pivot['IndexVal'] = np.where(
            denom != 0,
            (pivot['Radial'] - pivot['Tangential']) / denom,
            np.nan
        )
        return pivot[['subjID', 'TMS_Condition', 'IndexVal']].dropna()

    metrics = [
        ('AbsError', 'Abs Error Index'),
        ('RT', 'RT Index'),
        ('BiasTowardsDiagonal', 'Diag Attraction Index'),
        ('Precision', 'Precision Index'),
    ]

    cond_palette = {
        'NoTMS': '#607D8B',
        'TMS_inPF': '#E91E63',
        'TMS_outPF': '#FF9800'
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.suptitle(fig_title, fontsize=16)

    pairs     = [('NoTMS', 'TMS_inPF'), ('NoTMS', 'TMS_outPF'), ('TMS_inPF', 'TMS_outPF')]
    pair_xpos = [(0, 1), (0, 2), (1, 2)]
    print("  Running pairwise permutation tests (all pairs, Bonferroni ×3)...")

    for i, (met, label) in enumerate(metrics):
        ax = axes[i]

        if met == 'Precision':
            idx_df = compute_index(df_mix, 'Precision') if not df_mix.empty else pd.DataFrame()
        else:
            idx_df = compute_index(subj_means, met)

        if idx_df.empty:
            ax.set_title(f"{label} (no data)")
            continue

        sns.barplot(
            data=idx_df, x='TMS_Condition', y='IndexVal',
            palette=cond_palette,
            errorbar='se', ax=ax, capsize=0.1, errwidth=1.5, order=x_order
        )
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(label)

        # Flush draw so ylim is current before bracket placement
        ax.figure.canvas.draw()
        if custom_limits and met in custom_limits:
            ax.set_ylim(custom_limits[met])
            ax.figure.canvas.draw()

        # ── Pairwise permutation t-tests — ALWAYS shown for all 3 pairs ──────
        try:
            piv = idx_df.pivot(
                index='subjID', columns='TMS_Condition', values='IndexVal'
            )[x_order].dropna()

            ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            # Stack brackets above the plot (step upward per pair)
            bracket_base = ax.get_ylim()[1] + ylim_range * 0.03
            bracket_step  = ylim_range * 0.14

            for bi, ((c1, c2), (x1, x2)) in enumerate(zip(pairs, pair_xpos)):
                _, p_raw = _perm_paired_t_fig6(piv[c1].values, piv[c2].values)
                print(f"    - {met}: {c1} vs {c2} | p={p_raw:.4f}")
                # Uncorrected pairwise p-value
                annot = _stars(p_raw) if _stars(p_raw) else 'ns'
                _add_bracket(ax, x1, x2,
                             bracket_base + bi * bracket_step, annot)

            # Expand ylim so brackets aren't clipped
            new_top = bracket_base + len(pairs) * bracket_step + ylim_range * 0.05
            ax.set_ylim(ax.get_ylim()[0], new_top)

        except Exception as e:
            ax.set_title(f"{label}  (stats err: {e})", fontsize=9)

        if i >= 2:
            ax.set_xlabel("TMS Condition")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        ax.set_ylabel("(Radial − Tang) / (Radial + Tang)")

        if custom_limits and met in custom_limits:
            # Only re-apply floor; top already expanded for brackets
            current_top = ax.get_ylim()[1]
            ax.set_ylim(custom_limits[met][0], max(current_top, custom_limits[met][1]))




    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"  Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_radial_tangential_index_split(data, fig_title, filename, split_col='OffsetGroup', custom_limits=None):
    print(f"Generating Figure: {fig_title}...")

    data = data.copy()
    def make_tms_cond(row):
        if row['TMS_Group'] == 'No TMS': return 'NoTMS'
        return f"TMS_{row['Hemifield']}"
    data['TMS_Condition'] = data.apply(make_tms_cond, axis=1)
    x_order = ['NoTMS', 'TMS_inPF', 'TMS_outPF']
    
    if data[split_col].dtype.name == 'category':
        split_levels = list(data[split_col].cat.categories)
    else:
        split_levels = sorted(list(data[split_col].unique()))

    # --- Mixture Models for Precision ---
    print("  Fitting Mixture Models (split)...")
    mix_stats = []
    subjects = data['subjID'].unique()
    for lev in split_levels:
        for cond in x_order:
            for diag in ['Radial', 'Tangential']:
                for subj in subjects:
                    subset = data[(data[split_col]==lev) & (data['TMS_Condition']==cond) & 
                                  (data['DiagType']==diag) & (data['subjID']==subj)]
                    if len(subset) < 8: continue
                    angles = subset['errOri_scaled'].dropna().values
                    try:
                        res = md.fit_model(angles, 'mixtureModel')
                        if 0 <= res[0] <= 1 and 0 <= res[1] <= 50:
                            mix_stats.append({split_col: lev, 'TMS_Condition': cond, 'DiagType': diag, 
                                             'subjID': subj, 'Precision': res[1]})
                    except: pass
    df_mix = pd.DataFrame(mix_stats)

    # --- Means for other metrics ---
    subj_means = data.groupby(['subjID', split_col, 'TMS_Condition', 'DiagType'])[
        ['AbsError', 'RT', 'BiasTowardsDiagonal']
    ].mean().reset_index()

    def compute_index_split(df_in, val_col):
        piv = df_in.pivot_table(index=['subjID', split_col, 'TMS_Condition'], 
                                columns='DiagType', values=val_col).reset_index()
        if 'Radial' not in piv.columns or 'Tangential' not in piv.columns: return pd.DataFrame()
        piv = piv.dropna(subset=['Radial', 'Tangential'])
        piv['IndexVal'] = (piv['Radial'] - piv['Tangential']) / (piv['Radial'] + piv['Tangential'])
        return piv

    metrics = [('AbsError', 'Abs Error'), ('RT', 'RT'), 
               ('BiasTowardsDiagonal', 'Diag Attraction'), ('Precision', 'Precision')]
    
    cond_palette = {'NoTMS': '#607D8B', 'TMS_inPF': '#E91E63', 'TMS_outPF': '#FF9800'}
    pairs = [('NoTMS', 'TMS_inPF'), ('NoTMS', 'TMS_outPF'), ('TMS_inPF', 'TMS_outPF')]
    pair_x = [(0,1), (0,2), (1,2)]

    fig, axes = plt.subplots(len(metrics), len(split_levels), figsize=(5*len(split_levels), 4*len(metrics)))
    fig.suptitle(fig_title, fontsize=16, y=0.99)

    for row_idx, (met, label) in enumerate(metrics):
        idx_df = compute_index_split(df_mix if met=='Precision' else subj_means, met)
        
        for col_idx, lev in enumerate(split_levels):
            ax = axes[row_idx, col_idx]
            sub = idx_df[idx_df[split_col] == lev]
            
            if sub.empty:
                ax.set_title(f"{label} - {lev} (no data)")
                continue

            sns.barplot(data=sub, x='TMS_Condition', y='IndexVal', palette=cond_palette,
                        errorbar='se', ax=ax, capsize=0.1, order=x_order)
            ax.axhline(0, color='k', lw=0.8, ls='--')
            
            if row_idx == 0: ax.set_title(f"{lev}\n{label}")
            else: ax.set_title(label)

            # Pairwise Stats
            try:
                piv_stats = sub.pivot(index='subjID', columns='TMS_Condition', values='IndexVal')[x_order].dropna()
                ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                b_base = ax.get_ylim()[1] + ylim_range * 0.05
                b_step = ylim_range * 0.15
                
                for bi, ((c1, c2), (x1, x2)) in enumerate(zip(pairs, pair_x)):
                    _, p = _perm_paired_t_fig6(piv_stats[c1].values, piv_stats[c2].values)
                    print(f"    - {met} ({lev}): {c1} vs {c2} | p={p:.4f}")
                    annot = _stars(p) if _stars(p) else 'ns'
                    _add_bracket(ax, x1, x2, b_base + bi*b_step, annot)
                
                ax.set_ylim(ax.get_ylim()[0], b_base + 3*b_step + ylim_range*0.05)
            except: pass

            if row_idx < len(metrics)-1: ax.set_xlabel(""); ax.set_xticklabels([])
            ax.set_ylabel("Index (R-T)/(R+T)")
            if custom_limits and met in custom_limits:
                ax.set_ylim(custom_limits[met][0], max(ax.get_ylim()[1], custom_limits[met][1]))

    sns.despine()
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = os.path.join(output_dir, filename)
    print(f"  Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

plot_radial_tangential_index(
    df,
    fig_title="6. Radial/Tangential Contrast Index | (R − T) / (R + T) per Subject",
    filename="Fig6_RadialTangential_Index.png"
)

plot_radial_tangential_index_split(
    df,
    fig_title="7. Radial/Tangential Contrast Index | Split by Offset Group",
    filename="Fig7_RadialTangential_Index_SplitOffset.png"
)

print("Done! Grouped figures saved.\n")

