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
output_dir = os.path.join(gm.p['datd'], 'Figures', 'motorcertainty')
os.makedirs(output_dir, exist_ok=True)

def get_binary_cw(ori_array):
    """Returns 1 if net movement was CW, 0 if CCW."""
    if len(ori_array) < 2: return 0
    diffs = np.diff(ori_array)
    diffs[diffs > 90] -= 180
    diffs[diffs < -90] += 180
    return 1 if np.sum(diffs) < 0 else 0

print("Loading data...")
df = gm.master_df.copy()
df.reset_index(drop=True, inplace=True)

# User Request (V3 Refinement): Remove LargeError trials
print(f"Total trials before filtering: {len(df)}")
df = df[df['LargeError'] == 0].copy()
print(f"Total trials after dropping LargeError=1: {len(df)}")

# Compute binary motor direction from the knob trajectory
print("Computing motor direction from oriArray...")
df['Binary_CW'] = df['oriArray'].apply(get_binary_cw)

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

# Define 'As Expected' metric BEFORE filtering for Figure 5
def check_expected(row):
    if row['initOffset'] > 0: return 1 if row['Binary_CW'] == 1 else 0
    elif row['initOffset'] < 0: return 1 if row['Binary_CW'] == 0 else 0
    return np.nan

df['IsExpectedDirection'] = df.apply(check_expected, axis=1)
# Create a copy for Figure 5 that won't be filtered by motor direction
df_full = df.copy()

# Offset Group: Small (0-50) vs Large (50-90)
# 'include_lowest=True' makes the first interval inclusive (0 is included)
# bins=[0, 50, 90] -> (0, 50], (50, 90]
df['OffsetGroup'] = pd.cut(df['AbsInitOffset'], bins=[0, 50, 90], labels=['Small Offset', 'Large Offset'], include_lowest=True)

# --- CORRECT DIRECTION FILTER FOR LARGE OFFSETS ---
# For large offsets only, keep trials where subject moved in the direction needed:
#   positive initOffset -> subject moved CW (Binary_CW == 1)
#   negative initOffset -> subject moved CCW (Binary_CW == 0)
small_df = df[df['OffsetGroup'] == 'Small Offset'].copy()
large_correct = df[
    (df['OffsetGroup'] == 'Large Offset') &
    ((
        (df['initOffset'] > 0) & (df['Binary_CW'] == 1)
    ) | (
        (df['initOffset'] < 0) & (df['Binary_CW'] == 0)
    ) | (
        df['initOffset'] == 0  # keep ambiguous
    ))
].copy()
n_large_orig = (df['OffsetGroup'] == 'Large Offset').sum()
print(f"Large Offset: kept {len(large_correct)}/{n_large_orig} correct-direction trials ({len(large_correct)/n_large_orig*100:.1f}%)")
df = pd.concat([small_df, large_correct]).reset_index(drop=True)
print(f"Final dataframe: {len(df)} trials")



# Type Conversion for others
df['Hemifield'] = pd.Categorical(df['Hemifield'], categories=['outPF', 'inPF'], ordered=True)
df['OriCat'] = df['OriCat'].astype(int)

# --- ORIENTATION BINNING ---
# Bins: 0-180 in 5 deg steps
ori_bins = np.arange(0, 185, 5)
df['OriBin'] = pd.cut(df['targOriActual'], bins=ori_bins, include_lowest=True)
df['OriBinCenter'] = df['OriBin'].apply(lambda x: x.mid).astype(float)


# --- Reusable Plotting Functions (Updated for 2 Cols) ---

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

        if custom_limits == 'auto':
            pass
        elif custom_limits and met in custom_limits:
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
    fig, drama = plt.subplots(4, num_cols, figsize=(6*num_cols, 16), squeeze=False)
    fig.suptitle(f"{fig_title}", fontsize=16)
    
    for r, (met, label) in enumerate(metrics):
        for c, off_group in enumerate(offset_groups):
            ax = drama[r, c]
            
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

            if custom_limits == 'auto':
                pass
            elif custom_limits and met in custom_limits:
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

        if custom_limits == 'auto':
            pass
        elif custom_limits and 'BiasTowardsDiagonal' in custom_limits:
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

def plot_motor_accuracy(data, fig_title, filename):
    print(f"Generating Figure: {fig_title}...")
    
    # Bin initOffset into 5-degree bins for plotting (more continuous)
    bins = np.arange(-92.5, 97.5, 5)
    data = data.copy()
    data['OffsetBin'] = pd.cut(data['initOffset'], bins=bins)
    data['OffsetBinCenter'] = data['OffsetBin'].apply(lambda x: x.mid if pd.notnull(x) else np.nan).astype(float)
    
    # Aggregate per subject/bin
    # Using 'observed=False' to potentially keep empty bins, though not strictly needed here
    subj_agg = data.groupby(['subjID', 'TMS_Group', 'OffsetBinCenter'], observed=False)['IsExpectedDirection'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot with error bars representing SEM across subjects
    sns.lineplot(
        data=subj_agg, x='OffsetBinCenter', y='IsExpectedDirection', hue='TMS_Group',
        palette={'No TMS': 'k', 'TMS': 'r'}, marker='o', errorbar='se'
    )
    
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')
    
    plt.title(fig_title, fontsize=15)
    plt.xlabel("Initial Offset relative to Target (°)", fontsize=13)
    plt.ylabel("Fraction of Trials 'As Expected'", fontsize=13)
    plt.ylim(0, 1.05)
    plt.xlim(-95, 95)
    plt.legend(frameon=False)
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

# Fig 2: Binned (Hue: OriGroup)
limits_fig2 = {
    'AbsError': (4, 8.5),
    'RT': (1.0, 2.1),
    'BiasTowardsDiagonal': (-0.1, 5),
    'Precision': (5, 20)
}
plot_binned_grid(
    df, hue_col='OriGroup', palette=ori_group_palette,
    fig_title="2. Metrics by OriGroup (Combined Offsets)",
    filename="Fig2_Metrics_OriGroup_Combined.png",
    hue_order=['30_150', '60_120'],
    custom_limits=limits_fig2
)

# Fig 3: Binned (Hue: OriCat45)
limits_fig3 = {
    'AbsError': (4, 8.5),
    'RT': (1.0, 2.1),
    'BiasTowardsDiagonal': (-0.1, 5),
    'Precision': (5, 20)
}
plot_binned_grid_hemi(
    df, hue_col='OriCat45', palette=ori_cat45_palette,
    fig_title="3. Metrics by Hemisphere Stimulated (Combined Offsets)",
    filename="Fig3_Metrics_OriCat45_Hemi_Combined.png",
    hue_order=['Acute (30+60)', 'Obtuse (120+150)'],
    custom_limits=limits_fig3
)

# Fig 3c: Diagonal Bias Focus
plot_diagonal_bias_grid(
    df,
    fig_title="3c. Diagonal Bias Focus | Ori Groups (Combined Offsets)",
    filename="Fig3c_DiagonalBias_Combined.png",
    custom_limits={'BiasTowardsDiagonal': (-1, 6)}
)

# Fig 4a: DiagType Split by Offset (Small/Large) - Motor Certainty: Large = correct-direction only
diag_palette = {'nearDiag': '#E65100', 'offDiag': '#00B8D4'}
limits_fig4 = {
    'AbsError': (4, 8.5),
    'RT': (1.0, 2.1),
    'BiasTowardsDiagonal': (-0.1, 5),
    'Precision': (5, 20)
}

plot_binned_grid_inpf_outpf_split(
    df, hue_col='DiagType', palette=diag_palette,
    fig_title="4a. Metrics by Diagonal Proximity | Split by inPF vs outPF & Offset\n(Large Offset: correct-direction trials only)",
    filename="Fig4a_Metrics_DiagType_SplitOffset.png",
    hue_order=['nearDiag', 'offDiag'],
    bin_col='OffsetGroup',
    custom_limits=limits_fig4
)

# Fig 4b: Radial vs Tangential | Split by Expectation (Offsets > 75)
df_fig4b = df_full[df_full['AbsInitOffset'] > 75].copy()
df_fig4b['Expectation'] = df_fig4b['IsExpectedDirection'].map({1.0: 'As Expected', 0.0: 'Not As Expected'})

plot_binned_grid_inpf_outpf_split(
    df_fig4b, 
    hue_col='DiagType', 
    palette=diag_palette,
    fig_title="4b. Radial vs Tangential | Split by Motor Expectation (Offsets > 75°)",
    filename="Fig4b_Metrics_DiagType_Expectation_Split.png",
    hue_order=['nearDiag', 'offDiag'],
    bin_col='Expectation',
    custom_limits='auto'
)

# Fig 5: Motor Choice Accuracy (using full unfiltered data)
plot_motor_accuracy(
    df_full,
    fig_title="5. Motor Choice Accuracy: Percentage of Trials Moving Towards Target",
    filename="Fig5_MotorChoice_Accuracy.png"
)

print("Done! Grouped figures saved.\n")
