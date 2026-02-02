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

warnings.filterwarnings('ignore')
# User Request: No grids
sns.set_theme(style="white", context="notebook")
# Output Dir
output_dir = os.path.join(gm.p['datd'], 'Figures', 'exploratory')
os.makedirs(output_dir, exist_ok=True)


print("Loading data...")
df = gm.master_df.copy()
df.reset_index(drop=True, inplace=True)

# --- Feature Engineering ---
df.rename(columns={
    'TMScond_label': 'TMS',        
    'stimPF_label': 'Hemifield',   
    'targOriCategory': 'OriCat',
    'hemisphereStimulated': 'HemiStim'
}, inplace=True)

# User Request: Use Signed InitOffset (not Abs)
# df['AdjustmentAbs'] = np.abs(df['initOffset']) # REMOVED
df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1
df['AbsError'] = np.abs(df['errOri_scaled'])

# Type Conversion
df['TMS'] = df['TMS'].astype('category')
df['Hemifield'] = df['Hemifield'].astype('category')
df['OriCat'] = df['OriCat'].astype(int) # explicit int for sorting
df['TMS'] = df['TMS'].cat.reorder_categories(['No TMS', 'Early TMS', 'Middle TMS'], ordered=True)
df['Hemifield'] = df['Hemifield'].cat.reorder_categories(['outPF', 'inPF'], ordered=True)

# --- Reusable Plotting Function ---
def plot_adjustment_grid(data, hue_col, palette, fig_title, filename, hue_order=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('errOri_scaled', 'Signed Bias'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction')
    ]
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']
    
    if hue_order:
        unique_hues = hue_order
    else:
        unique_hues = np.sort(data[hue_col].unique())

    # Generate colors manually if palette is a string
    if isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(unique_hues))
        hue_map = dict(zip(unique_hues, colors))
    else:
        hue_map = palette

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    fig.suptitle(fig_title, fontsize=16)

    for r, (met, label) in enumerate(metrics):
        for c, tms in enumerate(tms_conds):
            ax = axes[r, c]
            subset = data[data['TMS'] == tms]
            
            for hue_val in unique_hues:
                hue_data = subset[subset[hue_col] == hue_val]
                if len(hue_data) < 5: continue
                
                # Check for NaNs
                clean = hue_data[['initOffset', met]].dropna()
                if len(clean) < 5: continue

                # Calc Stats (Quadratic)
                coeffs = np.polyfit(clean['initOffset'], clean[met], 2)
                p_poly = np.poly1d(coeffs)
                y_pred = p_poly(clean['initOffset'])
                y_true = clean[met]
                
                # R-squared
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                # Label: Always show R2 as requested
                lbl = f"{hue_val}: $R^2$={r2:.3f}"

                sns.regplot(
                    data=hue_data, x='initOffset', y=met,
                    ax=ax, color=hue_map[hue_val],
                    scatter_kws={'alpha': 0.1, 's': 5},
                    line_kws={'alpha': 0.8, 'label': lbl},
                    ci=None, # Disable CI for speed/clarity with many lines
                    order=2  # Quadratic Fit
                )
            
            ax.set_title(f"{label} | {tms}")
            if r == 3: ax.set_xlabel("Signed Init Offset (deg)")
            else: ax.set_xlabel("")
            if c == 0: ax.set_ylabel(label)
            else: ax.set_ylabel("")
            
            # Legend on every subplot to show R2 stats
            ax.legend(fontsize='x-small', loc='best')

            # User specific Y-limits
            if met == 'AbsError':
                ax.set_ylim(0, 20)
            elif met == 'errOri_scaled':
                ax.set_ylim(-20, 20)
            elif met == 'RT':
                ax.set_ylim(0.5, 3)
            elif met == 'BiasTowardsDiagonal':
                ax.set_ylim(-20, 20)
    
    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    return fig

import modeller as md

# --- New: Binned Bar Plot Function ---
def plot_binned_grid(data, hue_col, palette, fig_title, filename, bin_col='OffsetBin', hue_order=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('errOri_scaled', 'Signed Bias'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
        ('Precision', 'Precision (K)'),
        ('GuessRate', 'Guess Rate (g)')
    ]
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']

    fig, axes = plt.subplots(6, 3, figsize=(18, 27)) # 6 Rows now
    fig.suptitle(fig_title, fontsize=16)

    # Pre-calculate Mixture stats per SUBJECT per bin
    print("Fitting Mixture Models per SUBJECT (this may take a while)...")
    mixture_stats = []
    
    # Iterate through all combinations
    bins = data[bin_col].unique()
    hues = data[hue_col].unique()
    subjects = data['subjID'].unique()
    
    for tms in tms_conds:
        for b in bins:
            for h in hues:
                for subj in subjects:
                    subset = data[
                        (data['TMS'] == tms) & 
                        (data[bin_col] == b) & 
                        (data[hue_col] == h) &
                        (data['subjID'] == subj)
                    ]
                    
                    # Minimum trials for fit (per subject)
                    if len(subset) < 10:
                        # Skip or append NaN? safer to skip for barplot
                        continue
                    
                    # Fit
                    angles = subset['errOri_scaled'].dropna().values
                    try:
                        res = md.fit_model(angles, 'mixtureModel')
                        g, K = res
                        # Bounds check
                        if g < 0 or g > 1 or K < 0 or K > 50: # Sanity check
                             g, K = np.nan, np.nan
                    except:
                        g, K = np.nan, np.nan
                    
                    if not np.isnan(K):
                        mixture_stats.append({
                            'TMS': tms, 'OffsetBin': b, hue_col: h, 'subjID': subj,
                            'Precision': K, 'GuessRate': g
                        })
    
    df_mix = pd.DataFrame(mixture_stats)

    for r, (met, label) in enumerate(metrics):
        for c, tms in enumerate(tms_conds):
            ax = axes[r, c]
            
            # Determine source data
            if met in ['Precision', 'GuessRate']:
                plot_data = df_mix[df_mix['TMS'] == tms]
                # Now we DO have subject-level data, so errorbar='se' works!
                sns.barplot(
                    data=plot_data, x='OffsetBin', y=met, hue=hue_col,
                    palette=palette, errorbar='se', ax=ax,
                    capsize=0.1, errwidth=1.5, hue_order=hue_order
                )
            else:
                plot_data = data[data['TMS'] == tms]
                sns.barplot(
                    data=plot_data, x=bin_col, y=met, hue=hue_col,
                    palette=palette, errorbar='se', ax=ax,
                    capsize=0.1, errwidth=1.5, hue_order=hue_order
                )
            
            ax.set_title(f"{label} | {tms}")
            if r == 5: 
                ax.set_xlabel("Init Offset Bins")
                ax.tick_params(axis='x', rotation=45, labelsize='small')
            else: 
                ax.set_xlabel("")
                ax.set_xticklabels([])
            
            if c == 0: ax.set_ylabel(label)
            else: ax.set_ylabel("")

            # Y-limits
            if met == 'AbsError':
                ax.set_ylim(0, 12)
            elif met == 'errOri_scaled':
                ax.set_ylim(-7, 7)
                ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
            elif met == 'RT':
                ax.set_ylim(1.0, 2.5)
            elif met == 'BiasTowardsDiagonal':
                ax.set_ylim(-2, 8)
                ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
            elif met == 'Precision':
                ax.set_ylim(0, 25) 
            elif met == 'GuessRate':
                ax.set_ylim(-0.1, 0.1)

            # Smart Legend
            if r == 0 and c == 0:
                ax.legend(fontsize='x-small', loc='upper right', title=hue_col)
            elif ax.get_legend():
                ax.get_legend().remove()

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close() 
    return fig

# --- Feature Engineering: Binning ---
# User Request: [-90, -60, -30, 0, 30, 60, 90]
bins = [-90, -60, -30, 0, 30, 60, 90]
bin_labels = ["-90:-60", "-60:-30", "-30:0", "0:30", "30:60", "60:90"]
df['OffsetBin'] = pd.cut(df['initOffset'], bins=bins, labels=bin_labels)

# User Request: Abs Offset Bins [0, 30, 60, 90]
df['AbsInitOffset'] = np.abs(df['initOffset'])
abs_bins = [0, 30, 60, 90]
abs_labels = ["0:30", "30:60", "60:90"]
df['AbsOffsetBin'] = pd.cut(df['AbsInitOffset'], bins=abs_bins, labels=abs_labels, include_lowest=True)

# User Request: Palette Update
hemi_palette = {'inPF': '#e91e63', 'outPF': '#ff9800'} # Pinkish-red, Orange

# --- New: Aggregate Bar Plot (Figure 5) ---
def plot_aggregate_bars(data, hue_col, palette, fig_title, filename, hue_order=None):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('errOri_scaled', 'Signed Bias'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction'),
        ('Precision', 'Precision (K)'),
        ('GuessRate', 'Guess Rate (g)')
    ]
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']
    
    # 2 Plots x 3 Cols
    fig, axes = plt.subplots(2, 3, figsize=(18, 12)) 
    fig.suptitle(fig_title, fontsize=16)

    # 1. Pre-calculate Mixture stats (Collapsed across bins) per Subject
    print("Fitting Mixture Models per SUBJECT (Aggregated) ...")
    mixture_stats = []
    
    hues = data[hue_col].unique()
    subjects = data['subjID'].unique()
    
    for tms in tms_conds:
        for h in hues:
            for subj in subjects:
                subset = data[
                    (data['TMS'] == tms) & 
                    (data[hue_col] == h) &
                    (data['subjID'] == subj)
                ]
                
                # Minimum trials
                if len(subset) < 10: continue
                
                # Fit
                angles = subset['errOri_scaled'].dropna().values
                try:
                    res = md.fit_model(angles, 'mixtureModel')
                    g, K = res
                    if g < 0 or g > 1 or K < 0 or K > 100:
                            g, K = np.nan, np.nan
                except:
                    g, K = np.nan, np.nan
                
                if not np.isnan(K):
                    mixture_stats.append({
                        'TMS': tms, hue_col: h, 'subjID': subj,
                        'Precision': K, 'GuessRate': g
                    })
    
    df_mix = pd.DataFrame(mixture_stats)
    
    # 2. Pre-calculate Standard Metrics per Subject (Groupby)
    # We want one value per subject per condition
    # Group by TMS, Hemi, Subject -> Mean
    df_std = data.groupby(['TMS', hue_col, 'subjID'], observed=False)[['AbsError', 'errOri_scaled', 'RT', 'BiasTowardsDiagonal']].mean().reset_index()

    # Flatten axes for iteration
    axes_flat = axes.flatten()

    for i, (met, label) in enumerate(metrics):
        ax = axes_flat[i]
        
        # Determine source
        if met in ['Precision', 'GuessRate']:
            plot_data = df_mix
        else:
            plot_data = df_std
        
        sns.barplot(
            data=plot_data, x='TMS', y=met, hue=hue_col,
            palette=palette, errorbar='se', ax=ax,
            capsize=0.1, errwidth=1.5, hue_order=hue_order
        )
        
        ax.set_title(label)
        ax.set_xlabel("")
        if i % 3 == 0: ax.set_ylabel(label)
        else: ax.set_ylabel("")
        
        # Y-limits specifically for this aggregate plot?
        if met == 'AbsError': ax.set_ylim(0, 15)
        elif met == 'errOri_scaled': ax.set_ylim(-5, 5); ax.axhline(0, color='k', lw=0.5)
        elif met == 'RT': ax.set_ylim(1, 2.5)
        elif met == 'BiasTowardsDiagonal': ax.set_ylim(-1, 5); ax.axhline(0, color='k', lw=0.5)
        elif met == 'Precision': ax.set_ylim(0, 25)
        elif met == 'GuessRate': ax.set_ylim(-0.1, 0.1)

        # Legend
        if i == 0:
            ax.legend(title=hue_col, loc='upper left', bbox_to_anchor=(1, 1))
        else:
             if ax.get_legend(): ax.get_legend().remove()

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return fig

# --- Output Figures ---

# =============================================================================
# FIGURE 1: Hue = Hemifield (Scatter)
# =============================================================================
plot_adjustment_grid(
    df, 
    hue_col='Hemifield', 
    palette=hemi_palette, 
    fig_title="1. Impact of Signed Adjustment Distance (Hue: Hemifield)",
    filename="Fig1_Adjustment_Hemifield.png",
    hue_order=['inPF', 'outPF']
)

# =============================================================================
# FIGURE 2: Hue = OriCat (Scatter)
# =============================================================================
# ... (Keep existing Figure 2 code) ...
ori_palette = {
    30:  '#1f77b4', # Blue
    60:  '#2ca02c', # Green
    120: '#ff7f0e', # Orange
    150: '#d62728'  # Red
}

plot_adjustment_grid(
    df, 
    hue_col='OriCat', 
    palette=ori_palette, 
    fig_title="2. Impact of Signed Adjustment Distance (Hue: OriCat)",
    filename="Fig2_Adjustment_OriCat.png",
    hue_order=[30, 60, 120, 150]
)

# =============================================================================
# FIGURE 3: Binned Bars (Hue: Hemifield)
# =============================================================================
plot_binned_grid(
    df,
    hue_col='Hemifield',
    palette=hemi_palette, 
    fig_title="3. Binned Adjustment Impact (Hue: Hemifield)",
    filename="Fig3_Binned_Hemifield.png",
    hue_order=['inPF', 'outPF']
)

# =============================================================================
# FIGURE 4: Binned Bars (Hue: OriCat)
# =============================================================================
plot_binned_grid(
    df,
    hue_col='OriCat',
    palette=ori_palette,
    fig_title="4. Binned Adjustment Impact (Hue: OriCat)",
    filename="Fig4_Binned_OriCat.png",
    hue_order=[30, 60, 120, 150]
)

# =============================================================================
# FIGURE 5: Binned Abs(InitOffset) (Hue: Hemifield)
# =============================================================================
plot_binned_grid(
    df,
    hue_col='Hemifield',
    palette=hemi_palette, 
    fig_title="5. Binned Abs(Offset) Impact (Hue: Hemifield)",
    filename="Fig5_Binned_Abs_Hemifield.png",
    bin_col='AbsOffsetBin',
    hue_order=['inPF', 'outPF']
)

# =============================================================================
# FIGURE 6: Aggregate Bars (Hue: Hemifield)
# =============================================================================
plot_aggregate_bars(
    df,
    hue_col='Hemifield',
    palette=hemi_palette,
    fig_title="6. Aggregate TMS Impact (Hue: Hemifield)",
    filename="Fig6_Aggregate_TMS_Hemi.png",
    hue_order=['inPF', 'outPF']
)

print("Done! Figures saved.")

