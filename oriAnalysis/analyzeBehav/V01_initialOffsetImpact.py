
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import generate_behavdf as gm
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')
# User Request: No grids
sns.set_theme(style="white", context="notebook")
# Output Dir
output_dir = "/Users/mrugankdake/Documents/Clayspace/TMS_ori/Figures/exploratory"
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
def plot_adjustment_grid(data, hue_col, palette, fig_title, filename):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('errOri_scaled', 'Signed Bias'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction')
    ]
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']
    
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

# --- New: Binned Bar Plot Function ---
def plot_binned_grid(data, hue_col, palette, fig_title, filename):
    print(f"Generating Figure: {fig_title}...")
    metrics = [
        ('AbsError', 'Abs Error'), 
        ('errOri_scaled', 'Signed Bias'), 
        ('RT', 'Reaction Time'), 
        ('BiasTowardsDiagonal', 'Diag Attraction')
    ]
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']

    fig, axes = plt.subplots(4, 3, figsize=(18, 18)) # Slightly wider for bar plots
    fig.suptitle(fig_title, fontsize=16)

    for r, (met, label) in enumerate(metrics):
        for c, tms in enumerate(tms_conds):
            ax = axes[r, c]
            subset = data[data['TMS'] == tms]
            
            sns.barplot(
                data=subset, x='OffsetBin', y=met, hue=hue_col,
                palette=palette, errorbar='se', ax=ax,
                capsize=0.1, errwidth=1.5
            )
            
            ax.set_title(f"{label} | {tms}")
            if r == 3: 
                ax.set_xlabel("Init Offset Bins")
                ax.tick_params(axis='x', rotation=45, labelsize='small')
            else: 
                ax.set_xlabel("")
                ax.set_xticklabels([])
            
            if c == 0: ax.set_ylabel(label)
            else: ax.set_ylabel("")

            # User specific Y-limits (Bar Plots)
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
            
            # Smart Legend: Only on first subplot to reduce clutter
            if r == 0 and c == 0:
                ax.legend(fontsize='x-small', loc='upper right', title=hue_col)
            else:
                ax.get_legend().remove() if ax.get_legend() else None

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() # Close to free memory
    return fig

# --- Feature Engineering: Binning ---
# User Request: [-90, -60, -30, 0, 30, 60, 90]
bins = [-90, -60, -30, 0, 30, 60, 90]
bin_labels = ["-90:-60", "-60:-30", "-30:0", "0:30", "30:60", "60:90"]
df['OffsetBin'] = pd.cut(df['initOffset'], bins=bins, labels=bin_labels)

# =============================================================================
# FIGURE 1: Hue = Hemifield (Scatter)
# =============================================================================
plot_adjustment_grid(
    df, 
    hue_col='Hemifield', 
    palette={'outPF': 'blue', 'inPF': 'red'}, 
    fig_title="1. Impact of Signed Adjustment Distance (Hue: Hemifield)",
    filename="Fig1_Adjustment_Hemifield.png"
)

# =============================================================================
# FIGURE 2: Hue = OriCat (Scatter)
# =============================================================================
# User Request: 4 Distinct Colors for [30, 60, 120, 150]
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
    filename="Fig2_Adjustment_OriCat.png"
)

# =============================================================================
# FIGURE 3: Binned Bars (Hue: Hemifield)
# =============================================================================
plot_binned_grid(
    df,
    hue_col='Hemifield',
    palette={'outPF': 'blue', 'inPF': 'red'}, 
    fig_title="3. Binned Adjustment Impact (Hue: Hemifield)",
    filename="Fig3_Binned_Hemifield.png"
)

# =============================================================================
# FIGURE 4: Binned Bars (Hue: OriCat)
# =============================================================================
plot_binned_grid(
    df,
    hue_col='OriCat',
    palette=ori_palette,
    fig_title="4. Binned Adjustment Impact (Hue: OriCat)",
    filename="Fig4_Binned_OriCat.png"
)

print("Done! Figures saved.")

