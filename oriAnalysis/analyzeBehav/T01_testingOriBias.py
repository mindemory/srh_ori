"""
T01_testingOriBias.py
Orientation Bias Analysis - Testing orientation effects across bins

Extracted from V02_initialOffsetImpact_grouped.py Figure 6
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import generate_behavdf as gm
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="white", context="notebook")

# Output Dir - New folder for orientation bias
output_dir = os.path.join(gm.p['datd'], 'Figures', 'oribias')
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
df = gm.master_df.copy()
df.reset_index(drop=True, inplace=True)

# Remove LargeError trials
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
df['TMS_Group'] = pd.Categorical(df['TMS_Group'], categories=['No TMS', 'TMS'], ordered=True)

# Offset Group: Small (0-30) vs Large (30-90)
df['OffsetGroup'] = pd.cut(df['AbsInitOffset'], bins=[0, 30, 90], labels=['Small Offset', 'Large Offset'], include_lowest=True)

# Type Conversion for others
df['Hemifield'] = pd.Categorical(df['Hemifield'], categories=['outPF', 'inPF'], ordered=True)
df['OriCat'] = df['OriCat'].astype(int)

# --- ORIENTATION BINNING ---
# Bins: 0-180 in 5 deg steps
ori_bins = np.arange(0, 185, 5)
df['OriBin'] = pd.cut(df['targOriActual'], bins=ori_bins, include_lowest=True)
df['OriBinCenter'] = df['OriBin'].apply(lambda x: x.mid).astype(float)


# --- Plotting Function ---

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
            tick_indices = np.arange(0, 37, 3) # 0, 3, 6, ... 36
            tick_labels = [str(i*5) for i in tick_indices]
            
            # Only set ticks that fall within the range of bars (0-35)
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

# --- Generate Figure 1: Main TMS Experiment ---
plot_ori_bins(
    df, palette=None,
    fig_title="TMS Experiment | Orientation Bins (0-180deg) | Means ± SEM across Subjects",
    filename="OriBins_TMSEffect.png"
)


# ==============================================================================
# FIGURE 2: ZD DATASET (No TMS baseline comparison)
# ==============================================================================
print("\n" + "="*60)
print("Loading ZD Dataset (No TMS Baseline)...")
print("="*60)

import glob

zd_path = os.path.join(gm.p['datd'], 'data_beh_ZD')
zd_files = sorted(glob.glob(os.path.join(zd_path, 'sub-*_data_beh.csv')))
print(f"Found {len(zd_files)} ZD subject files")

# Load all ZD subjects
zd_dfs = []
for f in zd_files:
    subj_id = int(os.path.basename(f).split('-')[1].split('_')[0])
    subj_df = pd.read_csv(f)
    subj_df['subjID'] = subj_id
    zd_dfs.append(subj_df)

df_zd = pd.concat(zd_dfs, ignore_index=True)
print(f"Total ZD trials: {len(df_zd)}")
print(f"ZD Subjects: {df_zd['subjID'].nunique()}")

# --- ZD Feature Engineering ---
# Map corrOriRef to approximate orientation categories (for reference)
# corrOriRef 1 -> ~23deg, 2 -> ~68deg, 3 -> ~113deg, 4 -> ~158deg
df_zd['OriCat'] = df_zd['corrOriRef'].map({1: 23, 2: 68, 3: 113, 4: 158})

# Compute error the same way as in generate_behavdf.py
# errOri_scaled = (finalOri - targOriActual + 90) % 180 - 90
# In ZD: respOri = finalOri, corrOri = targOriActual
errTemp = df_zd['respOri'] - df_zd['corrOri']
df_zd['errOri_scaled'] = (errTemp + 90) % 180 - 90
df_zd['AbsError'] = np.abs(df_zd['errOri_scaled'])

# BiasTowardsDiagonal: for angles near 45, positive = toward diagonal
# For angles near 135, positive means toward 135 diagonal
# ZD orientation refs: 1 (~23, near 45), 2 (~68, near 45), 3 (~113, near 135), 4 (~158, near 135)
# Same logic as main experiment: flip sign for obtuse orientations (near 135)
df_zd['BiasTowardsDiagonal'] = df_zd['errOri_scaled'].copy()
# corrOriRef 2 and 4 are obtuse (68 is near 45, actually acute; 113 and 158 are obtuse)
# Actually: 23 and 68 are toward 45 diagonal, 113 and 158 are toward 135 diagonal
# For 45 diagonal (refs 1,2): positive error = toward diagonal
# For 135 diagonal (refs 3,4): need to flip to get "toward diagonal"
df_zd.loc[df_zd['corrOriRef'].isin([3, 4]), 'BiasTowardsDiagonal'] *= -1

print(f"ZD errOri_scaled: Mean={df_zd['errOri_scaled'].mean():.2f}, SD={df_zd['errOri_scaled'].std():.2f}")
print(f"ZD AbsError: Mean={df_zd['AbsError'].mean():.2f}")

# Compute initOffset the same way as in generate_behavdf.py
# initOffset = initOri - targOriActual (oriRespInit - corrOri in ZD)
initOffset_temp = df_zd['oriRespInit'] - df_zd['corrOri']
df_zd['initOffset'] = (initOffset_temp + 90) % 180 - 90
df_zd['AbsInitOffset'] = np.abs(df_zd['initOffset'])

print(f"ZD initOffset: Mean={df_zd['initOffset'].mean():.2f}, SD={df_zd['initOffset'].std():.2f}")
print(f"ZD AbsInitOffset: Mean={df_zd['AbsInitOffset'].mean():.2f}")
print(f"ZD Trials with AbsInitOffset < 30°: {(df_zd['AbsInitOffset'] < 30).sum()} ({(df_zd['AbsInitOffset'] < 30).mean()*100:.1f}%)")
print(f"ZD Trials with AbsInitOffset >= 30°: {(df_zd['AbsInitOffset'] >= 30).sum()} ({(df_zd['AbsInitOffset'] >= 30).mean()*100:.1f}%)")

# Offset Group for ZD (same as main experiment)
df_zd['OffsetGroup'] = pd.cut(df_zd['AbsInitOffset'], bins=[0, 30, 90], labels=['Small Offset', 'Large Offset'], include_lowest=True)

# Orientation binning
ori_bins_zd = np.arange(0, 185, 5)
df_zd['OriBin'] = pd.cut(df_zd['corrOri'], bins=ori_bins_zd, include_lowest=True)
df_zd['OriBinCenter'] = df_zd['OriBin'].apply(lambda x: x.mid).astype(float)

# Condition mapping (ZD has no TMS - conditions are likely different task parameters)
print(f"ZD Conditions: {df_zd['cond'].unique()}")
df_zd['Condition'] = df_zd['cond'].map({1: 'Cond 1', 2: 'Cond 2', 3: 'Cond 3'})


# --- Plot ZD initOffset Distribution ---
def plot_zd_offset_dist(data, fig_title, filename):
    """Plot distribution of initial offset in ZD dataset"""
    print(f"Generating Figure: {fig_title}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(fig_title, fontsize=14)
    
    # Left: Histogram of initOffset
    ax = axes[0]
    sns.histplot(data['initOffset'], bins=50, kde=True, ax=ax, color='steelblue')
    ax.axvline(0, color='r', lw=1.5, ls='--', label='Zero')
    ax.axvline(data['initOffset'].mean(), color='orange', lw=1.5, ls='-', label=f'Mean={data["initOffset"].mean():.1f}°')
    ax.set_xlabel('Initial Offset (deg)')
    ax.set_ylabel('Count')
    ax.set_title('Signed Initial Offset Distribution')
    ax.legend()
    
    # Right: Histogram of AbsInitOffset with threshold lines
    ax = axes[1]
    sns.histplot(data['AbsInitOffset'], bins=50, kde=True, ax=ax, color='darkorange')
    ax.axvline(30, color='r', lw=1.5, ls='--', label='30° threshold')
    ax.axvline(data['AbsInitOffset'].mean(), color='blue', lw=1.5, ls='-', label=f'Mean={data["AbsInitOffset"].mean():.1f}°')
    ax.set_xlabel('Absolute Initial Offset (deg)')
    ax.set_ylabel('Count')
    ax.set_title('Absolute Initial Offset Distribution')
    ax.legend()
    
    # Add offset group percentages as text
    small_pct = (data['AbsInitOffset'] < 30).mean() * 100
    large_pct = (data['AbsInitOffset'] >= 30).mean() * 100
    ax.text(0.95, 0.95, f'Small (<30°): {small_pct:.1f}%\nLarge (≥30°): {large_pct:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()


# Generate ZD offset distribution figure
plot_zd_offset_dist(
    df_zd,
    fig_title="ZD Dataset | Initial Offset Distribution",
    filename="ZD_InitOffset_Distribution.png"
)



def plot_ori_bins_zd(data, fig_title, filename):
    """Plot orientation bins for ZD dataset (no offset grouping, show by condition)"""
    print(f"Generating Figure: {fig_title}...")
    
    # Pre-aggregate per subject to get Means/Counts per bin
    print("Aggregating per subject/bin...")
    
    agg_funcs = {
        'BiasTowardsDiagonal': 'mean',
        'AbsError': 'mean',
        'subjID': 'size'
    }
    
    df_agg = data.groupby(['subjID', 'Condition', 'OriBinCenter'], observed=False).agg(agg_funcs).rename(columns={'subjID': 'TrialCount'}).reset_index()
    
    # Metrics to plot
    metrics = [
        ('TrialCount', 'Trial Count'),
        ('BiasTowardsDiagonal', 'Diag Attraction'),
        ('AbsError', 'Abs Error')
    ]
    
    conditions = ['Cond 1', 'Cond 2', 'Cond 3']
    
    # ZD only uses orientations around 4 cardinal-oblique points, so fewer bins
    # Get the actual bin centers present in data
    actual_centers = sorted(data['OriBinCenter'].dropna().unique())
    print(f"ZD has {len(actual_centers)} orientation bins: {actual_centers}")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(fig_title, fontsize=16)
    
    # Color palette for conditions
    cond_palette = {'Cond 1': '#2196F3', 'Cond 2': '#4CAF50', 'Cond 3': '#FF9800'}
    
    for r, (met, label) in enumerate(metrics):
        ax = axes[r]
        
        # Barplot with SEM
        sns.barplot(
            data=df_agg, x='OriBinCenter', y=met, hue='Condition',
            palette=cond_palette, errorbar='se', ax=ax, capsize=0.1, errwidth=1,
            order=actual_centers, hue_order=conditions
        )
        
        ax.set_title(f"{label}")
        
        if r == 2: 
            ax.set_xlabel("Orientation (deg)")
            ax.set_xticklabels([f'{c:.0f}' for c in actual_centers], rotation=45)
        else: 
            ax.set_xlabel("")
            ax.set_xticklabels([])
        
        ax.set_ylabel(label)
        
        if met == 'BiasTowardsDiagonal':
            ax.axhline(0, color='k', lw=1, ls='--')

        # Legend only in top
        if r == 0:
            ax.legend(loc='upper right', title='Condition')
        elif ax.get_legend():
            ax.get_legend().remove()
            
    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()


# Generate ZD Figure (separate)
plot_ori_bins_zd(
    df_zd,
    fig_title="ZD Dataset (No TMS) | Orientation Bins | Means ± SEM across Subjects",
    filename="OriBins_ZD_NoTMS.png"
)


# ==============================================================================
# FIGURE 3: COMBINED PLOT - TMS Experiment + ZD Cond1 Overlay
# ==============================================================================
print("\n" + "="*60)
print("Creating Combined Plot: TMS + ZD Cond1...")
print("="*60)

# Filter ZD to Cond1 only
df_zd_cond1 = df_zd[df_zd['cond'] == 1].copy()
print(f"ZD Cond1 trials: {len(df_zd_cond1)}")

# Tag datasets for combined plotting
df['Dataset'] = 'TMS NoTMS'  # Use only NoTMS from main experiment for comparison
df_tms_notms = df[df['TMS_Group'] == 'No TMS'].copy()

df_zd_cond1['Dataset'] = 'ZD Baseline'

# Make ZD compatible column names
df_zd_cond1_compat = df_zd_cond1.rename(columns={
    'corrOri': 'targOriActual'
})
df_zd_cond1_compat['OriBin'] = pd.cut(df_zd_cond1_compat['targOriActual'], bins=ori_bins, include_lowest=True)
df_zd_cond1_compat['OriBinCenter'] = df_zd_cond1_compat['OriBin'].apply(lambda x: x.mid).astype(float)


def plot_combined_ori_bins(data_tms, data_zd, fig_title, filename):
    """Plot TMS NoTMS and ZD Cond1 on same plot with full 0-180 x-axis"""
    print(f"Generating Figure: {fig_title}...")
    
    # Pre-aggregate both datasets
    print("Aggregating per subject/bin...")
    
    agg_funcs = {
        'BiasTowardsDiagonal': 'mean',
        'AbsError': 'mean',
        'subjID': 'size'
    }
    
    # TMS data
    df_tms_agg = data_tms.groupby(['subjID', 'OffsetGroup', 'OriBinCenter'], observed=False).agg(agg_funcs).rename(columns={'subjID': 'TrialCount'}).reset_index()
    df_tms_agg['Dataset'] = 'TMS NoTMS'
    
    # ZD data (no offset grouping in ZD)
    df_zd_agg = data_zd.groupby(['subjID', 'OriBinCenter'], observed=False).agg(agg_funcs).rename(columns={'subjID': 'TrialCount'}).reset_index()
    df_zd_agg['Dataset'] = 'ZD Baseline'
    
    # Metrics to plot
    metrics = [
        ('TrialCount', 'Trial Count'),
        ('BiasTowardsDiagonal', 'Diag Attraction'),
        ('AbsError', 'Abs Error')
    ]
    
    offset_groups = ['Small Offset', 'Large Offset']
    
    # Full 0-180 x-axis in 5 deg steps
    all_centers = np.arange(2.5, 180, 5) 
    
    # Color palette
    dataset_palette = {'TMS NoTMS': '#1f77b4', 'ZD Baseline': '#2ca02c'}
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(fig_title, fontsize=16)
    
    for r, (met, label) in enumerate(metrics):
        for c, off_group in enumerate(offset_groups):
            ax = axes[r, c]
            
            # Get TMS data for this offset group
            tms_subset = df_tms_agg[df_tms_agg['OffsetGroup'] == off_group].copy()
            
            # ZD has no offset groups - use all data
            zd_subset = df_zd_agg.copy()
            
            # Combine
            plot_df = pd.concat([tms_subset, zd_subset], ignore_index=True)
            
            if plot_df.empty: continue
            
            # Barplot with SEM
            sns.barplot(
                data=plot_df, x='OriBinCenter', y=met, hue='Dataset',
                palette=dataset_palette, errorbar='se', ax=ax, capsize=0.1, errwidth=1,
                order=all_centers, hue_order=['TMS NoTMS', 'ZD Baseline']
            )
            
            ax.set_title(f"{label} | {off_group}")
            
            if r == 2: ax.set_xlabel("Orientation (deg)")
            else: ax.set_xlabel("")
            
            if c == 0: ax.set_ylabel(label)
            else: ax.set_ylabel("")
            
            # Ticks at 0, 15, 30, ... 165
            tick_indices = np.arange(0, 37, 3)
            tick_labels = [str(i*5) for i in tick_indices]
            valid_indices = [i for i in tick_indices if i < len(all_centers)]
            valid_labels = [tick_labels[i] for i, idx in enumerate(tick_indices) if idx < len(all_centers)]
            
            ax.set_xticks(valid_indices)
            ax.set_xticklabels(valid_labels)
            
            if met == 'BiasTowardsDiagonal':
                ax.axhline(0, color='k', lw=1, ls='--')

            # Legend only in top left
            if r == 0 and c == 0:
                ax.legend(loc='upper right', title='Dataset')
            elif ax.get_legend():
                ax.get_legend().remove()
                
    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()


# Generate Combined Figure
plot_combined_ori_bins(
    df_tms_notms, df_zd_cond1_compat,
    fig_title="Combined: TMS NoTMS vs ZD Baseline (Cond1) | Orientation Bins",
    filename="OriBins_TMS_vs_ZD_Combined.png"
)

print(f"\nDone! All figures saved to {output_dir}")
