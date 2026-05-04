import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import generate_behavdf as gm

print("Loaded master_df.")

def get_cw_tendency_continuous(ori_array):
    """
    Returns the fraction of total distance moved in the Clockwise direction.
    1.0 = Pure CW rotation, 0.0 = Pure CCW rotation.
    """
    if len(ori_array) < 2: return 0.5
    diffs = np.diff(ori_array)
    diffs[diffs > 90] -= 180
    diffs[diffs < -90] += 180
    
    cw_dist = np.sum(np.abs(diffs[diffs < 0]))
    ccw_dist = np.sum(diffs[diffs > 0])
    total_dist = cw_dist + ccw_dist
    
    if total_dist == 0: return 0.5
    return cw_dist / total_dist

df = gm.master_df.copy()

print("Preprocessing and cleaning data...")
# --- FILTERING "BAD" TRIALS ---
n_orig = len(df)
df = df[df['LargeError'] == 0]
df = df[df['RT'] < 3.0]
df = df[df['oriSteps'] > 0]
print(f"Dropped {n_orig - len(df)} bad trials total (Outliers/Timeouts).")

print("Unwrapping arrays (~5 seconds)...")
df['CW_Tendency'] = df['oriArray'].apply(get_cw_tendency_continuous)
df['InitOffset'] = (df['initOri'] - df['targOriActual'] + 90) % 180 - 90

# Radial vs Tangential mapping logic (Updated per USER)
def get_ori_type(row):
    ori = row['targOriCategory']
    hemi = row['hemisphereStimulated']
    pf   = row['stimPF_label']
    
    if hemi == 'Left': # Phosphene in Lower Right (Right VF)
        if pf == 'inPF': # Right VF
            return 'Radial' if ori in [120, 150] else 'Tangential'
        else: # Left VF
            return 'Radial' if ori in [30, 60] else 'Tangential'
    else: # Phosphene in Lower Left (Left VF)
        if pf == 'inPF': # Left VF
            return 'Radial' if ori in [30, 60] else 'Tangential'
        else: # Right VF
            return 'Radial' if ori in [120, 150] else 'Tangential'

df['OriType'] = df.apply(get_ori_type, axis=1)

# --- DATA AGGREGATION FOR POPULATION CURVES (Panel A) ---
df['TMS'] = df['TMScond_label']
bin_edges = np.arange(-90, 92, 2)
df['Offset_Bin'] = pd.cut(df['InitOffset'], bins=bin_edges)

# Subject-level aggregation
sub_stats = df.groupby(
    ['subjID', 'TMS', 'Offset_Bin'], observed=True
).agg(
    Mean_Tendency=('CW_Tendency', 'mean'),
    Bin_Mid=('InitOffset', 'mean')
).reset_index()

# Circular Smoothing over 5 bins
n_sm = 5
smoothed_list = []
for (subj, tms), subj_df in sub_stats.groupby(['subjID', 'TMS']):
    subj_df = subj_df.sort_values('Bin_Mid').reset_index(drop=True)
    if len(subj_df) < 10: continue
    padded = pd.concat([subj_df.tail(n_sm), subj_df, subj_df.head(n_sm)]).reset_index(drop=True)
    padded['Mean_Tendency'] = padded['Mean_Tendency'].rolling(window=n_sm, center=True).mean()
    res = padded.iloc[n_sm:-n_sm].copy()
    smoothed_list.append(res)

sub_stats_smooth = pd.concat(smoothed_list)

# Population aggregation
pop_stats = sub_stats_smooth.groupby(
    ['TMS', 'Offset_Bin'], observed=True
).agg(
    Pop_Mean=('Mean_Tendency', 'mean'),
    Pop_SEM=('Mean_Tendency', lambda x: np.std(x) / np.sqrt(len(x))),
    Final_Bin_Mid=('Bin_Mid', 'mean')
).reset_index().dropna(subset=['Pop_Mean'])

from scipy.stats import norm
from scipy.optimize import curve_fit

def sine_func(x, amp, phase):
    # Sine wave with 180 degree period (wraps at -90 and 90)
    return 0.5 + amp * np.sin(np.deg2rad(x) * 2 + phase)

# --- Empirical Bayes Hierarchical Fitting (Shrinkage) ---
print("\nRunning Empirical Bayes Shrinkage (2-Pass Fit)...")

def run_eb_fit(data_df, group_col):
    # Pass 1: Get raw fits to estimate group-level priors
    raw_fits = []
    for (subj, tms, grp), subj_df in data_df.groupby(['subjID', 'TMS', group_col]):
        bin_agg = subj_df.groupby('Offset_Bin', observed=True).agg({'CW_Tendency': 'mean', 'InitOffset': 'mean'}).reset_index()
        bin_agg = bin_agg.sort_values('InitOffset').reset_index(drop=True)
        if len(bin_agg) < 15: continue
        x, y = bin_agg['InitOffset'].values, bin_agg['CW_Tendency'].values
        try:
            popt, _ = curve_fit(sine_func, x, y, p0=[0.3, 0], bounds=([0, -np.pi], [0.5, np.pi]))
            raw_fits.append({'subjID': subj, 'TMS': tms, 'Group': grp, 'Amp': popt[0], 'Phase': popt[1]})
        except: pass
    
    df_raw = pd.DataFrame(raw_fits)
    priors = df_raw.groupby(['TMS', 'Group']).agg({'Amp': ['mean', 'std'], 'Phase': ['mean', 'std']})
    
    # Pass 2: Regularized fitting (Shrinkage)
    eb_results = []
    for (subj, tms, grp), subj_df in data_df.groupby(['subjID', 'TMS', group_col]):
        bin_agg = subj_df.groupby('Offset_Bin', observed=True).agg({'CW_Tendency': 'mean', 'InitOffset': 'mean'}).reset_index()
        bin_agg = bin_agg.sort_values('InitOffset').reset_index(drop=True)
        if len(bin_agg) < 10: continue
        x, y = bin_agg['InitOffset'].values, bin_agg['CW_Tendency'].values
        
        # Get prior for this condition
        try:
            m_a, s_a = priors.loc[(tms, grp), ('Amp', 'mean')], priors.loc[(tms, grp), ('Amp', 'std')]
            m_p, s_p = priors.loc[(tms, grp), ('Phase', 'mean')], priors.loc[(tms, grp), ('Phase', 'std')]
        except: m_a, s_a, m_p, s_p = 0.3, 0.1, 0, 0.2
        
        def eb_loss(params):
            amp, phase = params
            data_loss = np.sum((y - sine_func(x, amp, phase))**2)
            # Penalty (Log-prior): (val - mean)^2 / var
            prior_loss = 0.1 * (((amp - m_a)**2 / (s_a**2 + 1e-5)) + ((phase - m_p)**2 / (s_p**2 + 1e-5)))
            return data_loss + prior_loss

        from scipy.optimize import minimize
        res = minimize(eb_loss, x0=[m_a, m_p], bounds=[(0, 0.5), (-np.pi, np.pi)])
        
        # R2 relative to shrunken fit
        y_fit = sine_func(x, *res.x)
        r2 = 1 - (np.sum((y - y_fit)**2) / (np.sum((y - np.mean(y))**2) + 1e-12))
        eb_results.append({'subjID': subj, 'TMS': tms, 'Group': grp, 'Amp': res.x[0], 'Phase': res.x[1], 'R2': r2})
    
    return pd.DataFrame(eb_results)

# Run breakdowns
res_hemi = run_eb_fit(df, 'stimPF_label')
res_hemi['Breakdown'] = 'Hemifield'
res_axis = run_eb_fit(df, 'OriType')
res_axis['Breakdown'] = 'Axis'

df_fit = pd.concat([res_hemi, res_axis])

# -------------------------------------------------------
# Visualization Dashboard (6 Panels - EB Shrunk)
# -------------------------------------------------------
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3)
order = ['No TMS', 'Early TMS', 'Middle TMS']

# Muted Palettes from Paper Style
hemi_palette = {'inPF': '#5D6D7E', 'outPF': '#EB984E'}
light_hemi_pal = {'inPF': '#D5DBDB', 'outPF': '#FAD7A0'}

axis_palette = {'Radial': '#A569BD', 'Tangential': '#45B39D'}
light_axis_pal = {'Radial': '#EBDEF0', 'Tangential': '#D1F2EB'}

tms_box_pal = {'No TMS': '#E0E0E0', 'Early TMS': '#FFCDD2', 'Middle TMS': '#BBDEFB'}
tms_point_pal = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}

# Panel 1: Population Curves (Equal size now)
ax1 = fig.add_subplot(gs[0, 0])
for tms, color in tms_point_pal.items():
    cond_df = pop_stats[pop_stats['TMS'] == tms]
    if len(cond_df) == 0: continue
    ax1.plot(cond_df['Final_Bin_Mid'], cond_df['Pop_Mean'], color=color, linewidth=2, label=tms, alpha=0.8)
    ax1.fill_between(cond_df['Final_Bin_Mid'], cond_df['Pop_Mean'] - cond_df['Pop_SEM'], cond_df['Pop_Mean'] + cond_df['Pop_SEM'], color=color, alpha=0.1)
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
ax1.set_title("1. Population Motor Rotation", fontweight='bold')
ax1.legend(frameon=False, fontsize=8)

# Panel 2: R2 Consolidated
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=df_fit, x='TMS', y='R2', order=order, ax=ax2, palette=tms_box_pal, showfliers=False, width=0.6)
sns.stripplot(data=df_fit, x='TMS', y='R2', order=order, ax=ax2, palette=tms_point_pal, alpha=0.4, size=4)
ax2.set_title("2. Goodness of Fit (R^2)", fontweight='bold')
ax2.set_ylim(0, 1.05)

# Metrics Mapping
def plot_panel(row, col, breakdown, m, label, main_pal, light_pal, start_num):
    ax = fig.add_subplot(gs[row, col])
    data = df_fit[df_fit['Breakdown'] == breakdown]
    # Boxes use light palette
    sns.boxplot(data=data, x='TMS', y=m, hue='Group', palette=light_pal, ax=ax, showfliers=False, order=order, width=0.6)
    # Dots use saturated palette
    sns.stripplot(data=data, x='TMS', y=m, hue='Group', palette=main_pal, ax=ax, alpha=0.5, size=4, dodge=True, order=order, legend=False)
    ax.set_title(f"{start_num}. {label} ({breakdown})", fontweight='bold')
    ax.set_xlabel(""); ax.set_ylabel("")
    if m == 'Amp': ax.set_ylim(0, 0.55)
    if m == 'Phase': ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

# Row 1 (Hemifield Metrics)
plot_panel(0, 2, 'Hemifield', 'Amp', 'Amplitude', hemi_palette, light_hemi_pal, 3)
plot_panel(1, 0, 'Hemifield', 'Phase', 'Phase', hemi_palette, light_hemi_pal, 4)

# Row 2 (Axis Metrics - Renamed to Radial Sensitivity)
plot_panel(1, 1, 'Axis', 'Amp', 'Amplitude', axis_palette, light_axis_pal, 5)
ax_tmp = fig.axes[-1]; ax_tmp.set_title("5. Amplitude (Radial Sensitivity)", fontweight='bold')

plot_panel(1, 2, 'Axis', 'Phase', 'Phase', axis_palette, light_axis_pal, 6)
ax_tmp = fig.axes[-1]; ax_tmp.set_title("6. Phase (Radial Sensitivity)", fontweight='bold')

plt.tight_layout()

# -------------------------------------------------------
# Statistics Engine (Conditional)
# -------------------------------------------------------
def run_perm_test(x, y=None, n_perms=10000):
    """Simple permutation test for 1-sample (vs 0) or 2-sample paired."""
    x = np.array(x)
    if y is not None:
        # Paired Difference
        diff = x - np.array(y)
        obs = np.mean(diff)
        null_dist = []
        for _ in range(n_perms):
            signs = np.random.choice([-1, 1], size=len(diff))
            null_dist.append(np.mean(diff * signs))
    else:
        # 1-sample vs 0
        obs = np.mean(x)
        null_dist = []
        for _ in range(n_perms):
            signs = np.random.choice([-1, 1], size=len(x))
            null_dist.append(np.mean(x * signs))
    
    p = np.mean(np.abs(null_dist) >= np.abs(obs))
    p = max(p, 1/n_perms) # Floor at resolution
    cohen_d = obs / (np.std(x - y) if y is not None else np.std(x))
    return p, cohen_d

if 'output_dir' in locals() or 'output_dir' in globals():
    # 1. Save Figure
    save_path = os.path.join(output_dir, 'Fig3', 'Fig3B_MotorPreference.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.savefig(save_path.replace('.png', '.svg'))
    print(f"Figure saved to: {save_path}")

    # 2. Generate Stats
    stats_rows = []
    tms_order = ['No TMS', 'Early TMS', 'Middle TMS']
    
    # R2 Stats (Consolidated)
    for tms in tms_order:
        vals = df_fit[df_fit['TMS'] == tms]['R2'].dropna().values
        if len(vals) > 5:
            p, d = run_perm_test(vals)
            stats_rows.append({'Analysis': 'R2_Fit', 'TMS': tms, 'Group_A': 'All', 'Group_B': 'Zero', 
                               'Mean_A': np.mean(vals), 'SEM_A': stats.sem(vals), 'p_perm': p, 'Cohen_d': d})

    # Metric Stats (Hemi & Axis)
    for breakdown, grp_names in [('Hemifield', ('inPF', 'outPF')), ('Axis', ('Radial', 'Tangential'))]:
        for m in ['Amp', 'Phase']:
            sub = df_fit[df_fit['Breakdown'] == breakdown]
            for tms in tms_order:
                piv = sub[sub['TMS'] == tms].pivot(index='subjID', columns='Group', values=m).dropna()
                if not piv.empty and grp_names[0] in piv.columns and grp_names[1] in piv.columns:
                    v_a, v_b = piv[grp_names[0]].values, piv[grp_names[1]].values
                    p, d = run_perm_test(v_a, v_b)
                    stats_rows.append({
                        'Analysis': f"{breakdown}_{m}", 'TMS': tms, 'Group_A': grp_names[0], 'Group_B': grp_names[1],
                        'Mean_A': np.mean(v_a), 'SEM_A': stats.sem(v_a), 'Mean_B': np.mean(v_b), 'SEM_B': stats.sem(v_b),
                        'p_perm': p, 'Cohen_d': d
                    })

    stats_df = pd.DataFrame(stats_rows)
    # Manual formatting for precision requirements
    def format_p(p):
        return f"{p:.4e}" if p < 0.001 else f"{p:.4f}"
    
    if not stats_df.empty:
        stats_df['p_perm'] = stats_df['p_perm'].apply(format_p)
        for col in ['Mean_A', 'SEM_A', 'Mean_B', 'SEM_B', 'Cohen_d']:
            if col in stats_df.columns:
                stats_df[col] = stats_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)

    stats_csv = os.path.join(output_dir, 'Fig3', 'Fig3B_MotorPreference_Stats.csv')
    stats_df.to_csv(stats_csv, index=False)
    print(f"Stats saved to: {stats_csv}")

plt.show()
