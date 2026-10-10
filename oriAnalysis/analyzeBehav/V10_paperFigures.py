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
import sys
import joblib

warnings.filterwarnings('ignore')
sns.set_theme(style="white", context="notebook")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Output Dir
output_dir = os.path.join(gm.p['datd'], 'Figures', 'paperFigures')
os.makedirs(output_dir, exist_ok=True)

# --- Global color scheme: darker = inPF / Radial, lighter = outPF / Tangential ---
COL_A_BOX   = '#4472C4'   # darker blue  — inPF / Radial
COL_A_STRIP = '#1A3A6B'   # deep navy
COL_B_BOX   = '#A9C4E8'   # lighter blue — outPF / Tangential
COL_B_STRIP = '#5B8DC7'   # mid blue

# TMS condition colors — matching Fig 1B scatter (k / r / b)
TMS_BOX_PAL   = {'No TMS': '#9E9E9E', 'Early TMS': '#EF9A9A', 'Middle TMS': '#90CAF9'}
TMS_STRIP_PAL = {'No TMS': 'k',       'Early TMS': '#C62828',  'Middle TMS': '#1565C0'}
# Paired dark/light shades per TMS (kept for other uses)
TMS_DARK  = {'No TMS': '#424242', 'Early TMS': '#C62828', 'Middle TMS': '#0D47A1'}
TMS_LIGHT = {'No TMS': '#BDBDBD', 'Early TMS': '#FFCDD2', 'Middle TMS': '#BBDEFB'}

import matplotlib.patches as mpatches

def _draw_paired_boxes(ax, long_df, x_order, A, B, tms_groups,
                       show_legend=False, leg_labels=None):
    """Draw touching paired boxes (A=full, B=0.65 alpha) using custom positions."""
    bw   = 0.28   # box width
    gap  = 0.15   # gap between TMS groups (center-to-center of pair vs next pair)

    pos_map = {}
    xtick_pos, xtick_lbl = [], []
    for i, t in enumerate(tms_groups):
        base = i * (2 * bw + gap)
        pos_map[f'{t}|{A}'] = base
        pos_map[f'{t}|{B}'] = base + bw   # touching, zero gap
        xtick_pos.append(base + bw / 2)   # midpoint of pair
        xtick_lbl.append(t)

    rng = np.random.default_rng(42)
    for key in x_order:
        pos = pos_map.get(key)
        if pos is None: continue
        t   = key.split('|')[0]
        cond = key.split('|', 1)[1]
        vals = long_df[long_df['TMS_Cond'] == key]['NormValue'].dropna().values
        if len(vals) < 3: continue

        is_A = (cond == A)
        col  = TMS_BOX_PAL[t]
        scol = TMS_STRIP_PAL[t]
        al   = 1.0 if is_A else 0.65
        sal  = 0.7 if is_A else 0.35

        ax.boxplot([vals], positions=[pos], widths=bw, patch_artist=True,
                   showfliers=False, manage_ticks=False,
                   boxprops=dict(facecolor=col, alpha=al, linewidth=0.8, edgecolor=scol),
                   medianprops=dict(color='white', linewidth=2),
                   whiskerprops=dict(color=scol, alpha=al, linewidth=0.8),
                   capprops=dict(color=scol, alpha=al, linewidth=0.8))
        jitter = rng.uniform(-bw * 0.28, bw * 0.28, len(vals))
        ax.scatter(pos + jitter, vals, color=scol, alpha=sal, s=11, zorder=3)

    ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=9)
    xl = pos_map[x_order[0]] - bw * 0.8
    xr = pos_map[x_order[-1]] + bw * 1.8
    ax.set_xlim(xl, xr)
    if ax.get_legend(): ax.get_legend().remove()
    if show_legend:
        notms_col = TMS_BOX_PAL['No TMS']
        leg_a, leg_b = leg_labels or (A, B)
        handles = [
            mpatches.Patch(facecolor=notms_col, alpha=1.0, edgecolor='k', lw=0.8, label=leg_a),
            mpatches.Patch(facecolor=notms_col, alpha=0.65, edgecolor='k', lw=0.8, label=leg_b),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper right')

print("Loading data...")
df = gm.master_df.copy()
df.reset_index(drop=True, inplace=True)

# Remove LargeError trials
df = df[df['LargeError'] == 0].copy()

# --- Feature Engineering ---
df.rename(columns={'TMScond_label': 'TMS', 'stimPF_label': 'Hemifield', 'targOriCategory': 'OriCat', 'hemisphereStimulated': 'HemiStim'}, inplace=True)
df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1
df['AbsError'] = np.abs(df['errOri_scaled'])
df['AbsInitOffset'] = np.abs(df['initOffset'])

# Continuous Motor Preference (CW Tendency)
def get_cw_tendency_continuous(arr):
    if not isinstance(arr, (list, np.ndarray)) or len(arr) < 2: return np.nan
    diffs = np.diff(arr)
    diffs = (diffs + 90) % 180 - 90
    return 1.0 if np.sum(diffs) > 0 else 0.0

df['CW_Tendency'] = df['oriArray'].apply(get_cw_tendency_continuous)
df['initOffset'] = (df['initOri'] - df['targOriActual'] + 90) % 180 - 90 # Standardize

# Orientation Axis Mapping (Radial vs Tangential)
def get_ori_type(row):
    ori, hemi, pf = row['OriCat'], row['HemiStim'], row['Hemifield']
    if hemi == 'Left': # Stimulating Left Hemi -> Phosphene in Right VF
        if pf == 'inPF': # Right VF
            return 'Radial' if ori in [120, 150] else 'Tangential'
        else: # Left VF
            return 'Radial' if ori in [30, 60] else 'Tangential'
    else: # Stimulating Right Hemi -> Phosphene in Left VF
        if pf == 'inPF': # Left VF
            return 'Radial' if ori in [30, 60] else 'Tangential'
        else: # Right VF
            return 'Radial' if ori in [120, 150] else 'Tangential'

df['DiagType'] = df.apply(get_ori_type, axis=1)

# --- GROUPING LOGIC ---
df['TMS_Group'] = df['TMS'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
df['TMS_Group'] = pd.Categorical(df['TMS_Group'], categories=['No TMS', 'TMS'], ordered=True)
df['OffsetGroup'] = pd.cut(df['AbsInitOffset'], bins=[0, 25.1, 44.9, 90.1], labels=['Small Offset', 'Medium Offset', 'Large Offset'], include_lowest=True)
df['OffsetGroup'] = df['OffsetGroup'].cat.remove_categories(['Medium Offset'])
df['Hemifield'] = pd.Categorical(df['Hemifield'], categories=['outPF', 'inPF'], ordered=True)
df['OriCat'] = df['OriCat'].astype(int)
df['OriCat45'] = df['OriCat'].replace({30: 'Acute (30+60)', 60: 'Acute (30+60)', 120: 'Obtuse (120+150)', 150: 'Obtuse (120+150)'})

# --- Statistical Helpers ---
def run_permutation_test(vals, n_perms=10000):
    vals = np.array(vals); vals = vals[~np.isnan(vals)]
    if len(vals) == 0: return 1.0
    obs = np.mean(vals)
    null = [np.mean(vals * np.random.choice([-1, 1], size=len(vals))) for _ in range(n_perms)]
    p = np.mean(np.abs(null) >= np.abs(obs))
    return max(p, 1/n_perms)

def run_paired_permutation_test(v1, v2, n_perms=10000):
    diffs = np.array(v1) - np.array(v2); diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0: return 1.0
    obs = np.mean(diffs)
    null = [np.mean(diffs * np.random.choice([-1, 1], size=len(diffs))) for _ in range(n_perms)]
    p = np.mean(np.abs(null) >= np.abs(obs))
    return max(p, 1/n_perms)

def get_sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

def plot_contrast_grid(data, contrast_cols, fig_title, filename, bin_col=None, bins=None, paired=False, metrics_subset=None, fig_folder=None, return_data=False):
    if not return_data: print(f"Generating Figure: {fig_title}...")
    save_dir = os.path.join(output_dir, fig_folder) if fig_folder else output_dir
    os.makedirs(save_dir, exist_ok=True)
    metrics = metrics_subset if metrics_subset else [('AbsError', 'Abs Error'), ('RT', 'Reaction Time'), ('BiasTowardsDiagonal', 'Diag Attraction'), ('Precision', 'Precision (K)')]
    tms_groups = ['No TMS', 'Early TMS', 'Middle TMS']
    A, B = contrast_cols; p_col = 'Hemifield' if A in ['inPF', 'outPF'] else 'DiagType'
    g_cols = ['subjID', 'TMS', p_col]
    if bin_col: g_cols.append(bin_col)
    
    df_s = data.groupby(g_cols, observed=False)[['AbsError', 'RT', 'BiasTowardsDiagonal']].mean().reset_index()
    mix_s = []
    for t in tms_groups:
        for v in [A, B]:
            s_bins = bins if bins else [None]
            for b in s_bins:
                for subj in data['subjID'].unique():
                    mask = (data['TMS'] == t) & (data[p_col] == v) & (data['subjID'] == subj)
                    if bin_col and b: mask &= (data[bin_col] == b)
                    sub = data[mask]
                    if len(sub) < 10: continue
                    try:
                        res = md.fit_model(sub['errOri_scaled'].dropna().values, 'mixtureModel', max_trials=100) # Faster trials for sensitivity
                        item = {'TMS': t, p_col: v, 'subjID': subj, 'Precision': res[1]}
                        if bin_col: item[bin_col] = b
                        mix_s.append(item)
                    except: pass
    df_m = pd.DataFrame(mix_s); df_all = pd.merge(df_s, df_m, on=g_cols, how='outer') if not df_m.empty else df_s
    
    # Calculate normalized ratios A/(A+B) and B/(A+B) for all metrics
    results_list = []
    for met, lab in metrics:
        if met not in df_all.columns: continue
        piv = df_all.pivot_table(index=['subjID', 'TMS'], columns=p_col, values=met, observed=False).reset_index()
        if A in piv.columns and B in piv.columns:
            av_d, bv_d = (np.abs(piv[A]), np.abs(piv[B])) if met == 'BiasTowardsDiagonal' else (piv[A], piv[B])
            total = av_d + bv_d + 1e-12
            piv['CI'] = (av_d - bv_d) / total  # keep for return_data backward compat
            piv[f'{A}_norm'] = av_d / total
            piv[f'{B}_norm'] = bv_d / total
            piv['Metric'] = lab
            results_list.append(piv)
    
    if return_data: 
        return pd.concat(results_list) if results_list else pd.DataFrame()

    n_bins = len(bins) if bins else 1
    fig, axes = plt.subplots(1, len(metrics) * n_bins, figsize=(6 * len(metrics), 5), squeeze=False)
    stats_rows = []
    
    for im, (met, lab) in enumerate(metrics):
        for cb in range(n_bins):
            ax = axes[0, im + len(metrics)*cb]; bv = bins[cb] if bins else None
            piv = next(d for d in results_list if d['Metric'].iloc[0] == lab)

            long_rows = []
            for t in tms_groups:
                sub = piv[piv['TMS'] == t].dropna(subset=[f'{A}_norm', f'{B}_norm'])
                for _, row in sub.iterrows():
                    long_rows.append({'TMS': t, 'Condition': A, 'TMS_Cond': f'{t}|{A}', 'NormValue': row[f'{A}_norm'], 'subjID': row['subjID']})
                    long_rows.append({'TMS': t, 'Condition': B, 'TMS_Cond': f'{t}|{B}', 'NormValue': row[f'{B}_norm'], 'subjID': row['subjID']})
            long_df = pd.DataFrame(long_rows)
            x_order = [f'{t}|{c}' for t in tms_groups for c in [A, B]]

            _draw_paired_boxes(ax, long_df, x_order, A, B, tms_groups,
                               show_legend=(im == 0 and cb == 0))

            # Stats: permutation test vs 0.5
            for it, t in enumerate(tms_groups):
                sub = piv[piv['TMS'] == t].dropna(subset=[f'{A}_norm', f'{B}_norm'])
                for col_key, cond in [(f'{A}_norm', A), (f'{B}_norm', B)]:
                    v = sub[col_key].dropna().values
                    if len(v) > 5:
                        diffs = v - 0.5
                        p = run_permutation_test(diffs)
                        mean_v, sem_v = np.mean(v), stats.sem(v)
                        cohen_d = np.mean(diffs) / np.std(diffs)
                        t_stat, _ = stats.ttest_1samp(v, 0.5)
                        stats_rows.append({'Metric': lab, 'Condition': cond, 'TMS': t, 'Bin': bv,
                                           'Mean': mean_v, 'SEM': sem_v, 'p_perm': p,
                                           'Cohen_d': cohen_d, 't_stat': t_stat, 'N': len(v)})

            ax.set_title(f"{lab}" + (f"\n({bv})" if bv else ""))
            ax.set_ylabel(f"Normalized {lab}")
            
            # Set dynamic ylim based on user request
            if lab == 'Abs Error':
                ax.set_ylim(0.4, 0.6)
            elif lab == 'Reaction Time':
                ax.set_ylim(0.45, 0.55)
            else:
                ax.set_ylim(0.3, 0.7)
            
            ax.set_xlabel('')
            sns.despine(ax=ax, offset=5)

    if stats_rows:
        sdf = pd.DataFrame(stats_rows)
        def format_p(p): return f"{p:.4e}" if p < 0.001 else f"{p:.4f}"
        sdf['p_perm'] = sdf['p_perm'].apply(format_p)
        for c in ['Mean', 'SEM', 'Cohen_d']:
            if c in sdf.columns: sdf[c] = sdf[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)
        sdf.to_csv(os.path.join(save_dir, filename.replace('.png', '_Stats.csv')), index=False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.svg')))
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.pdf')))
    plt.close()

def plot_scatter_in_vs_out(data, fig_title, filename, metrics_subset=None, fig_folder=None, p_col='Hemifield', A='inPF', B='outPF'):
    print(f"Generating Figure: {fig_title}...")
    save_dir = os.path.join(output_dir, fig_folder) if fig_folder else output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = metrics_subset if metrics_subset else [('AbsError', 'Abs Error'), ('RT', 'Reaction Time'), ('Precision', 'Precision (K)')]
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']
    df_means = data.groupby(['subjID', 'TMS', p_col], observed=False)[['AbsError', 'RT']].mean().reset_index()
    
    mixture_stats = []
    for t in tms_conds:
        for v in [A, B]:
            for s in data['subjID'].unique():
                sub = data[(data['TMS'] == t) & (data[p_col] == v) & (data['subjID'] == s)]
                if len(sub) < 10: continue
                try:
                    res = md.fit_model(sub['errOri_scaled'].dropna().values, 'mixtureModel', max_trials=100)
                    mixture_stats.append({'TMS': t, p_col: v, 'subjID': s, 'Precision': res[1]})
                except: pass
    df_mix = pd.DataFrame(mixture_stats)
    df_all = pd.merge(df_means, df_mix, on=['subjID', 'TMS', p_col], how='outer') if not df_mix.empty else df_means
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False); axes = axes.flatten()
    for i, (met, label) in enumerate(metrics):
        ax = axes[i]
        piv = df_all.pivot_table(index=['subjID', 'TMS'], columns=p_col, values=met, observed=False).reset_index()
        if A not in piv.columns or B not in piv.columns: continue
        for t in tms_conds:
            sub = piv[piv['TMS'] == t].dropna(subset=[A, B])
            if len(sub) == 0: continue
            if met == 'Precision':
                count_val = (sub[A] < sub[B]).sum(); sym = '<'
            else:
                count_val = (sub[A] > sub[B]).sum(); sym = '>'
            label_text = f"{t} (n={count_val} {A}{sym}{B})"
            ax.scatter(sub[B], sub[A], color={'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}[t], alpha=0.4, s=20)
            ax.errorbar(sub[B].mean(), sub[A].mean(), xerr=sub[B].sem(), yerr=sub[A].sem(), fmt='o', color={'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}[t], markersize=10, markeredgecolor='white', label=label_text)
        
        v = piv[[A, B]].values.flatten(); l, h = np.nanmin(v), np.nanmax(v)
        ax.plot([l, h], [l, h], 'k--', alpha=0.3)
        ax.set_aspect('equal', 'box'); ax.set_title(label); ax.set_xlabel(B); ax.set_ylabel(A)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.svg')))
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.pdf')))
    plt.close()

def plot_scatter_diag_contrast_in_vs_out(data, fig_title, filename, metrics_subset=None, fig_folder=None):
    print(f"Generating Figure: {fig_title}...")
    save_dir = os.path.join(output_dir, fig_folder) if fig_folder else output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = metrics_subset if metrics_subset else [('AbsError', 'Abs Error'), ('RT', 'Reaction Time'), ('Precision', 'Precision (K)')]
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']
    df_means = data.groupby(['subjID', 'TMS', 'Hemifield', 'DiagType'], observed=False)[['AbsError', 'RT']].mean().reset_index()
    mixture_stats = []
    for t in tms_conds:
        for h in ['inPF', 'outPF']:
            for d in ['Radial', 'Tangential']:
                for s in data['subjID'].unique():
                    subset = data[(data['TMS'] == t) & (data['Hemifield'] == h) & (data['DiagType'] == d) & (data['subjID'] == s)]
                    if len(subset) < 10: continue
                    try:
                        res = md.fit_model(subset['errOri_scaled'].dropna().values, 'mixtureModel')
                        mixture_stats.append({'TMS': t, 'Hemifield': h, 'DiagType': d, 'subjID': s, 'Precision': res[1]})
                    except: pass
    df_mix = pd.DataFrame(mixture_stats); df_all = pd.merge(df_means, df_mix, on=['subjID', 'TMS', 'Hemifield', 'DiagType'], how='outer') if not df_mix.empty else df_means
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False); axes = axes.flatten()
    for i, (met, label) in enumerate(metrics):
        ax = axes[i]; piv = df_all.pivot_table(index=['subjID', 'TMS', 'Hemifield'], columns='DiagType', values=met, observed=False).reset_index()
        if 'Radial' not in piv.columns or 'Tangential' not in piv.columns: continue
        piv['CI'] = (piv['Radial'] - piv['Tangential']) / (piv['Radial'] + piv['Tangential'] + 1e-12)
        piv_h = piv.pivot_table(index=['subjID', 'TMS'], columns='Hemifield', values='CI', observed=False).reset_index()
        if 'inPF' not in piv_h.columns or 'outPF' not in piv_h.columns: continue
        for t in tms_conds:
            sub = piv_h[piv_h['TMS'] == t].dropna(subset=['inPF', 'outPF'])
            if met == 'Precision':
                count_val = (sub['inPF'] < sub['outPF']).sum(); sym = '<'
            else:
                count_val = (sub['inPF'] > sub['outPF']).sum(); sym = '>'
            label_text = f"{t} (n={count_val} in{sym}out)"
            ax.scatter(sub['outPF'], sub['inPF'], color={'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}[t], alpha=0.4, s=30, edgecolor='white')
            ax.errorbar(sub['outPF'].mean(), sub['inPF'].mean(), xerr=sub['outPF'].sem(), yerr=sub['inPF'].sem(), fmt='o', color={'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}[t], markersize=10, markeredgecolor='white', label=label_text)
        v = piv_h[['inPF', 'outPF']].values.flatten(); l, h = np.nanmin(v), np.nanmax(v)
        ax.plot([l, h], [l, h], 'k--', alpha=0.3); ax.axhline(0, color='k', alpha=0.1); ax.axvline(0, color='k', alpha=0.1)
        ax.set_aspect('equal', 'box'); ax.set_title(f"Radial Sensitivity: {label}"); ax.set_xlabel("outPF (CI)"); ax.set_ylabel("inPF (CI)")
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.svg')))
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.pdf')))
    plt.close()

def plot_scatter_radial_vs_tangential_overlay(data, fig_title, filename, hemi='inPF', metrics_subset=None, fig_folder=None):
    print(f"Generating Figure: {fig_title}...")
    save_dir = os.path.join(output_dir, fig_folder) if fig_folder else output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = metrics_subset if metrics_subset else [('AbsError', 'Abs Error'), ('Precision', 'Precision (K)')]
    palette = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}
    df_means = data.groupby(['subjID', 'TMS', 'Hemifield', 'DiagType'], observed=False)[['AbsError', 'RT']].mean().reset_index()
    mix_s = []
    for t in ['No TMS', 'Early TMS', 'Middle TMS']:
        for d in ['Radial', 'Tangential']:
            for s in data['subjID'].unique():
                subset = data[(data['TMS'] == t) & (data['Hemifield'] == hemi) & (data['DiagType'] == d) & (data['subjID'] == s)]
                if len(subset) < 10: continue
                try:
                    res = md.fit_model(subset['errOri_scaled'].dropna().values, 'mixtureModel', max_trials=100); mix_s.append({'TMS': t, 'Hemifield': hemi, 'DiagType': d, 'subjID': s, 'Precision': res[1]})
                except: pass
    df_mix = pd.DataFrame(mix_s); df_all = pd.merge(df_means, df_mix, on=['subjID', 'TMS', 'Hemifield', 'DiagType'], how='outer') if not df_mix.empty else df_means
    
    # 3 rows, 2 columns layout
    # Row 1: Left = Abs Error (Radial vs Tangential Overlay), Right = Precision (Radial vs Tangential Overlay)
    # Row 2: Left = Radial Abs Error (No TMS vs Early/Middle TMS), Right = Radial Precision (No TMS vs Early/Middle TMS)
    # Row 3: Left = Tangential Abs Error (No TMS vs Early/Middle TMS), Right = Tangential Precision (No TMS vs Early/Middle TMS)
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    
    # Row 1 (Overlay plots)
    for i, (met, label) in enumerate(metrics):
        ax = axes[0, i]
        piv = df_all.pivot_table(index=['subjID', 'TMS'], columns='DiagType', values=met, observed=False).reset_index()
        if 'Radial' not in piv.columns or 'Tangential' not in piv.columns: continue
        for t in ['No TMS', 'Early TMS', 'Middle TMS']:
            sub = piv[piv['TMS'] == t].dropna(subset=['Radial', 'Tangential'])
            if len(sub) == 0: continue
            t_stat, _ = stats.ttest_rel(sub['Tangential'].values, sub['Radial'].values)
            p_val = run_paired_permutation_test(sub['Tangential'].values, sub['Radial'].values, n_perms=10000)
            p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            label_text = f"{t} (t={t_stat:.2f}, {p_str})"
            ax.scatter(sub['Radial'], sub['Tangential'], color=palette[t], alpha=0.4, s=30, edgecolor='white')
            ax.errorbar(sub['Radial'].mean(), sub['Tangential'].mean(), xerr=sub['Radial'].sem(), yerr=sub['Tangential'].sem(), fmt='o', color=palette[t], markersize=10, markeredgecolor='white', label=label_text)
        
        all_v = np.concatenate([piv['Radial'].dropna().values, piv['Tangential'].dropna().values])
        if len(all_v) > 0:
            l, h = np.nanmin(all_v), np.nanmax(all_v)
            ax.plot([l, h], [l, h], 'k--', alpha=0.3)
        ax.set_aspect('equal', 'box')
        ax.set_title(f"{label} (Radial vs Tangential)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Radial")
        ax.set_ylabel("Tangential")
        ax.legend(fontsize=8, frameon=False, loc='upper left')
        sns.despine(ax=ax, offset=5)
        
    # Row 2 and Row 3 (Comparisons vs No TMS)
    for r_idx, d_type in enumerate(['Radial', 'Tangential'], start=1):
        for m_idx, (met, label) in enumerate(metrics):
            ax = axes[r_idx, m_idx]
            piv_tms = df_all.pivot_table(index='subjID', columns=['TMS', 'DiagType'], values=met, observed=False)
            
            col_no_tms = ('No TMS', d_type)
            if col_no_tms not in piv_tms.columns:
                continue
                
            all_v = []
            
            # Plot both Early TMS and Middle TMS
            for t_comp, color in [('Early TMS', 'r'), ('Middle TMS', 'b')]:
                col_comp = (t_comp, d_type)
                if col_comp in piv_tms.columns:
                    sub_df = piv_tms[[col_no_tms, col_comp]].dropna()
                    x_vals = sub_df[col_no_tms].values
                    y_vals = sub_df[col_comp].values
                    
                    if len(x_vals) > 0:
                        t_stat, _ = stats.ttest_rel(y_vals, x_vals)
                        p_val = run_paired_permutation_test(y_vals, x_vals, n_perms=10000)
                        p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
                        label_text = f"{t_comp} (t={t_stat:.2f}, {p_str})"
                        
                        ax.scatter(x_vals, y_vals, color=color, alpha=0.4, s=30, edgecolor='white')
                        ax.errorbar(x_vals.mean(), y_vals.mean(), xerr=stats.sem(x_vals), yerr=stats.sem(y_vals), fmt='o', color=color, markersize=10, markeredgecolor='white', label=label_text)
                        
                        all_v.extend(x_vals)
                        all_v.extend(y_vals)
                        
            if len(all_v) > 0:
                l, h = np.nanmin(all_v), np.nanmax(all_v)
                ax.plot([l, h], [l, h], 'k--', alpha=0.3)
                
            ax.set_aspect('equal', 'box')
            ax.set_title(f"{d_type} - {label}\n(TMS vs No TMS)", fontsize=11, fontweight='bold')
            ax.set_xlabel("No TMS")
            ax.set_ylabel("TMS")
            ax.legend(fontsize=8, frameon=False, loc='upper left')
            sns.despine(ax=ax, offset=5)
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.svg')))
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.pdf')))
    plt.close()

def plot_sat_grid(data, fig_title, filename, fig_folder=None):
    print(f"Generating Figure: {fig_title}...")
    save_dir = os.path.join(output_dir, fig_folder) if fig_folder else output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes_2d = plt.subplots(2, 2, figsize=(14, 12)); axes = axes_2d.flatten()
    tms_colors = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}
    
    # 0. RT Distribution KDE
    sns.kdeplot(data=data, x='RT', hue='TMS', palette=tms_colors, ax=axes[0], fill=False, linewidth=2, common_norm=False)
    axes[0].set_title('RT distribution', fontweight='bold', pad=15)
    axes[0].set_xlabel('Reaction Time (s)'); axes[0].set_ylabel('Density')
    
    # 1. Compile Smoothing (3-bin rolling average on 25 deciles)
    n_bins, window_size = 25, 3
    decile_data = []
    for (subj, tms), subj_df in data.groupby(['subjID', 'TMS']):
        if len(subj_df) < n_bins: continue
        subj_df = subj_df.copy(); subj_df['abs_initOffset'] = np.abs(subj_df['initOffset'])
        
        # RT deciles
        subj_df['RT_Decile'] = pd.qcut(subj_df['RT'], q=n_bins, labels=False, duplicates='drop')
        agg_rt = subj_df.groupby('RT_Decile').agg({'RT': 'mean', 'AbsError': 'mean'}).reset_index()
        agg_rt.update(agg_rt[['RT', 'AbsError']].rolling(window=window_size, center=True).mean())
        
        # Offset deciles
        subj_df['Offset_Decile'] = pd.qcut(subj_df['abs_initOffset'], q=n_bins, labels=False, duplicates='drop')
        agg_off = subj_df.groupby('Offset_Decile').agg({'abs_initOffset': 'mean', 'RT': 'mean', 'AbsError': 'mean'}).reset_index()
        agg_off.update(agg_off[['abs_initOffset', 'RT', 'AbsError']].rolling(window=3, center=True).mean())
        
        agg_rt['subjID'], agg_rt['TMS'] = subj, tms
        agg_off['subjID'], agg_off['TMS'] = subj, tms
        decile_data.append((agg_rt, agg_off))
    
    df_rt_all = pd.concat([d[0] for d in decile_data])
    df_off_all = pd.concat([d[1] for d in decile_data])
    df_rt_all['Group_RT'] = df_rt_all.groupby(['RT_Decile', 'TMS'])['RT'].transform('mean')
    df_off_all['Group_Offset'] = df_off_all.groupby(['Offset_Decile', 'TMS'])['abs_initOffset'].transform('mean')
    
    # Plot 1: Angular Error vs Reaction Time 
    sns.lineplot(data=df_rt_all, x='Group_RT', y='AbsError', hue='TMS', palette=tms_colors, errorbar=('ci', 95), marker=None, ax=axes[1], linewidth=3)
    axes[1].set_title('Angular Error vs Reaction Time', fontweight='bold', pad=15)
    axes[1].set_xlabel('Reaction Time (s)'); axes[1].set_ylabel('Angular Error (°)')
    
    # Plot 2: RT vs Absolute Init Offset
    sns.lineplot(data=df_off_all, x='Group_Offset', y='RT', hue='TMS', palette=tms_colors, errorbar=('ci', 95), marker=None, ax=axes[2], linewidth=3)
    axes[2].set_title('Reaction Time vs. Init Offset', fontweight='bold', pad=15)
    axes[2].set_xlabel('Absolute Init Offset (°)'); axes[2].set_ylabel('Reaction Time (s)')
    
    # Plot 3: Angular Error vs Absolute Init Offset
    sns.lineplot(data=df_off_all, x='Group_Offset', y='AbsError', hue='TMS', palette=tms_colors, errorbar=('ci', 95), marker=None, ax=axes[3], linewidth=3)
    axes[3].set_title('Angular Error vs. Init Offset', fontweight='bold', pad=15)
    axes[3].set_xlabel('Absolute Init Offset (°)'); axes[3].set_ylabel('Angular Error (°)')
    
    for ax in axes: ax.legend(fontsize=8, frameon=False); sns.despine(ax=ax, offset=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.svg')))
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.pdf')))
    plt.close()

def compute_sat_stats(data, filename):
    print("Computing SAT Statistics...")
    stats_list = []
    data = data.copy(); data['AbsInitOffset'] = np.abs(data['initOffset'])
    
    for t in ['No TMS', 'Early TMS', 'Middle TMS']:
        for s in data['subjID'].unique():
            sub = data[(data['TMS'] == t) & (data['subjID'] == s)]
            if len(sub) < 20: continue
            
            # Relationships
            r_err_off, _ = stats.spearmanr(sub['AbsInitOffset'], sub['AbsError'], nan_policy='omit')
            r_rt_off, _ = stats.spearmanr(sub['AbsInitOffset'], sub['RT'], nan_policy='omit')
            r_err_rt, _ = stats.spearmanr(sub['RT'], sub['AbsError'], nan_policy='omit')
            
            stats_list.append({'subjID': s, 'TMS': t, 'r_Error_Offset': r_err_off, 'r_RT_Offset': r_rt_off, 'r_Error_RT': r_err_rt})
    
    df_stats = pd.DataFrame(stats_list)
    results = []
    for t in ['No TMS', 'Early TMS', 'Middle TMS']:
        sub_t = df_stats[df_stats['TMS'] == t]
        for met in ['r_Error_Offset', 'r_RT_Offset', 'r_Error_RT']:
            vals = sub_t[met].dropna().values
            if len(vals) == 0: continue
            p = run_permutation_test(vals)
            results.append({'TMS': t, 'Metric': met, 'Mean_r': np.mean(vals), 'SEM_r': stats.sem(vals), 'p_perm': p, 'd': np.mean(vals)/np.std(vals)})
    
    res_df = pd.DataFrame(results)
    def format_p(p): return f"{p:.4e}" if p < 0.001 else f"{p:.4f}"
    res_df['p_perm'] = res_df['p_perm'].apply(format_p)
    for c in ['Mean_r', 'SEM_r', 'd']:
        if c in res_df.columns: res_df[c] = res_df[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)
    res_df.to_csv(os.path.join(output_dir, 'Fig3', filename), index=False)
    print(f"SAT Stats saved to: {os.path.join(output_dir, 'Fig3', filename)}")
    return res_df

# --- Motor Preference Figure ---
# (Triggered at the end of script)

# MAIN CALLS
# Fig 1 & 1A
# plot_contrast_grid(df, ('inPF', 'outPF'), "1. Selective TMS Effect: (inPF - outPF) / (inPF + outPF)", "Fig1_ContrastIndex_Hemifield.png", metrics_subset=[('AbsError', 'Abs Error'), ('RT', 'Reaction Time'), ('Precision', 'Precision (K)')], fig_folder='Fig1')
# plot_scatter_in_vs_out(df, "1A. Individual Performance: outPF vs inPF", "Fig1A_Scatter_in_vs_out.png", metrics_subset=[('AbsError', 'Abs Error'), ('RT', 'Reaction Time'), ('Precision', 'Precision (K)')], fig_folder='Fig1')

# Fig 2 suite
metrics_fig2 = [('AbsError', 'Abs Error'), ('Precision', 'Precision (K)')]
plot_contrast_grid(df, ('Radial', 'Tangential'), "2. Radial Sensitivity: (Radial - Tangential) / (Radial + Tangential)", "Fig2_ContrastIndex_DiagType.png", paired=False, metrics_subset=metrics_fig2, fig_folder='Fig2')
plot_scatter_diag_contrast_in_vs_out(df, "2A. Radial Sensitivity: outPF vs inPF", "Fig2A_Scatter_diag_contrast_in_vs_out.png", metrics_subset=metrics_fig2, fig_folder='Fig2')
plot_scatter_radial_vs_tangential_overlay(df, "2B. Radial Sensitivity Overlay", "Fig2B_Scatter_Radial_vs_Tangential.png", metrics_subset=metrics_fig2, fig_folder='Fig2')

def _draw_overlay_boxes(ax, data_dict, tms_groups, A, B, show_legend=False):
    """4 boxes per TMS group grouped by hemifield:
      inPF-Radial (solid) | inPF-Tangential (solid lighter)  [mid_gap]
      outPF-Radial (hatched) | outPF-Tangential (hatched lighter)
    """
    bw         = 0.22   # box width
    inner_gap  = 0.04   # gap between Radial and Tangential within the same hemifield
    mid_gap    = 0.14   # gap between inPF pair and outPF pair
    tms_gap    = 0.30   # extra space between consecutive TMS groups

    step = 4 * bw + inner_gap + mid_gap + tms_gap   # full width of one TMS group
    pos_map = {}
    xtick_pos, xtick_lbl = [], []

    for i, t in enumerate(tms_groups):
        base = i * step
        # inPF pair first, outPF pair second — mid_gap between them
        pos_map[f'{t}|{A}|inPF']  = base
        pos_map[f'{t}|{B}|inPF']  = base + bw + inner_gap
        pos_map[f'{t}|{A}|outPF'] = base + 2*bw + inner_gap + mid_gap
        pos_map[f'{t}|{B}|outPF'] = base + 3*bw + 2*inner_gap + mid_gap
        # x-tick at the centre of the 4 boxes
        xtick_pos.append(base + 1.5*bw + inner_gap + mid_gap / 2)
        xtick_lbl.append(t)

    rng = np.random.default_rng(42)
    key_order = [f'{t}|{c}|{h}' for t in tms_groups for h in ['inPF', 'outPF'] for c in [A, B]]
    for key in key_order:
        pos  = pos_map.get(key)
        if pos is None: continue
        t, cond, hemi = key.split('|')
        vals = data_dict.get(key, np.array([]))
        if len(vals) < 3: continue

        is_A  = (cond == A)
        col   = TMS_BOX_PAL[t]
        scol  = TMS_STRIP_PAL[t]
        al    = 1.0 if is_A else 0.65
        sal   = 0.7 if is_A else 0.35
        hatch = None if hemi == 'inPF' else '///'

        ax.boxplot([vals], positions=[pos], widths=bw, patch_artist=True,
                   showfliers=False, manage_ticks=False,
                   boxprops=dict(facecolor=col, alpha=al, linewidth=0.8,
                                 edgecolor=scol, hatch=hatch),
                   medianprops=dict(color='white', linewidth=2),
                   whiskerprops=dict(color=scol, alpha=al, linewidth=0.8),
                   capprops=dict(color=scol, alpha=al, linewidth=0.8))
        jitter = rng.uniform(-bw * 0.28, bw * 0.28, len(vals))
        ax.scatter(pos + jitter, vals, color=scol, alpha=sal, s=10, zorder=3)

    ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=9)
    xl = pos_map[key_order[0]] - bw * 0.8
    xr = pos_map[key_order[-1]] + bw * 1.8
    ax.set_xlim(xl, xr)
    if ax.get_legend(): ax.get_legend().remove()
    if show_legend:
        nc = TMS_BOX_PAL['No TMS']
        handles = [
            mpatches.Patch(facecolor=nc, alpha=1.0,  edgecolor='k', lw=0.8, hatch=None,  label=f'inPF · {A}'),
            mpatches.Patch(facecolor=nc, alpha=0.65, edgecolor='k', lw=0.8, hatch=None,  label=f'inPF · {B}'),
            mpatches.Patch(facecolor=nc, alpha=1.0,  edgecolor='k', lw=0.8, hatch='///', label=f'outPF · {A}'),
            mpatches.Patch(facecolor=nc, alpha=0.65, edgecolor='k', lw=0.8, hatch='///', label=f'outPF · {B}'),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper right')


def plot_contrast_grid_by_hemifield(data, fig_title, filename, metrics_subset=None, fig_folder=None):
    """Fig 2C: Radial vs Tangential contrast overlaid for inPF (solid) and outPF (hatched).
    1 row × N-metric columns — 4 boxes per TMS group."""
    print(f"Generating Figure: {fig_title}...")
    save_dir = os.path.join(output_dir, fig_folder) if fig_folder else output_dir
    os.makedirs(save_dir, exist_ok=True)

    metrics    = metrics_subset if metrics_subset else [('AbsError', 'Abs Error'), ('Precision', 'Precision (K)')]
    hemifields = ['inPF', 'outPF']
    tms_groups = ['No TMS', 'Early TMS', 'Middle TMS']
    A, B       = 'Radial', 'Tangential'

    n_cols = len(metrics)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), squeeze=False)
    stats_rows = []

    # ── Aggregate data for BOTH hemifields ──────────────────────────────────
    all_results = {}   # {hemi: list of pivot DataFrames (one per metric)}
    for hemi in hemifields:
        hemi_data = data[data['Hemifield'] == hemi].copy()
        g_cols    = ['subjID', 'TMS', 'DiagType']
        df_s      = hemi_data.groupby(g_cols, observed=False)[
                        ['AbsError', 'RT', 'BiasTowardsDiagonal']].mean().reset_index()
        mix_s = []
        for t in tms_groups:
            for d in [A, B]:
                for subj in hemi_data['subjID'].unique():
                    mask = ((hemi_data['TMS'] == t) & (hemi_data['DiagType'] == d) &
                            (hemi_data['subjID'] == subj))
                    sub  = hemi_data[mask]
                    if len(sub) < 10: continue
                    try:
                        res = md.fit_model(sub['errOri_scaled'].dropna().values,
                                           'mixtureModel', max_trials=100)
                        mix_s.append({'TMS': t, 'DiagType': d, 'subjID': subj,
                                      'Precision': res[1]})
                    except: pass
        df_m   = pd.DataFrame(mix_s)
        df_all = pd.merge(df_s, df_m, on=g_cols, how='outer') if not df_m.empty else df_s

        results_list = []
        for met, lab in metrics:
            if met not in df_all.columns: continue
            piv = df_all.pivot_table(index=['subjID', 'TMS'], columns='DiagType',
                                     values=met, observed=False).reset_index()
            if A in piv.columns and B in piv.columns:
                av_d  = np.abs(piv[A]) if met == 'BiasTowardsDiagonal' else piv[A]
                bv_d  = np.abs(piv[B]) if met == 'BiasTowardsDiagonal' else piv[B]
                total = av_d + bv_d + 1e-12
                piv[f'{A}_norm'] = av_d / total
                piv[f'{B}_norm'] = bv_d / total
                piv['Metric']    = lab
                results_list.append(piv)
        all_results[hemi] = results_list

    # ── Plot: one column per metric ─────────────────────────────────────────
    for im, (met, lab) in enumerate(metrics):
        ax = axes[0, im]

        # Build data_dict: key = '{tms}|{cond}|{hemi}' → array of norm values
        data_dict = {}
        for hemi in hemifields:
            match = [d for d in all_results[hemi] if d['Metric'].iloc[0] == lab]
            if not match: continue
            piv = match[0]
            for t in tms_groups:
                sub = piv[piv['TMS'] == t].dropna(subset=[f'{A}_norm', f'{B}_norm'])
                data_dict[f'{t}|{A}|{hemi}'] = sub[f'{A}_norm'].values
                data_dict[f'{t}|{B}|{hemi}'] = sub[f'{B}_norm'].values

        _draw_overlay_boxes(ax, data_dict, tms_groups, A, B, show_legend=(im == 0))
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel('Normalized Value')
        if lab == 'Abs Error':
            ax.set_ylim(0.35, 0.65)
        elif 'Precision' in lab:
            ax.set_ylim(0.2, 0.8)
        ax.set_xlabel('')
        sns.despine(ax=ax, offset=5)

        # ── Stats: paired Radial_vs_Tangential per hemifield × TMS ──────────
        for hemi in hemifields:
            match = [d for d in all_results[hemi] if d['Metric'].iloc[0] == lab]
            if not match: continue
            piv = match[0]
            for t in tms_groups:
                sub      = piv[piv['TMS'] == t].dropna(subset=[f'{A}_norm', f'{B}_norm'])
                v_a      = sub[f'{A}_norm'].dropna().values
                v_b      = sub[f'{B}_norm'].dropna().values
                n_paired = min(len(v_a), len(v_b))
                if n_paired > 5:
                    p_pair    = run_paired_permutation_test(v_a[:n_paired], v_b[:n_paired])
                    diff_mean = np.mean(v_a[:n_paired] - v_b[:n_paired])
                    diff_std  = np.std(v_a[:n_paired] - v_b[:n_paired])
                    t_pair, _ = stats.ttest_rel(v_a[:n_paired], v_b[:n_paired])
                    stats_rows.append({
                        'Hemifield': hemi, 'Metric': lab,
                        'Condition': f'{A}_vs_{B}', 'TMS': t, 'Bin': None,
                        'Mean': diff_mean,
                        'SEM':  stats.sem(v_a[:n_paired] - v_b[:n_paired]),
                        'p_perm':  p_pair,
                        'Cohen_d': diff_mean / diff_std if diff_std > 0 else np.nan,
                        't_stat':  t_pair, 'N': n_paired,
                    })

    plt.suptitle(fig_title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.svg')), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, filename.replace('.png', '.pdf')), bbox_inches='tight')
    plt.close()

    if stats_rows:
        sdf = pd.DataFrame(stats_rows)
        def format_p(p): return f"{p:.4e}" if p < 0.001 else f"{p:.4f}"
        sdf['p_perm'] = sdf['p_perm'].apply(format_p)
        for c in ['Mean', 'SEM', 'Cohen_d']:
            if c in sdf.columns:
                sdf[c] = sdf[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)
        sdf.to_csv(os.path.join(save_dir, filename.replace('.png', '_Stats.csv')), index=False)

# Fig 2C: Radial vs Tangential overlaid — inPF solid, outPF hatched
plot_contrast_grid_by_hemifield(df, "2C. Radial Sensitivity by Hemifield (inPF solid / outPF hatched)", "Fig2C_ContrastIndex_DiagType_byHemifield.png", metrics_subset=metrics_fig2, fig_folder='Fig2')



# Fig 3: SAT & Motor Preference
#plot_sat_grid(df, "3. Speed-Accuracy Tradeoff Dashboard", "Fig3_SAT.png", fig_folder='Fig3')
#compute_sat_stats(df, "Fig3_SAT_Stats.csv")

# Fig 4: Threshold Sensitivity (No TMS)
# Fig 4: Threshold Sensitivity (Sliding Window)
def process_sliding_window(start, data, width=30):
    end = start + width
    print(f"  Processing window: [{start}, {end}]...")
    mask = (data['AbsInitOffset'] >= start) & (data['AbsInitOffset'] < end)
    sub = data[mask].copy()
    
    thresh_results = []
    # Hemifield
    h_data = plot_contrast_grid(sub, ('inPF', 'outPF'), "", "", metrics_subset=[('AbsError', 'Accuracy'), ('Precision', 'Precision')], return_data=True)
    if h_data is not None and not h_data.empty:
        h_data = h_data.reset_index(drop=True).copy()
        h_data['Type'] = 'Hemifield (HCI)'
        h_data['WindowStart'] = start
        thresh_results.append(h_data)
    
    # Radial Sensitivity
    r_data = plot_contrast_grid(sub, ('Radial', 'Tangential'), "", "", metrics_subset=[('AbsError', 'Accuracy'), ('Precision', 'Precision')], return_data=True)
    if r_data is not None and not r_data.empty:
        r_data = r_data.reset_index(drop=True).copy()
        r_data['Type'] = 'Radial Sensitivity (RSI)'
        r_data['WindowStart'] = start
        thresh_results.append(r_data)
        
    return pd.concat(thresh_results) if thresh_results else pd.DataFrame()

def plot_threshold_sensitivity(data):
    print("Generating Figure 4: Sliding Window Analysis (Parallel, All TMS)...")
    window_starts = np.arange(0, 61, 5)
    
    all_subject_data = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(process_sliding_window)(start, data) for start in window_starts
    )
    
    full_df = pd.concat(all_subject_data).reset_index(drop=True)
    if full_df.empty: return
    full_df['WindowCenter'] = full_df['WindowStart'] + 15
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    types = ['Hemifield (HCI)', 'Radial Sensitivity (RSI)']
    metrics = ['Accuracy', 'Precision']
    tms_colors = {'No TMS': '#E0E0E0', 'Early TMS': '#FFCDD2', 'Middle TMS': '#BBDEFB'}
    
    for r, t in enumerate(types):
        for c, m in enumerate(metrics):
            ax = axes[r, c]
            sub = full_df[(full_df['Type'] == t) & (full_df['Metric'] == m)]
            
            if not sub.empty:
                sns.lineplot(data=sub, x='WindowCenter', y='CI', hue='TMS', ax=ax, palette=tms_colors, 
                             hue_order=['No TMS', 'Early TMS', 'Middle TMS'], errorbar='se', marker='o', linewidth=2)
                ax.axhline(0, color='k', linestyle='--', lw=1, alpha=0.5)
            
            ax.set_title(f"{t}: {m}", fontweight='bold')
            ax.set_ylabel("Contrast Index")
            if r == 1: ax.set_xlabel("Window Center (deg)")
            else: ax.set_xlabel("")
            if ax.get_legend(): ax.get_legend().remove()
            sns.despine(ax=ax, offset=10)
        
    # Add a single legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Save Fig 4A
    save_dir = os.path.join(output_dir, 'Fig4')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'Fig4A_SlidingWindow.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'Fig4A_SlidingWindow.svg'))
    plt.savefig(os.path.join(save_dir, 'Fig4A_SlidingWindow.pdf'))
    plt.close()
    print(f"Fig 4A saved to: {save_dir}")

def generate_figure_4(data):
    # Panel A: Sliding Window
    # plot_threshold_sensitivity(data)
    
    # Filter for Small Offsets (< 45 degrees)
    sub_45 = data[data['AbsInitOffset'] < 45].copy()
    print("\n--- Generating Figure 4B & 4C (Threshold < 45 deg) ---")
    
    save_dir = os.path.join(output_dir, 'Fig4')
    os.makedirs(save_dir, exist_ok=True)
    tms_groups = ['No TMS', 'Early TMS', 'Middle TMS']
    tms_colors = {'No TMS': '#E0E0E0', 'Early TMS': '#FFCDD2', 'Middle TMS': '#BBDEFB'}
    metrics_4b = [('AbsError', 'Accuracy'), ('Precision', 'Precision')]
    stats_rows = []

    # Get aggregated data
    h_data = plot_contrast_grid(sub_45, ('inPF', 'outPF'), "", "", metrics_subset=metrics_4b, return_data=True)
    r_data = plot_contrast_grid(sub_45, ('Radial', 'Tangential'), "", "", metrics_subset=metrics_4b, return_data=True)

    # Panel 4B: Consolidated Boxplots (2x2) — normalized ratios
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ic, (met_col, lab) in enumerate(metrics_4b):

        # HCI Row
        ax_h = axes[0, ic]
        sub_h = h_data[h_data['Metric'] == lab]
        long_h = []
        for t in tms_groups:
            s = sub_h[sub_h['TMS'] == t].dropna(subset=['inPF_norm', 'outPF_norm'])
            for _, row in s.iterrows():
                long_h.append({'TMS': t, 'Condition': 'inPF',  'TMS_Cond': f'{t}|inPF',  'NormValue': row['inPF_norm'],  'subjID': row['subjID']})
                long_h.append({'TMS': t, 'Condition': 'outPF', 'TMS_Cond': f'{t}|outPF', 'NormValue': row['outPF_norm'], 'subjID': row['subjID']})
        long_h_df = pd.DataFrame(long_h)
        x_h = [f'{t}|{c}' for t in tms_groups for c in ['inPF', 'outPF']]
        _draw_paired_boxes(ax_h, long_h_df, x_h, 'inPF', 'outPF', tms_groups,
                           show_legend=(ic == 0), leg_labels=('inPF', 'outPF'))
        # Titles based on metric and row
        h_title = f"{'Abs Error' if lab == 'Accuracy' else lab} (spatial differences)"
        r_title = f"{'Abs Error' if lab == 'Accuracy' else lab} (radial sensitivity)"
        
        ax_h.set_title(h_title); ax_h.set_ylabel("Normalized Value"); ax_h.set_ylim(0.2, 0.8); ax_h.set_xlabel('')
        for it, t in enumerate(tms_groups):
            s = sub_h[sub_h['TMS'] == t].dropna(subset=['inPF_norm', 'outPF_norm'])
            for col_key, cond in [('inPF_norm', 'inPF'), ('outPF_norm', 'outPF')]:
                v = s[col_key].dropna().values
                if len(v) > 5:
                    diffs = v - 0.5
                    p = run_permutation_test(diffs); mean_v, sem_v = np.mean(v), stats.sem(v); d = np.mean(diffs)/np.std(diffs)
                    stats_rows.append({'Panel': '4B', 'Type': 'HCI', 'Condition': cond, 'Metric': lab, 'TMS': t, 'Mean': mean_v, 'SEM': sem_v, 'p': p, 'd': d})
        sns.despine(ax=ax_h, offset=5)

        # RSI Row
        ax_r = axes[1, ic]
        sub_r = r_data[r_data['Metric'] == lab]
        long_r = []
        for t in tms_groups:
            s = sub_r[sub_r['TMS'] == t].dropna(subset=['Radial_norm', 'Tangential_norm'])
            for _, row in s.iterrows():
                long_r.append({'TMS': t, 'Condition': 'Radial',     'TMS_Cond': f'{t}|Radial',     'NormValue': row['Radial_norm'],     'subjID': row['subjID']})
                long_r.append({'TMS': t, 'Condition': 'Tangential', 'TMS_Cond': f'{t}|Tangential', 'NormValue': row['Tangential_norm'], 'subjID': row['subjID']})
        long_r_df = pd.DataFrame(long_r)
        x_r = [f'{t}|{c}' for t in tms_groups for c in ['Radial', 'Tangential']]
        _draw_paired_boxes(ax_r, long_r_df, x_r, 'Radial', 'Tangential', tms_groups,
                           show_legend=(ic == 0), leg_labels=('Radial', 'Tangential'))
        ax_r.set_title(r_title); ax_r.set_ylabel("Normalized Value"); ax_r.set_ylim(0.2, 0.8); ax_r.set_xlabel('')
        for it, t in enumerate(tms_groups):
            s = sub_r[sub_r['TMS'] == t].dropna(subset=['Radial_norm', 'Tangential_norm'])
            for col_key, cond in [('Radial_norm', 'Radial'), ('Tangential_norm', 'Tangential')]:
                v = s[col_key].dropna().values
                if len(v) > 5:
                    diffs = v - 0.5
                    p = run_permutation_test(diffs); mean_v, sem_v = np.mean(v), stats.sem(v); d = np.mean(diffs)/np.std(diffs)
                    stats_rows.append({'Panel': '4B', 'Type': 'RSI', 'Condition': cond, 'Metric': lab, 'TMS': t, 'Mean': mean_v, 'SEM': sem_v, 'p': p, 'd': d})
        sns.despine(ax=ax_r, offset=5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig4B_Boxplots.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'Fig4B_Boxplots.svg'))
    plt.savefig(os.path.join(save_dir, 'Fig4B_Boxplots.pdf'))
    plt.close()

    # Panel 4C: Combined 2x2 (Row 0 = HCI like Fig 1B, Row 1 = RSI like Fig 2C)
    print("Generating Figure 4C...")
    metrics_4c = [('AbsError', 'Accuracy'), ('Precision', 'Precision')]
    palette = {'No TMS': 'k', 'Early TMS': 'r', 'Middle TMS': 'b'}
    tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']

    # Data prep: HCI (same as plot_scatter_in_vs_out)
    df_hci_means = sub_45.groupby(['subjID', 'TMS', 'Hemifield'], observed=False)[['AbsError', 'RT']].mean().reset_index()
    mix_hci = []
    for t in tms_conds:
        for v in ['inPF', 'outPF']:
            for s in sub_45['subjID'].unique():
                sub = sub_45[(sub_45['TMS'] == t) & (sub_45['Hemifield'] == v) & (sub_45['subjID'] == s)]
                if len(sub) < 10: continue
                try:
                    res = md.fit_model(sub['errOri_scaled'].dropna().values, 'mixtureModel', max_trials=100)
                    mix_hci.append({'TMS': t, 'Hemifield': v, 'subjID': s, 'Precision': res[1]})
                except: pass
    df_hci = pd.merge(df_hci_means, pd.DataFrame(mix_hci), on=['subjID', 'TMS', 'Hemifield'], how='outer') if mix_hci else df_hci_means

    # Data prep: RSI (same as plot_scatter_radial_vs_tangential_overlay)
    df_rsi_means = sub_45.groupby(['subjID', 'TMS', 'DiagType'], observed=False)[['AbsError', 'RT']].mean().reset_index()
    mix_rsi = []
    for t in tms_conds:
        for d in ['Radial', 'Tangential']:
            for s in sub_45['subjID'].unique():
                sub = sub_45[(sub_45['TMS'] == t) & (sub_45['DiagType'] == d) & (sub_45['subjID'] == s)]
                if len(sub) < 10: continue
                try:
                    res = md.fit_model(sub['errOri_scaled'].dropna().values, 'mixtureModel', max_trials=100)
                    mix_rsi.append({'TMS': t, 'DiagType': d, 'subjID': s, 'Precision': res[1]})
                except: pass
    df_rsi = pd.merge(df_rsi_means, pd.DataFrame(mix_rsi), on=['subjID', 'TMS', 'DiagType'], how='outer') if mix_rsi else df_rsi_means

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ic, (met, lab) in enumerate(metrics_4c):
        # Row 0: HCI — inPF vs outPF
        ax = axes[0, ic]
        piv = df_hci.pivot_table(index=['subjID', 'TMS'], columns='Hemifield', values=met, observed=False).reset_index()
        if 'inPF' in piv.columns and 'outPF' in piv.columns:
            for t in tms_conds:
                sub = piv[piv['TMS'] == t].dropna(subset=['inPF', 'outPF'])
                if sub.empty: continue
                count = (sub['inPF'] < sub['outPF']).sum() if met == 'Precision' else (sub['inPF'] > sub['outPF']).sum()
                sym = '<' if met == 'Precision' else '>'
                ax.scatter(sub['outPF'], sub['inPF'], color=palette[t], alpha=0.4, s=30, edgecolor='white')
                ax.errorbar(sub['outPF'].mean(), sub['inPF'].mean(), xerr=sub['outPF'].sem(), yerr=sub['inPF'].sem(),
                            fmt='o', color=palette[t], markersize=10, markeredgecolor='white',
                            label=f"{t} (n={count} in{sym}out)")
            v = piv[['inPF', 'outPF']].values.flatten(); lo, hi = np.nanmin(v), np.nanmax(v)
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3)
        ax.set_aspect('equal', 'box'); ax.set_title(f"HCI: {lab}"); ax.set_xlabel("outPF"); ax.set_ylabel("inPF")
        ax.legend(fontsize=8, frameon=False, loc='upper left'); sns.despine(ax=ax, offset=10)

        # Row 1: RSI — Radial vs Tangential
        ax = axes[1, ic]
        piv = df_rsi.pivot_table(index=['subjID', 'TMS'], columns='DiagType', values=met, observed=False).reset_index()
        if 'Radial' in piv.columns and 'Tangential' in piv.columns:
            for t in tms_conds:
                sub = piv[piv['TMS'] == t].dropna(subset=['Radial', 'Tangential'])
                if sub.empty: continue
                count = (sub['Radial'] > sub['Tangential']).sum() if met == 'Precision' else (sub['Radial'] < sub['Tangential']).sum()
                sym = '>' if met == 'Precision' else '<'
                ax.scatter(sub['Radial'], sub['Tangential'], color=palette[t], alpha=0.4, s=30, edgecolor='white')
                ax.errorbar(sub['Radial'].mean(), sub['Tangential'].mean(), xerr=sub['Radial'].sem(), yerr=sub['Tangential'].sem(),
                            fmt='o', color=palette[t], markersize=10, markeredgecolor='white',
                            label=f"{t} (n={count} rad{sym}tang)")
            all_v = np.concatenate([piv['Radial'].dropna().values, piv['Tangential'].dropna().values])
            lo, hi = np.nanmin(all_v), np.nanmax(all_v)
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3)
        ax.set_aspect('equal', 'box'); ax.set_title(f"RSI: {lab}"); ax.set_xlabel("Radial"); ax.set_ylabel("Tangential")
        ax.legend(fontsize=8, frameon=False, loc='upper left'); sns.despine(ax=ax, offset=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig4C_Scatter.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'Fig4C_Scatter.svg'))
    plt.savefig(os.path.join(save_dir, 'Fig4C_Scatter.pdf'))
    plt.close()

    # Panel 4D: 2 rows comparison plots (Row 0 = Radial, Row 1 = Tangential)
    print("Generating Figure 4D...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    metrics_4d = [('AbsError', 'Abs Error'), ('Precision', 'Precision (K)')]
    
    for r_idx, d_type in enumerate(['Radial', 'Tangential']):
        for m_idx, (met, label) in enumerate(metrics_4d):
            ax = axes[r_idx, m_idx]
            df_rsi_filtered = df_rsi[df_rsi['DiagType'] == d_type]
            piv_tms = df_rsi_filtered.pivot_table(index='subjID', columns='TMS', values=met, observed=False)
            
            if 'No TMS' not in piv_tms.columns:
                continue
                
            all_v = []
            
            # Plot both Early TMS and Middle TMS
            for t_comp, color in [('Early TMS', 'r'), ('Middle TMS', 'b')]:
                if t_comp in piv_tms.columns:
                    sub_df = piv_tms[['No TMS', t_comp]].dropna()
                    x_vals = sub_df['No TMS'].values
                    y_vals = sub_df[t_comp].values
                    
                    if len(x_vals) > 0:
                        t_stat, _ = stats.ttest_rel(y_vals, x_vals)
                        p_val = run_paired_permutation_test(y_vals, x_vals, n_perms=10000)
                        p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
                        label_text = f"{t_comp} (t={t_stat:.2f}, {p_str})"
                        
                        ax.scatter(x_vals, y_vals, color=color, alpha=0.4, s=30, edgecolor='white')
                        ax.errorbar(x_vals.mean(), y_vals.mean(), xerr=stats.sem(x_vals), yerr=stats.sem(y_vals), fmt='o', color=color, markersize=10, markeredgecolor='white', label=label_text)
                        
                        all_v.extend(x_vals)
                        all_v.extend(y_vals)
                        
            if len(all_v) > 0:
                l, h = np.nanmin(all_v), np.nanmax(all_v)
                ax.plot([l, h], [l, h], 'k--', alpha=0.3)
                
            ax.set_aspect('equal', 'box')
            ax.set_title(f"{d_type} - {label}\n(TMS vs No TMS)", fontsize=11, fontweight='bold')
            ax.set_xlabel("No TMS")
            ax.set_ylabel("TMS")
            ax.legend(fontsize=8, frameon=False, loc='upper left')
            sns.despine(ax=ax, offset=5)
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig4D_Scatter.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'Fig4D_Scatter.svg'))
    plt.savefig(os.path.join(save_dir, 'Fig4D_Scatter.pdf'))
    plt.close()
    
    pd.DataFrame(stats_rows).to_csv(os.path.join(save_dir, 'Fig4_Stats.csv'), index=False)
    print(f"Fig 4 panels and stats saved to: {save_dir}")
generate_figure_4(df)

# print("\n--- Triggering V05 Motor Preference Analysis ---")
# exec(open("V05_oriRotation.py").read())
