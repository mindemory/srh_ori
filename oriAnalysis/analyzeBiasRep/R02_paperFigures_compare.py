"""
R02_paperFigures_compare.py

Side-by-side comparison of Study 1 (Dake, George, Serfaty & Curtis 2026) and
the bias replication for every manuscript figure, plus diagnostics. All
quantities come from study_results.compute(), which reproduces
analyzeBehav/V10_paperFigures.py (validated against the published Study 1
values). Radial/Tangential labels follow the paper (Study 1: hemisphere from
V1ori_TMS_meta.csv). The geometry analyses are in R03_accuracy_geometry.py.

Usage: python R02_paperFigures_compare.py [--force]   (--force recomputes the cache)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

from biasrep_common import COMPARE_DIR, run_permutation_test
import panels as P
from study_results import compute

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
np.random.seed(0)
force = '--force' in sys.argv

R1 = compute('orig', force=force)
R2 = compute('rep', force=force)
OUT = COMPARE_DIR
T1 = f"Study 1 (N={R1['N']})"
T2 = f"Replication (N={R2['N']})"
STUDY_PAIR = [(R1, T1), (R2, T2)]


def lims_of(*vals_cols):
    v = np.concatenate([np.ravel(x) for x in vals_cols]); v = v[~np.isnan(v)]
    lo, hi = v.min(), v.max(); pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def letter(ax, s):
    ax.set_title(s, loc='left', fontweight='bold', fontsize=13)


def grid_2col(nrows, height_per_row=5.0):
    fig = plt.figure(figsize=(12, height_per_row * nrows))
    gs = GridSpec(nrows, 2, figure=fig, hspace=0.55, wspace=0.45)
    return fig, [[fig.add_subplot(gs[r, c]) for r in range(nrows)] for c in range(2)]


# ---------------------------------------------------------------------------
# Figure 2: phosphene-field effects
# ---------------------------------------------------------------------------
print('Figure 2...')
fig, cols = grid_2col(4)
for (R, title), a in zip(STUDY_PAIR, cols):
    g = R['groups']
    P.paired_boxes(a[0], R['piv'][('hemi', 'AbsError')], 'inPF', 'outPF', g, ylim=(0.38, 0.62), ylabel='Normalized\nAbsolute Error')
    P.scatter_pair(a[1], R['vals_hemi'], 'Hemifield', 'inPF', 'outPF', 'AbsError', g, 'Abs. Error (°), outPF', 'Abs. Error (°), inPF',
                   lims=lims_of(R1['vals_hemi']['AbsError'], R2['vals_hemi']['AbsError']))
    P.paired_boxes(a[2], R['piv'][('hemi', 'Precision')], 'inPF', 'outPF', g, ylim=(0.28, 0.72), ylabel='Normalized\nPrecision (κ)')
    P.scatter_pair(a[3], R['vals_hemi'], 'Hemifield', 'inPF', 'outPF', 'Precision', g, 'Precision (κ), outPF', 'Precision (κ), inPF',
                   lims=lims_of(R1['vals_hemi']['Precision'], R2['vals_hemi']['Precision']))
    for ax, l in zip(a, 'ABCD'): letter(ax, l)
    P.column_titles([a[0]], [title])
fig.suptitle('Figure 2 — Phosphene field effects\n(numbers under boxes: inPF−outPF index and the paper\'s permutation p)', y=0.935)
P.save(fig, OUT, 'C_Fig2_PhospheneField')

# ---------------------------------------------------------------------------
# Figure 3: radial sensitivity
# ---------------------------------------------------------------------------
print('Figure 3...')
fig, cols = grid_2col(4)
err_l = lims_of(R1['vals_rad']['AbsError'], R2['vals_rad']['AbsError'])
kap_l = lims_of(R1['vals_rad']['Precision'], R2['vals_rad']['Precision'])
for (R, title), a in zip(STUDY_PAIR, cols):
    g = R['groups']
    P.paired_boxes(a[0], R['piv'][('rad', 'AbsError')], 'Radial', 'Tangential', g, ylim=(0.38, 0.62), ylabel='Normalized\nAbsolute Error')
    P.scatter_pair(a[1], R['vals_rad'], 'DiagType', 'Radial', 'Tangential', 'AbsError', g, 'Abs. Error (°), Radial',
                   'Abs. Error (°), Tangential', x_is_A=True, lims=err_l)
    P.paired_boxes(a[2], R['piv'][('rad', 'Precision')], 'Radial', 'Tangential', g, ylim=(0.28, 0.72), ylabel='Normalized\nPrecision (κ)')
    P.scatter_pair(a[3], R['vals_rad'], 'DiagType', 'Radial', 'Tangential', 'Precision', g, 'Precision (κ), Radial',
                   'Precision (κ), Tangential', x_is_A=True, lims=kap_l)
    for ax, l in zip(a, 'ABCD'): letter(ax, l)
    P.column_titles([a[0]], [title])
fig.suptitle('Figure 3 — Radial sensitivity\n(numbers under boxes: radial index (R−T)/(R+T) and the paper\'s permutation p)', y=0.935)
P.save(fig, OUT, 'C_Fig3_RadialSensitivity')

# ---------------------------------------------------------------------------
# Figure 4: probe angle and response dynamics
# ---------------------------------------------------------------------------
print('Figure 4...')
fig, cols = grid_2col(4)
for (R, title), a in zip(STUDY_PAIR, cols):
    g = R['groups']
    for t in g:
        col, lab = P.TMS_LINE_PAL[t], P.TMS_SHORT[t]
        sns.lineplot(data=R['df_off'][R['df_off']['TMS'] == t], x='Group_Offset', y='RT', color=col, errorbar=('ci', 95), ax=a[0], lw=2, label=lab)
        sns.lineplot(data=R['df_rt'][R['df_rt']['TMS'] == t], x='Group_RT', y='AbsError', color=col, errorbar=('ci', 95), ax=a[1], lw=2, label=lab)
        sns.lineplot(data=R['df_off'][R['df_off']['TMS'] == t], x='Group_Offset', y='AbsError', color=col, errorbar=('ci', 95), ax=a[2], lw=2, label=lab)
        p = R['pcw'][R['pcw']['TMS'] == t].sort_values('X')
        a[3].plot(p['X'], p['Pop_Mean'], color=col, lw=1.8, label=lab)
        a[3].fill_between(p['X'], p['Pop_Mean'] - p['Pop_SEM'], p['Pop_Mean'] + p['Pop_SEM'], color=col, alpha=0.2)
    a[0].set(xlabel='Absolute Δ Probe Angle (°)', ylabel='Response Time (s)', xlim=(0, 90), ylim=(1.1, 2.4))
    a[1].set(xlabel='Response Time (s)', ylabel='Angular Error (°)', xlim=(0.9, 2.8), ylim=(4.5, 9.5))
    a[2].set(xlabel='Absolute Δ Probe Angle (°)', ylabel='Angular Error (°)', xlim=(0, 90), ylim=(4.5, 9.5))
    a[3].axvspan(-45, 45, color='0.88', zorder=0); a[3].axhline(0.5, color='k', ls='--', lw=1); a[3].axvline(0, color='k', ls='--', lw=1)
    a[3].set(xlabel='Signed Δ Probe Angle (°)', ylabel='Proportion of CW knob rotations', xlim=(-90, 90), ylim=(0, 1))
    sat = R['sat']
    for ax, met in zip(a[:3], ['Offset_RT', 'RT_Error', 'Offset_Error']):
        txt = '\n'.join(f"{P.TMS_SHORT[t]}: ρ={sat[sat.TMS == t][met].mean():.3f}, {P.fmt_p(run_permutation_test(sat[sat.TMS == t][met].values))}" for t in g)
        ax.text(0.98, 0.03, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    for ax, l in zip(a, 'ABCD'):
        letter(ax, l); sns.despine(ax=ax, offset=5)
        if ax.get_legend(): ax.legend(fontsize=7, frameon=False, loc='upper left')
    P.column_titles([a[0]], [title])
fig.suptitle('Figure 4 — Probe angle and response dynamics\n(per-subject Spearman ρ as in V10 compute_sat_stats; the replication\'s probe offsets span ±30°)', y=0.935)
P.save(fig, OUT, 'C_Fig4_ProbeAngle')

# ---------------------------------------------------------------------------
# Figure 5: |probe| < 45 deg
# ---------------------------------------------------------------------------
print('Figure 5...')
fig, cols = grid_2col(4)
for (R, title), a in zip(STUDY_PAIR, cols):
    g = R['groups']
    P.paired_boxes(a[0], R['piv'][('hemi45', 'AbsError')], 'inPF', 'outPF', g, ylim=(0.2, 0.8), ylabel='Normalized\nAbsolute Error')
    P.paired_boxes(a[1], R['piv'][('hemi45', 'Precision')], 'inPF', 'outPF', g, ylim=(0.2, 0.8), ylabel='Normalized\nPrecision (κ)')
    P.paired_boxes(a[2], R['piv'][('rad45', 'AbsError')], 'Radial', 'Tangential', g, ylim=(0.2, 0.8), ylabel='Normalized\nAbsolute Error')
    P.paired_boxes(a[3], R['piv'][('rad45', 'Precision')], 'Radial', 'Tangential', g, ylim=(0.2, 0.8), ylabel='Normalized\nPrecision (κ)')
    for ax, l in zip(a, 'ABCD'): letter(ax, l)
    P.column_titles([a[0]], [title + (' (all trials < 30°)' if R['study'] == 'rep' else '')])
fig.suptitle('Figure 5 — Trials with |Δ probe| < 45°', y=0.93)
P.save(fig, OUT, 'C_Fig5_SmallProbeAngles')

# ---------------------------------------------------------------------------
# Supplementary Figure 1
# ---------------------------------------------------------------------------
print('Supplementary Figure 1...')
fig, cols = grid_2col(4)
rt_l = lims_of(R1['vals_hemi']['RT'], R2['vals_hemi']['RT'])
for (R, title), a in zip(STUDY_PAIR, cols):
    g = R['groups']
    P.paired_boxes(a[0], R['piv'][('hemi', 'RT')], 'inPF', 'outPF', g, ylim=(0.44, 0.56), ylabel='Normalized\nReaction Time')
    P.scatter_pair(a[1], R['vals_hemi'], 'Hemifield', 'inPF', 'outPF', 'RT', g, 'RT (s), outPF', 'RT (s), inPF', lims=rt_l)
    for ax, met, lab, yl in [(a[2], 'AbsError', 'Abs Error', (0.33, 0.67)), (a[3], 'Precision', 'Precision (κ)', (0.18, 0.82))]:
        P.overlay_boxes(ax, {h: R['piv'][('rad_hemi', met, h)] for h in ['inPF', 'outPF']}, 'Radial', 'Tangential',
                        g, ylim=yl, ylabel=f'Normalized {lab}')
    for ax, l in zip(a, 'ABCD'): letter(ax, l)
    P.column_titles([a[0]], [title])
fig.suptitle('Supplementary Figure 1 — Reaction time by hemifield; radial sensitivity by hemifield', y=0.93)
P.save(fig, OUT, 'C_SuppFig1')

# ---------------------------------------------------------------------------
# Diagnostics: radial index across subsets and days
# ---------------------------------------------------------------------------
print('Diagnostics...')
fig = plt.figure(figsize=(17, 13))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.6)
for c, (met, lab) in enumerate([('Abs Error', 'abs error (negative = radial better)'), ('Precision (K)', 'κ (positive = radial better)')]):
    ax = fig.add_subplot(gs[0, c])
    rows = []
    for R, name in [(R1, 'Study 1'), (R2, 'Replication')]:
        for sub in (['all', '<45', '<30'] if R['study'] == 'orig' else ['all']):
            st = R['stats']
            for t in R['groups']:
                r = st[(st['Metric'] == met) & (st['Contrast'] == 'Radial-Tangential') & (st['Subset'] == sub) & (st['TMS'] == t)]
                if len(r): rows.append((f'{name} · {sub}', t, r['CI_mean'].iloc[0], r['CI_sem'].iloc[0], r['N'].iloc[0], r['p_perm'].iloc[0]))
    for i, (name, t, m, se, n, p) in enumerate(rows):
        y = len(rows) - i
        ax.errorbar(m, y, xerr=stats.t.ppf(.975, n - 1) * se, fmt='o', color=P.TMS_LINE_PAL[t], ms=6, capsize=2)
        ax.text(1.02, y, f'{name} · {P.TMS_SHORT[t]}  ({m:+.3f}, {P.fmt_p(p)})', transform=ax.get_yaxis_transform(), va='center', fontsize=8)
    ax.axvline(0, color='k', ls='--', lw=1); ax.set_yticks([])
    ax.set_xlabel(f'Radial index (R−T)/(R+T), {lab}; 95% CI')
    letter(ax, 'AB'[c]); sns.despine(ax=ax, left=True)
for c, (R, title) in enumerate(STUDY_PAIR):
    ax = fig.add_subplot(gs[1, c])
    pd_ = R['day_index'].pivot_table(index='subjID', columns='Day', values='CI')
    for _, row in pd_.iterrows():
        ax.plot([0, 1], [row['Day 1'], row['Day 2']], color='0.6', alpha=0.6, marker='o', ms=3)
    ax.errorbar([0, 1], pd_.mean(), yerr=pd_.sem(), color='m', lw=3, capsize=4, marker='s', ms=8)
    d = (pd_['Day 1'] - pd_['Day 2']).dropna().values
    ax.set_title(f"{title}\nDay 1 − Day 2 = {d.mean():+.3f} ± {stats.sem(d):.3f}, {P.fmt_p(run_permutation_test(d))}", fontsize=10)
    ax.axhline(0, color='k', ls='--', lw=1); ax.set_xticks([0, 1]); ax.set_xticklabels(['Day 1', 'Day 2']); ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel('Radial index, abs error (pooled over TMS)')
    letter(ax, 'CD'[c]); sns.despine(ax=ax, offset=5)
fig.suptitle('Radial index across studies and probe-offset subsets (A, B) and across testing days (C, D)', y=0.96)
P.save(fig, OUT, 'C_Diagnostics_Index')

fig = plt.figure(figsize=(14, 11))
gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)
for c, (R, title) in enumerate(STUDY_PAIR):
    g = R['groups']
    ax = fig.add_subplot(gs[0, c])
    oris = sorted(R['ori_vf']['OriCat'].unique())
    for i, t in enumerate(g):
        for ori in oris:
            gg = R['ori_vf'][(R['ori_vf']['TMS'] == t) & (R['ori_vf']['OriCat'] == ori)].groupby('VF')['AbsError'].agg(['mean', 'sem']).reindex(['LVF', 'RVF'])
            acute = ori < 90
            ax.errorbar(np.array([0, 1]) + i * 2.6 + (oris.index(ori) - (len(oris) - 1) / 2) * 0.07, gg['mean'], yerr=gg['sem'],
                        color=P.TMS_LINE_PAL[t], marker='o' if acute else 's', ls='-' if acute else '--', capsize=2, ms=4,
                        label=f'{ori}°' if i == 0 else None)
    ax.set_xticks([x + i * 2.6 for i in range(len(g)) for x in (0, 1)])
    ax.set_xticklabels([f'{v}\n{P.TMS_SHORT[t].split()[0]}' for t in g for v in ('LVF', 'RVF')], fontsize=8)
    ax.set_ylabel('Abs error (°)'); ax.legend(fontsize=7, frameon=False, ncol=2, title='solid = acute (radial in LVF)', title_fontsize=7)
    ax.set_title(title + '\norientation × visual field', fontsize=10); letter(ax, 'AB'[c]); sns.despine(ax=ax, offset=5)

    ax = fig.add_subplot(gs[1, c])
    anc = R['anchor']
    sns.barplot(data=anc, x='TMS', y='slope', hue='DiagType', order=g, ax=ax, errorbar='se', palette=['#4472C4', '#A9C4E8'])
    sns.stripplot(data=anc, x='TMS', y='slope', hue='DiagType', order=g, ax=ax, dodge=True, color='k', alpha=0.35, size=3, legend=False)
    ax.axhline(0, color='k', lw=1); ax.set_ylim(-0.15, 0.45)
    ax.set_ylabel('Pull toward probe start\n(slope; |Δprobe| < 30°)'); ax.set_xlabel('')
    ax.set_title('Pull toward the starting orientation', fontsize=10); letter(ax, 'CD'[c]); sns.despine(ax=ax, offset=5)
fig.suptitle('Orientation × visual field and pull toward the probe start', y=0.96)
P.save(fig, OUT, 'C_Diagnostics_OriVF_Anchoring')

pd.concat([R1['stats'].assign(Study='Study 1'), R2['stats'].assign(Study='Replication')]).to_csv(
    os.path.join(OUT, 'Stats_PaperContrasts.csv'), index=False)
print(f'\nOutputs in {OUT}')
