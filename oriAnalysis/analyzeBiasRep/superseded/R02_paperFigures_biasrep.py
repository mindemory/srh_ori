"""
R02_paperFigures_biasrep.py

Regenerates the manuscript figures (Dake, George, Serfaty & Curtis 2026) for the
bias-replication cohort, using the same processing as the paper code
(analyzeBehav/V10_paperFigures.py, plus V05_oriRotation.py for Fig 4D):

  - LargeError (global 3 SD) trials removed
  - Radial/Tangential: obtuse orientations radial in the right visual field,
    acute orientations radial in the left visual field (V10 get_ori_type)
  - Per subject x TMS x condition: mean AbsError / RT, and mixture-model kappa
    from modeller.fit_model(errOri_scaled, 'mixtureModel', max_trials=100),
    cells with < 10 trials skipped
  - Normalized ratios A/(A+B), B/(A+B); sign-flip permutation test (10,000)
    of A_norm - 0.5, Cohen's d = mean / SD (ddof=0)
  - The manuscript reports the contrast index CI = A_norm - B_norm
    (= (A-B)/(A+B)); its p and d are identical to the A_norm test above.

The only changes from V10 are the ones the design forces: TMS groups are
No TMS and Middle TMS (no Early TMS in this study), and the orientation sets
include 45/135. The < 45 deg probe-offset filter (Fig 5) is applied as in the
paper; in this study every trial already has |offset| < 30 deg.

Also writes a direct No TMS vs Middle TMS test of each contrast index, which
the paper does not report.

Each figure gets an extra row treating geometry as continuous (angle of each
target from the true radial axis at the stimulus location; see
biasrep_geometry.py): a continuous anisotropy model, behavioural RDMs with a
radial-expansion RSA, and outcome-vs-distance-from-radial curves.

Usage: python R02_paperFigures_biasrep.py [data_root]
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

from biasrep_common import (DATA_ROOT, TMS_GROUPS, load_df, run_permutation_test,
                            subject_values, normalized, contrast_stats)
import biasrep_geometry as bg

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
np.random.seed(0)

output_dir = os.path.join(DATA_ROOT, 'Figures', 'paperFigures_biasrep')
os.makedirs(output_dir, exist_ok=True)

TMS_BOX_PAL = {'No TMS': '#9E9E9E', 'Early TMS': '#EF9A9A', 'Middle TMS': '#90CAF9'}
TMS_STRIP_PAL = {'No TMS': 'k', 'Early TMS': '#C62828', 'Middle TMS': '#1565C0'}
TMS_LINE_PAL = {'No TMS': 'k', 'Early TMS': '#C62828', 'Middle TMS': '#29A3E0'}
TMS_SHORT = {'No TMS': 'None', 'Middle TMS': 'Middle Delay'}

df = bg.add_geometry(load_df())

# ---------------------------------------------------------------------------
# Plotting helpers (V10 _draw_paired_boxes / _draw_overlay_boxes / scatters)
# ---------------------------------------------------------------------------
def draw_paired_boxes(ax, piv, A, B, leg_labels=None, ylim=None, ylabel=''):
    bw, gap = 0.28, 0.15
    rng = np.random.default_rng(42)
    xt = []
    for i, t in enumerate(TMS_GROUPS):
        base = i * (2 * bw + gap)
        xt.append(base + bw / 2)
        sub = piv[piv['TMS'] == t].dropna(subset=[f'{A}_norm', f'{B}_norm'])
        for pos, cond in [(base, A), (base + bw, B)]:
            vals = sub[f'{cond}_norm'].values
            if len(vals) < 3: continue
            is_A = cond == A
            col, scol = TMS_BOX_PAL[t], TMS_STRIP_PAL[t]
            al, sal = (1.0, 0.7) if is_A else (0.65, 0.35)
            ax.boxplot([vals], positions=[pos], widths=bw, patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops=dict(facecolor=col, alpha=al, linewidth=0.8, edgecolor=scol),
                       medianprops=dict(color='white', linewidth=2),
                       whiskerprops=dict(color=scol, alpha=al, linewidth=0.8),
                       capprops=dict(color=scol, alpha=al, linewidth=0.8))
            ax.scatter(pos + rng.uniform(-bw * 0.28, bw * 0.28, len(vals)), vals, color=scol, alpha=sal, s=11, zorder=3)
    ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks(xt); ax.set_xticklabels(['None', 'Middle\nDelay'])
    ax.set_xlim(-bw * 0.8, (len(TMS_GROUPS) - 1) * (2 * bw + gap) + bw * 1.8)
    if ylim: ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel); ax.set_xlabel('TMS')
    nc = TMS_BOX_PAL['No TMS']
    la, lb = leg_labels or (A, B)
    ax.legend(handles=[mpatches.Patch(facecolor='#424242', label=la),
                       mpatches.Patch(facecolor=nc, alpha=0.65, label=lb)], frameon=True, fontsize=8, loc='upper right')
    sns.despine(ax=ax, offset=5)


def draw_overlay_boxes(ax, data_dict, A, B, ylim=None, ylabel=''):
    bw, inner_gap, mid_gap, tms_gap = 0.22, 0.04, 0.14, 0.30
    step = 4 * bw + inner_gap + mid_gap + tms_gap
    rng = np.random.default_rng(42)
    xt = []
    for i, t in enumerate(TMS_GROUPS):
        base = i * step
        pos_map = {(A, 'inPF'): base, (B, 'inPF'): base + bw + inner_gap,
                   (A, 'outPF'): base + 2 * bw + inner_gap + mid_gap, (B, 'outPF'): base + 3 * bw + 2 * inner_gap + mid_gap}
        xt.append(base + 1.5 * bw + inner_gap + mid_gap / 2)
        for (cond, hemi), pos in pos_map.items():
            vals = data_dict.get((t, cond, hemi), np.array([]))
            if len(vals) < 3: continue
            is_A = cond == A
            col, scol = TMS_BOX_PAL[t], TMS_STRIP_PAL[t]
            al, sal = (1.0, 0.7) if is_A else (0.65, 0.35)
            ax.boxplot([vals], positions=[pos], widths=bw, patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops=dict(facecolor=col, alpha=al, linewidth=0.8, edgecolor=scol,
                                     hatch=None if hemi == 'inPF' else '///'),
                       medianprops=dict(color='white', linewidth=2),
                       whiskerprops=dict(color=scol, alpha=al, linewidth=0.8),
                       capprops=dict(color=scol, alpha=al, linewidth=0.8))
            ax.scatter(pos + rng.uniform(-bw * 0.28, bw * 0.28, len(vals)), vals, color=scol, alpha=sal, s=10, zorder=3)
    ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks(xt); ax.set_xticklabels(['None', 'Middle\nDelay'])
    if ylim: ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel); ax.set_xlabel('TMS')
    nc = TMS_BOX_PAL['No TMS']
    ax.legend(handles=[mpatches.Patch(facecolor=nc, label=f'inPF · {A}'),
                       mpatches.Patch(facecolor=nc, alpha=0.65, label=f'inPF · {B}'),
                       mpatches.Patch(facecolor=nc, hatch='///', label=f'outPF · {A}'),
                       mpatches.Patch(facecolor=nc, alpha=0.65, hatch='///', label=f'outPF · {B}')],
              frameon=False, fontsize=7, loc='upper right')
    sns.despine(ax=ax, offset=5)


def draw_scatter(ax, vals, p_col, A, B, met, xlabel, ylabel, x_is_A=False):
    piv = vals.pivot_table(index=['subjID', 'TMS'], columns=p_col, values=met, observed=False).reset_index()
    xc, yc = (A, B) if x_is_A else (B, A)
    for t in TMS_GROUPS:
        sub = piv[piv['TMS'] == t].dropna(subset=[A, B])
        c = TMS_LINE_PAL[t]
        ax.scatter(sub[xc], sub[yc], color=c, alpha=0.45, s=30, edgecolor='white')
        ax.errorbar(sub[xc].mean(), sub[yc].mean(), xerr=sub[xc].sem(), yerr=sub[yc].sem(), fmt='o', color=c,
                    markersize=9, markeredgecolor='white', label=TMS_SHORT[t])
    v = piv[[A, B]].values.flatten(); lo, hi = np.nanmin(v), np.nanmax(v)
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', alpha=0.6, lw=1)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect('equal', 'box'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=True, loc='upper left', title='TMS', title_fontsize=8)
    sns.despine(ax=ax, offset=5)


# ---------------------------------------------------------------------------
# Plotting helpers for the continuous-geometry panels
# ---------------------------------------------------------------------------
def fmt_p(p):
    return 'p<.001' if p < 0.001 else f'p={p:.3f}'


def draw_hemi_raw_boxes(ax, tab, value, ylabel, ref=0.0):
    """Boxes of a per-subject parameter for inPF (solid) / outPF (lighter) x TMS,
    with the inPF - outPF paired p under each pair."""
    bw, gap = 0.28, 0.15
    rng = np.random.default_rng(42)
    xt = []
    for i, t in enumerate(TMS_GROUPS):
        base = i * (2 * bw + gap)
        xt.append(base + bw / 2)
        w = tab[tab['TMS'] == t].pivot(index='subjID', columns='Hemifield', values=value)
        for pos, hemi in [(base, 'inPF'), (base + bw, 'outPF')]:
            vals = w[hemi].dropna().values
            al, sal = (1.0, 0.7) if hemi == 'inPF' else (0.65, 0.35)
            col, scol = TMS_BOX_PAL[t], TMS_STRIP_PAL[t]
            ax.boxplot([vals], positions=[pos], widths=bw, patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops=dict(facecolor=col, alpha=al, linewidth=0.8, edgecolor=scol),
                       medianprops=dict(color='white', linewidth=2),
                       whiskerprops=dict(color=scol, alpha=al, linewidth=0.8),
                       capprops=dict(color=scol, alpha=al, linewidth=0.8))
            ax.scatter(pos + rng.uniform(-bw * 0.28, bw * 0.28, len(vals)), vals, color=scol, alpha=sal, s=11, zorder=3)
        d = (w['inPF'] - w['outPF']).dropna().values
        ax.text(base + bw / 2, 0.02, f'in−out {fmt_p(run_permutation_test(d))}', transform=ax.get_xaxis_transform(),
                ha='center', fontsize=7)
    ax.axhline(ref, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks(xt); ax.set_xticklabels(['None', 'Middle\nDelay'])
    ax.set_xlim(-bw * 0.8, (len(TMS_GROUPS) - 1) * (2 * bw + gap) + bw * 1.8)
    ax.set_ylabel(ylabel); ax.set_xlabel('TMS')
    nc = TMS_BOX_PAL['No TMS']
    ax.legend(handles=[mpatches.Patch(facecolor='#424242', label='inPF'),
                       mpatches.Patch(facecolor=nc, alpha=0.65, label='outPF')], frameon=True, fontsize=8, loc='upper right')
    sns.despine(ax=ax, offset=5)


def draw_tms_paired(ax, tab, value, ylabel, title, ref=0.0):
    """Per-subject value under No TMS and Middle TMS joined by lines, with the
    test of each condition against `ref` (in the title) and the paired difference."""
    w = tab.pivot(index='subjID', columns='TMS', values=value)[TMS_GROUPS].dropna()
    for _, row in w.iterrows():
        ax.plot([0, 1], row.values, color='0.7', lw=1, zorder=1)
    rng = np.random.default_rng(42)
    p_txt = []
    for i, t in enumerate(TMS_GROUPS):
        v = w[t].values
        ax.boxplot([v], positions=[i], widths=0.35, patch_artist=True, showfliers=False, manage_ticks=False,
                   boxprops=dict(facecolor=TMS_BOX_PAL[t], alpha=0.8, linewidth=0.8, edgecolor=TMS_STRIP_PAL[t]),
                   medianprops=dict(color='white', linewidth=2),
                   whiskerprops=dict(color=TMS_STRIP_PAL[t], linewidth=0.8), capprops=dict(color=TMS_STRIP_PAL[t], linewidth=0.8))
        ax.scatter(i + rng.uniform(-0.06, 0.06, len(v)), v, color=TMS_STRIP_PAL[t], s=14, zorder=3)
        p_txt.append(f'{TMS_SHORT[t]} vs {ref:g}: {fmt_p(run_permutation_test(v - ref))}')
    ax.set_title(f'{title}\n' + '; '.join(p_txt), fontsize=9)
    d = (w['No TMS'] - w['Middle TMS']).values
    ax.text(0.5, 0.02, f'None − Middle = {d.mean():+.3f} ± {stats.sem(d):.3f}, {fmt_p(run_permutation_test(d))}',
            transform=ax.transAxes, ha='center', fontsize=8)
    ax.axhline(ref, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['None', 'Middle\nDelay']); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(ylabel); ax.set_xlabel('TMS')
    sns.despine(ax=ax, offset=5)


def draw_rdm(ax, mat, title, cmap='viridis', vmin=0.4, vmax=16, cbar_label="d' (log scale)", log=True):
    mat = mat.copy()
    np.fill_diagonal(mat, np.nan)
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax) if log else matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(mat, cmap=cmap, norm=norm)
    ax.axhline(5.5, color='w', lw=1.5); ax.axvline(5.5, color='w', lw=1.5)
    ax.set_xticks(range(12)); ax.set_yticks(range(12))
    ax.set_xticklabels(COND_TICKS, rotation=90, fontsize=7); ax.set_yticklabels(COND_TICKS, fontsize=7)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8); cb.ax.tick_params(labelsize=7)


def draw_dist_curve(ax, data, value, ylabel, split_hemi=False):
    """Group mean ± SEM of a per-subject value in 15° bins of distance from the radial axis."""
    d = data.copy()
    d['DistBin'] = pd.cut(d['DistRadial'], bins=np.arange(0, 91, 15))
    keys = ['subjID', 'TMS', 'DistBin'] + (['Hemifield'] if split_hemi else [])
    b = d.groupby(keys, observed=True)[value].mean().reset_index()
    b['mid'] = b['DistBin'].apply(lambda x: x.mid).astype(float)
    groups = [(t, h) for t in TMS_GROUPS for h in (['inPF', 'outPF'] if split_hemi else [None])]
    for t, h in groups:
        g = b[(b['TMS'] == t) & ((b['Hemifield'] == h) if h else True)].groupby('mid')[value].agg(['mean', 'sem'])
        off = (1.2 if t == 'Middle TMS' else -1.2) + ((0.5 if h == 'outPF' else -0.5) if h else 0)
        ax.errorbar(g.index + off, g['mean'], yerr=g['sem'], color=TMS_LINE_PAL[t], marker='o', ms=5, capsize=2,
                    ls='--' if h == 'outPF' else '-', label=f'{TMS_SHORT[t]}' + (f' · {h}' if h else ''))
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90]); ax.set_xlim(-3, 93)
    ax.set_xlabel('Distance of target from radial axis (°)\n0 = radial, 90 = tangential')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax, offset=5)


def panel_letters(axes, letters):
    for a, l in zip(axes, letters):
        a.set_title(l, loc='left', fontweight='bold', fontsize=14)


def save(fig, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}')


# ---------------------------------------------------------------------------
# Compute everything once (paper analyses)
# ---------------------------------------------------------------------------
stats_rows = []

print('Fitting hemifield cells...')
vals_hemi = subject_values(df, 'Hemifield', ['inPF', 'outPF'])
print('Fitting radial/tangential cells...')
vals_rad = subject_values(df, 'DiagType', ['Radial', 'Tangential'])
print('Fitting radial/tangential x hemifield cells...')
vals_rad_hemi = subject_values(df, 'DiagType', ['Radial', 'Tangential'], extra_cols=('Hemifield',))

piv = {}
for met, lab in [('AbsError', 'Abs Error'), ('RT', 'Reaction Time'), ('Precision', 'Precision (K)')]:
    piv[('hemi', met)] = normalized(vals_hemi, 'Hemifield', 'inPF', 'outPF', met)
    stats_rows += contrast_stats(piv[('hemi', met)], 'inPF', 'outPF', lab, {'Figure': 'Fig2 / SuppFig1A', 'Subset': 'all'})
for met, lab in [('AbsError', 'Abs Error'), ('Precision', 'Precision (K)')]:
    piv[('rad', met)] = normalized(vals_rad, 'DiagType', 'Radial', 'Tangential', met)
    stats_rows += contrast_stats(piv[('rad', met)], 'Radial', 'Tangential', lab, {'Figure': 'Fig3', 'Subset': 'all'})
    for hemi in ['inPF', 'outPF']:
        p = normalized(vals_rad_hemi[vals_rad_hemi['Hemifield'] == hemi], 'DiagType', 'Radial', 'Tangential', met)
        piv[('rad_hemi', met, hemi)] = p
        stats_rows += contrast_stats(p, 'Radial', 'Tangential', lab, {'Figure': 'SuppFig1C/D', 'Subset': hemi})

# Fig 5: |probe offset| < 45 deg (V10 generate_figure_4)
sub_45 = df[df['AbsInitOffset'] < 45].copy()
same_45 = len(sub_45) == len(df)
print(f'Fig 5 subset (<45 deg): {len(sub_45)} of {len(df)} trials')
if same_45:
    vals_hemi45, vals_rad45 = vals_hemi, vals_rad
else:
    vals_hemi45 = subject_values(sub_45, 'Hemifield', ['inPF', 'outPF'])
    vals_rad45 = subject_values(sub_45, 'DiagType', ['Radial', 'Tangential'])
for met, lab in [('AbsError', 'Accuracy'), ('Precision', 'Precision')]:
    piv[('hemi45', met)] = normalized(vals_hemi45, 'Hemifield', 'inPF', 'outPF', met)
    piv[('rad45', met)] = normalized(vals_rad45, 'DiagType', 'Radial', 'Tangential', met)
    stats_rows += contrast_stats(piv[('hemi45', met)], 'inPF', 'outPF', lab, {'Figure': 'Fig5A/B', 'Subset': '<45'})
    stats_rows += contrast_stats(piv[('rad45', met)], 'Radial', 'Tangential', lab, {'Figure': 'Fig5C/D', 'Subset': '<45'})

# ---------------------------------------------------------------------------
# Continuous geometry
# ---------------------------------------------------------------------------
print('Fitting continuous anisotropy models...')
an_tms = bg.anisotropy_table(df, ['subjID', 'TMS'])
an_hemi = bg.anisotropy_table(df, ['subjID', 'TMS', 'Hemifield'])


def rdm_tables(data, keys):
    rdms, rows = {}, []
    for k, sub in data.groupby(keys, observed=True):
        rdm, phi = bg.subject_rdm(sub)
        rdms[k] = rdm
        rows.append({**dict(zip(keys, k)), 'rsa': bg.rsa_radial(rdm, phi), 'block': bg.block_index(rdm)})
    return rdms, pd.DataFrame(rows)


print('Building behavioural RDMs...')
rdm_tms, rsa_tms = rdm_tables(df, ['subjID', 'TMS'])
rdm_hemi, rsa_hemi = rdm_tables(df, ['subjID', 'TMS', 'Hemifield'])
if same_45:
    an_45, rdm_45, rsa_45 = an_tms, rdm_tms, rsa_tms
else:
    an_45 = bg.anisotropy_table(sub_45, ['subjID', 'TMS'])
    rdm_45, rsa_45 = rdm_tables(sub_45, ['subjID', 'TMS'])

cond_dist = df.groupby(['DiagType', 'JitBin'])['DistRadial'].mean()
COND_TICKS = [f"{'T' if t == 'Tangential' else 'R'} {cond_dist[(t, j)]:.0f}°" for t, j in bg.COND_ORDER]


def mean_rdm(rdms, tms):
    return np.nanmean(np.stack([m for (s, t), m in rdms.items() if t == tms]), axis=0)


def delta_rdm(rdms):
    """Mean over subjects of log(d' No TMS / d' Middle TMS); > 0 = conditions
    more discriminable without TMS. Scale-free, so the large between-category
    distances do not swamp the within-category structure."""
    subs = sorted({s for s, _ in rdms})
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nanmean(np.stack([np.log(rdms[(s, 'No TMS')] / rdms[(s, 'Middle TMS')]) for s in subs
                                    if (s, 'No TMS') in rdms and (s, 'Middle TMS') in rdms]), axis=0)


geo_rows = []


def geo_stats(tab, value, label, subset, by_hemi=False):
    for t in TMS_GROUPS:
        for h in (['inPF', 'outPF'] if by_hemi else [None]):
            v = tab[(tab['TMS'] == t) & ((tab['Hemifield'] == h) if h else True)][value].dropna().values
            geo_rows.append({'Measure': label, 'Subset': subset, 'TMS': t, 'Hemifield': h or 'both', 'N': len(v),
                             'Mean': v.mean(), 'SEM': stats.sem(v), 'p_perm_vs0': run_permutation_test(v),
                             'Cohen_d': v.mean() / np.std(v)})
    if by_hemi:
        for t in TMS_GROUPS:
            w = tab[tab['TMS'] == t].pivot(index='subjID', columns='Hemifield', values=value).dropna()
            d = (w['inPF'] - w['outPF']).values
            geo_rows.append({'Measure': label, 'Subset': subset, 'TMS': t, 'Hemifield': 'inPF minus outPF', 'N': len(d),
                             'Mean': d.mean(), 'SEM': stats.sem(d), 'p_perm_vs0': run_permutation_test(d),
                             'Cohen_d': d.mean() / np.std(d)})
    else:
        w = tab.pivot(index='subjID', columns='TMS', values=value).dropna()
        d = (w['No TMS'] - w['Middle TMS']).values
        geo_rows.append({'Measure': label, 'Subset': subset, 'TMS': 'No TMS minus Middle TMS', 'Hemifield': 'both',
                         'N': len(d), 'Mean': d.mean(), 'SEM': stats.sem(d), 'p_perm_vs0': run_permutation_test(d),
                         'Cohen_d': d.mean() / np.std(d)})


for tab, by_h, subset in [(an_tms, False, 'all'), (an_hemi, True, 'all'), (an_45, False, '<45')]:
    geo_stats(tab, 'a', 'Anisotropy a (precision, radial>tangential if >0)', subset, by_h)
    geo_stats(tab, 'm', 'Radial bias m (deg; >0 away from radial axis)', subset, by_h)
    geo_stats(tab, 'c', 'Cardinal precision c', subset, by_h)
    geo_stats(tab, 'mc', 'Cardinal bias mc (deg; >0 away from cardinals)', subset, by_h)
for tab, by_h, subset in [(rsa_tms, False, 'all'), (rsa_hemi, True, 'all'), (rsa_45, False, '<45')]:
    geo_stats(tab, 'rsa', 'RSA partial rho (radial-expanded | isotropic)', subset, by_h)
    geo_stats(tab, 'block', 'RDM block index log(radial/tangential d\')', subset, by_h)

# ---------------------------------------------------------------------------
# Fig 2: phosphene-field effects (+ E, F continuous geometry by hemifield)
# ---------------------------------------------------------------------------
print('Figure 2...')
fig = plt.figure(figsize=(11, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
ax = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
draw_paired_boxes(ax[0], piv[('hemi', 'AbsError')], 'inPF', 'outPF', ylim=(0.4, 0.6), ylabel='Normalized\nAbsolute Error (°)')
draw_scatter(ax[1], vals_hemi, 'Hemifield', 'inPF', 'outPF', 'AbsError', 'Abs. Error (°), outPF', 'Abs. Error (°), inPF')
draw_paired_boxes(ax[2], piv[('hemi', 'Precision')], 'inPF', 'outPF', ylim=(0.3, 0.7), ylabel='Normalized\nPrecision (κ)')
draw_scatter(ax[3], vals_hemi, 'Hemifield', 'inPF', 'outPF', 'Precision', 'Precision (κ), outPF', 'Precision (κ), inPF')
draw_hemi_raw_boxes(ax[4], an_hemi, 'a', 'Continuous radial anisotropy a\n(κ ∝ exp(a·cos2φ); >0 = radial more precise)')
ax[4].set_title('Precision vs angle from radial axis', fontsize=10)
draw_hemi_raw_boxes(ax[5], an_hemi, 'm', 'Bias relative to radial axis m (°)\n(<0 = pulled toward radial axis)')
ax[5].set_title('Report bias vs angle from radial axis', fontsize=10)
panel_letters(ax, 'ABCDEF')
fig.suptitle('Bias replication — Phosphene field effects (N = %d)' % df['subjID'].nunique(), y=0.92)
save(fig, 'Fig2_PhospheneField')

# ---------------------------------------------------------------------------
# Fig 3: radial sensitivity (+ E-J continuous geometry and RDMs)
# ---------------------------------------------------------------------------
print('Figure 3...')
fig = plt.figure(figsize=(16, 21))
gs = GridSpec(4, 6, figure=fig, hspace=0.55, wspace=1.6)
top = [fig.add_subplot(gs[r, c:c + 3]) for r in range(2) for c in (0, 3)]
draw_paired_boxes(top[0], piv[('rad', 'AbsError')], 'Radial', 'Tangential', ylim=(0.4, 0.6), ylabel='Normalized\nAbsolute Error (°)')
draw_scatter(top[1], vals_rad, 'DiagType', 'Radial', 'Tangential', 'AbsError', 'Abs. Error (°), Radial', 'Abs. Error (°), Tangential', x_is_A=True)
draw_paired_boxes(top[2], piv[('rad', 'Precision')], 'Radial', 'Tangential', ylim=(0.3, 0.7), ylabel='Normalized\nPrecision (κ)')
draw_scatter(top[3], vals_rad, 'DiagType', 'Radial', 'Tangential', 'Precision', 'Precision (κ), Radial', 'Precision (κ), Tangential', x_is_A=True)
bot = [fig.add_subplot(gs[r, c:c + 2]) for r in (2, 3) for c in (0, 2, 4)]
draw_dist_curve(bot[0], df, 'AbsError', 'Absolute error (°)')
bot[0].set_title('Error vs angle from radial axis', fontsize=10)
draw_rdm(bot[1], mean_rdm(rdm_tms, 'No TMS'), "Behavioural RDM (d') — None")
for _a in bot[1:4]: _a.set_xlabel('T = tangential, R = radial bins;\nlabel = mean distance from radial axis', fontsize=8)
draw_rdm(bot[2], mean_rdm(rdm_tms, 'Middle TMS'), "Behavioural RDM (d') — Middle Delay")
dr = delta_rdm(rdm_tms)
lim = np.nanmax(np.abs(dr))
draw_rdm(bot[3], dr, "RDM contrast: log(d' None / d' Middle)\n>0 = more discriminable without TMS", cmap='RdBu_r', vmin=-lim, vmax=lim, cbar_label='log ratio', log=False)
draw_tms_paired(bot[4], an_tms, 'a', 'Continuous radial anisotropy a\n(>0 = radial more precise)', 'Continuous anisotropy (all trials)')
draw_tms_paired(bot[5], rsa_tms, 'rsa', 'RSA: partial ρ with radial-expanded\nmodel RDM | isotropic distance', 'RDM fit to radial expansion')
panel_letters(top + bot, 'ABCDEFGHIJ')
fig.suptitle('Bias replication — Radial sensitivity (N = %d)' % df['subjID'].nunique(), y=0.91)
save(fig, 'Fig3_RadialSensitivity')

# ---------------------------------------------------------------------------
# Fig 4: probe angle and response dynamics (V10 plot_sat_grid + V05 P(CW))
#        (+ E, F response dynamics vs angle from radial axis)
# ---------------------------------------------------------------------------
print('Figure 4...')
n_bins, window_size = 25, 3
rt_parts, off_parts, sat_rows = [], [], []
for (subj, tms), sdf in df.groupby(['subjID', 'TMS']):
    if len(sdf) < n_bins: continue
    sdf = sdf.copy()
    sdf['RT_Decile'] = pd.qcut(sdf['RT'], q=n_bins, labels=False, duplicates='drop')
    a = sdf.groupby('RT_Decile').agg({'RT': 'mean', 'AbsError': 'mean'}).reset_index()
    a.update(a[['RT', 'AbsError']].rolling(window=window_size, center=True).mean())
    sdf['Offset_Decile'] = pd.qcut(sdf['AbsInitOffset'], q=n_bins, labels=False, duplicates='drop')
    b = sdf.groupby('Offset_Decile').agg({'AbsInitOffset': 'mean', 'RT': 'mean', 'AbsError': 'mean'}).reset_index()
    b.update(b[['AbsInitOffset', 'RT', 'AbsError']].rolling(window=3, center=True).mean())
    a['subjID'], a['TMS'], b['subjID'], b['TMS'] = subj, tms, subj, tms
    rt_parts.append(a); off_parts.append(b)
    sat_rows.append({'subjID': subj, 'TMS': tms,
                     'spearman_Offset_RT': stats.spearmanr(sdf['AbsInitOffset'], sdf['RT'])[0],
                     'spearman_RT_Error': stats.spearmanr(sdf['RT'], sdf['AbsError'])[0],
                     'spearman_Offset_Error': stats.spearmanr(sdf['AbsInitOffset'], sdf['AbsError'])[0],
                     'pearson_Offset_RT': stats.pearsonr(sdf['AbsInitOffset'], sdf['RT'])[0],
                     'pearson_RT_Error': stats.pearsonr(sdf['RT'], sdf['AbsError'])[0],
                     'pearson_Offset_Error': stats.pearsonr(sdf['AbsInitOffset'], sdf['AbsError'])[0]})
df_rt = pd.concat(rt_parts); df_off = pd.concat(off_parts)
df_rt['Group_RT'] = df_rt.groupby(['RT_Decile', 'TMS'])['RT'].transform('mean')
df_off['Group_Offset'] = df_off.groupby(['Offset_Decile', 'TMS'])['AbsInitOffset'].transform('mean')
sat = pd.DataFrame(sat_rows)
sat_stats = []
for t in TMS_GROUPS:
    for c in [c for c in sat.columns if c.startswith(('spearman', 'pearson'))]:
        v = sat.loc[sat['TMS'] == t, c].dropna().values
        sat_stats.append({'TMS': t, 'Metric': c, 'Mean_r': np.mean(v), 'SEM_r': stats.sem(v),
                          'p_perm': run_permutation_test(v), 'd': np.mean(v) / np.std(v), 'N': len(v)})
pd.DataFrame(sat_stats).to_csv(os.path.join(output_dir, 'Fig4_SAT_Stats.csv'), index=False)


def cw_tendency(ori_array):  # V05 get_cw_tendency_continuous
    if len(ori_array) < 2: return 0.5
    diffs = np.diff(ori_array)
    diffs[diffs > 90] -= 180
    diffs[diffs < -90] += 180
    cw, ccw = np.sum(np.abs(diffs[diffs < 0])), np.sum(diffs[diffs > 0])
    return 0.5 if cw + ccw == 0 else cw / (cw + ccw)


pcw = df[(df['RT'] < 3.0) & (df['oriSteps'] > 0)].copy()  # V05 filters
pcw['CW_Tendency'] = pcw['oriArray'].apply(cw_tendency)
pcw['Offset_Bin'] = pd.cut(pcw['initOffset'], bins=np.arange(-90, 92, 2))
ss = pcw.groupby(['subjID', 'TMS', 'Offset_Bin'], observed=True).agg(
    Mean_Tendency=('CW_Tendency', 'mean'), Bin_Mid=('initOffset', 'mean')).reset_index()
sm = []
for _, sdf in ss.groupby(['subjID', 'TMS']):
    sdf = sdf.sort_values('Bin_Mid').reset_index(drop=True)
    # V05 smooths circularly over the full -90..90 range; here the offsets only
    # span -30..30, so wrapping the ends would mix -30 with +30. Plain 5-bin
    # centred rolling mean instead.
    sdf['Mean_Tendency'] = sdf['Mean_Tendency'].rolling(window=5, center=True, min_periods=3).mean()
    sm.append(sdf)
pop = pd.concat(sm).groupby(['TMS', 'Offset_Bin'], observed=True).agg(
    Pop_Mean=('Mean_Tendency', 'mean'), Pop_SEM=('Mean_Tendency', lambda x: np.std(x) / np.sqrt(len(x))),
    X=('Bin_Mid', 'mean')).reset_index()

# Pull toward the starting orientation, by distance from the radial axis
anc = []
df['DistBin3'] = pd.cut(df['DistRadial'], bins=[0, 30, 60, 90], include_lowest=True)
for (s, t, b), sub in df.groupby(['subjID', 'TMS', 'DistBin3'], observed=True):
    if len(sub) < 15: continue
    anc.append({'subjID': s, 'TMS': t, 'mid': b.mid, 'slope': np.polyfit(sub['initOffset'], sub['errOri_scaled'], 1)[0]})
anc = pd.DataFrame(anc)

fig = plt.figure(figsize=(12, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
ax = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
for t in TMS_GROUPS:
    c = TMS_LINE_PAL[t]; lab = TMS_SHORT[t]
    sns.lineplot(data=df_off[df_off['TMS'] == t], x='Group_Offset', y='RT', color=c, errorbar=('ci', 95), ax=ax[0], lw=2.5, label=lab)
    sns.lineplot(data=df_rt[df_rt['TMS'] == t], x='Group_RT', y='AbsError', color=c, errorbar=('ci', 95), ax=ax[1], lw=2.5, label=lab)
    sns.lineplot(data=df_off[df_off['TMS'] == t], x='Group_Offset', y='AbsError', color=c, errorbar=('ci', 95), ax=ax[2], lw=2.5, label=lab)
    p = pop[pop['TMS'] == t].sort_values('X')
    ax[3].plot(p['X'], p['Pop_Mean'], color=c, lw=2, label=lab)
    ax[3].fill_between(p['X'], p['Pop_Mean'] - p['Pop_SEM'], p['Pop_Mean'] + p['Pop_SEM'], color=c, alpha=0.2)
    g = anc[anc['TMS'] == t].groupby('mid')['slope'].agg(['mean', 'sem'])
    ax[5].errorbar(g.index + (2 if t == 'Middle TMS' else -2), g['mean'], yerr=g['sem'], color=c, marker='o', capsize=3, label=lab)
ax[0].set(xlabel='Absolute Δ Probe Angle (°)', ylabel='Response Time (s)', xlim=(0, 90))
ax[1].set(xlabel='Response Time (s)', ylabel='Angular Error (°)')
ax[2].set(xlabel='Absolute Δ Probe Angle (°)', ylabel='Angular Error (°)', xlim=(0, 90))
ax[3].axvspan(-45, 45, color='0.85', alpha=0.6, zorder=0)
ax[3].axhline(0.5, color='k', ls='--', lw=1); ax[3].axvline(0, color='k', ls='--', lw=1)
ax[3].set(xlabel='Signed Δ Probe Angle (°)', ylabel='Proportion of CW knob rotations', xlim=(-90, 90), ylim=(0, 1))
draw_dist_curve(ax[4], df, 'RT', 'Response Time (s)')
ax[4].set_title('Response time vs angle from radial axis', fontsize=10)
ax[5].axhline(0, color='k', lw=1)
ax[5].set(xlabel='Distance of target from radial axis (°, 30° bins)', ylabel='Pull toward start\n(signed error per ° of offset)',
          xticks=[15, 45, 75])
ax[5].set_title('Pull toward probe start vs angle from radial axis', fontsize=10)
for a in ax:
    sns.despine(ax=a, offset=5)
    if a.get_legend(): a.legend(fontsize=8, frameon=False)
panel_letters(ax, 'ABCDEF')
fig.suptitle('Bias replication — Probe angle and response dynamics\n(axes in A–D match the manuscript; probe offsets here span ±30°)', y=0.94)
save(fig, 'Fig4_ProbeAngle')

# ---------------------------------------------------------------------------
# Fig 5: small probe angles (< 45 deg) (+ E, F continuous geometry)
# ---------------------------------------------------------------------------
print('Figure 5...')
fig = plt.figure(figsize=(11, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
ax = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
draw_paired_boxes(ax[0], piv[('hemi45', 'AbsError')], 'inPF', 'outPF', ylim=(0.2, 0.8), ylabel='Normalized\nAbsolute Error (°)')
draw_paired_boxes(ax[1], piv[('hemi45', 'Precision')], 'inPF', 'outPF', ylim=(0.2, 0.8), ylabel='Normalized\nPrecision (κ)')
draw_paired_boxes(ax[2], piv[('rad45', 'AbsError')], 'Radial', 'Tangential', ylim=(0.2, 0.8), ylabel='Normalized\nAbsolute Error (°)')
draw_paired_boxes(ax[3], piv[('rad45', 'Precision')], 'Radial', 'Tangential', ylim=(0.2, 0.8), ylabel='Normalized\nPrecision (κ)')
for a, ttl in zip(ax[:4], ['Memory Error', 'Memory Precision', 'Memory Error', 'Memory Precision']):
    a.set_title(ttl, fontsize=10)
dr45 = delta_rdm(rdm_45)
lim = np.nanmax(np.abs(dr45))
draw_rdm(ax[4], dr45, "RDM contrast: log(d' None / d' Middle)\n(|Δprobe| < 45°)", cmap='RdBu_r', vmin=-lim, vmax=lim, cbar_label='log ratio', log=False)
draw_tms_paired(ax[5], an_45, 'a', 'Continuous radial anisotropy a\n(>0 = radial more precise)', 'Continuous anisotropy (|Δprobe| < 45°)')
panel_letters(ax, 'ABCDEF')
fig.suptitle('Bias replication — |Δ probe| < 45°' + (' (all trials in this study: E, F identical to Fig 3H, I)' if same_45 else ''), y=0.92)
save(fig, 'Fig5_SmallProbeAngles')

# ---------------------------------------------------------------------------
# Supplementary Fig 1: RT by hemifield; radial sensitivity by hemifield
#                      (+ E, F continuous geometry by hemifield)
# ---------------------------------------------------------------------------
print('Supplementary Figure 1...')
fig = plt.figure(figsize=(12, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
ax = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
draw_paired_boxes(ax[0], piv[('hemi', 'RT')], 'inPF', 'outPF', ylim=(0.45, 0.55), ylabel='Normalized\nReaction Time')
draw_scatter(ax[1], vals_hemi, 'Hemifield', 'inPF', 'outPF', 'RT', 'RT (s), outPF', 'RT (s), inPF')
for col, met, lab, yl in [(2, 'AbsError', 'Abs Error', (0.35, 0.65)), (3, 'Precision', 'Precision (κ)', (0.2, 0.8))]:
    dd = {}
    for hemi in ['inPF', 'outPF']:
        p = piv[('rad_hemi', met, hemi)]
        for t in TMS_GROUPS:
            s = p[p['TMS'] == t].dropna(subset=['Radial_norm', 'Tangential_norm'])
            dd[(t, 'Radial', hemi)] = s['Radial_norm'].values
            dd[(t, 'Tangential', hemi)] = s['Tangential_norm'].values
    draw_overlay_boxes(ax[col], dd, 'Radial', 'Tangential', ylim=yl, ylabel=f'Normalized {lab}')
draw_dist_curve(ax[4], df, 'AbsError', 'Absolute error (°)', split_hemi=True)
ax[4].set_title('Error vs angle from radial axis, by hemifield', fontsize=10)
draw_hemi_raw_boxes(ax[5], rsa_hemi, 'rsa', 'RSA: partial ρ with radial-expanded\nmodel RDM | isotropic distance')
ax[5].set_title('RDM fit to radial expansion, by hemifield', fontsize=10)
panel_letters(ax, 'ABCDEF')
fig.suptitle('Bias replication — Supplementary Figure 1', y=0.92)
save(fig, 'SuppFig1')

# ---------------------------------------------------------------------------
# Stats and per-subject values
# ---------------------------------------------------------------------------
sdf = pd.DataFrame(stats_rows)
sdf.to_csv(os.path.join(output_dir, 'Stats_AllContrasts.csv'), index=False)
gdf = pd.DataFrame(geo_rows)
gdf.to_csv(os.path.join(output_dir, 'Stats_ContinuousGeometry.csv'), index=False)
pd.concat([vals_hemi.assign(Split='Hemifield'), vals_rad.assign(Split='DiagType'),
           vals_rad_hemi.assign(Split='DiagType x Hemifield')]).to_csv(
    os.path.join(output_dir, 'SubjectValues.csv'), index=False)
pd.concat([an_tms.assign(Split='TMS'), an_hemi.assign(Split='TMS x Hemifield')]).merge(
    pd.concat([rsa_tms, rsa_hemi]), how='left').to_csv(os.path.join(output_dir, 'SubjectValues_ContinuousGeometry.csv'), index=False)

pd.set_option('display.width', 220)
show = sdf[['Figure', 'Subset', 'Metric', 'Contrast', 'TMS', 'N', 'CI_mean', 'CI_sem', 'p_perm', 'Cohen_d']].copy()
for c in ['CI_mean', 'CI_sem', 'Cohen_d']: show[c] = show[c].map(lambda x: f'{x:.3f}')
show['p_perm'] = show['p_perm'].map(lambda x: f'{x:.4f}')
print(show.to_string(index=False))
print('\nContinuous geometry:')
g = gdf.copy()
for c in ['Mean', 'SEM', 'Cohen_d']: g[c] = g[c].map(lambda x: f'{x:+.3f}')
g['p_perm_vs0'] = g['p_perm_vs0'].map(lambda x: f'{x:.4f}')
print(g.to_string(index=False))
print('\nSAT correlations:')
print(pd.DataFrame(sat_stats).round(4).to_string(index=False))
print(f'\nOutputs in {output_dir}')
