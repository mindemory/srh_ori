"""
Panel-drawing helpers shared by the comparison figures. Styling follows
analyzeBehav/V10_paperFigures.py; every function takes the study's TMS groups
so the same panel works for Study 1 (None/Early/Middle) and the replication
(None/Middle).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from biasrep_common import run_permutation_test

TMS_BOX_PAL = {'No TMS': '#9E9E9E', 'Early TMS': '#EF9A9A', 'Middle TMS': '#90CAF9'}
TMS_STRIP_PAL = {'No TMS': 'k', 'Early TMS': '#C62828', 'Middle TMS': '#1565C0'}
TMS_LINE_PAL = {'No TMS': 'k', 'Early TMS': '#D32F2F', 'Middle TMS': '#29A3E0'}
TMS_SHORT = {'No TMS': 'None', 'Early TMS': 'Early Delay', 'Middle TMS': 'Middle Delay'}
TMS_TICK = {'No TMS': 'None', 'Early TMS': 'Early\nDelay', 'Middle TMS': 'Middle\nDelay'}


def fmt_p(p):
    return 'p<.001' if p < 0.001 else f'p={p:.3f}'


def _box(ax, vals, pos, width, t, is_A, hatch=None, s=11):
    col, scol = TMS_BOX_PAL[t], TMS_STRIP_PAL[t]
    al, sal = (1.0, 0.7) if is_A else (0.65, 0.35)
    ax.boxplot([vals], positions=[pos], widths=width, patch_artist=True, showfliers=False, manage_ticks=False,
               boxprops=dict(facecolor=col, alpha=al, linewidth=0.8, edgecolor=scol, hatch=hatch),
               medianprops=dict(color='white', linewidth=2),
               whiskerprops=dict(color=scol, alpha=al, linewidth=0.8),
               capprops=dict(color=scol, alpha=al, linewidth=0.8))
    rng = np.random.default_rng(int(pos * 1000) % 2**32)
    ax.scatter(pos + rng.uniform(-width * 0.28, width * 0.28, len(vals)), vals, color=scol, alpha=sal, s=s, zorder=3)


def paired_boxes(ax, piv, A, B, groups, ylim=None, ylabel='', ref=0.5, leg=True):
    """V10 _draw_paired_boxes: normalized A/(A+B) and B/(A+B) per TMS group.
    The p under each pair is the paper's test (A_norm vs 0.5)."""
    bw, gap = 0.28, 0.15
    xt = []
    for i, t in enumerate(groups):
        base = i * (2 * bw + gap)
        xt.append(base + bw / 2)
        sub = piv[piv['TMS'] == t].dropna(subset=[f'{A}_norm', f'{B}_norm'])
        for pos, cond in [(base, A), (base + bw, B)]:
            if len(sub) >= 3:
                _box(ax, sub[f'{cond}_norm'].values, pos, bw, t, cond == A)
        if len(sub) > 5:
            ci = sub['CI'].values
            ax.text(base + bw / 2, 0.01, f'{ci.mean():+.3f}\n{fmt_p(run_permutation_test(sub[f"{A}_norm"].values - ref))}',
                    transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
    ax.axhline(ref, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks(xt); ax.set_xticklabels([TMS_TICK[t] for t in groups], fontsize=9)
    ax.set_xlim(-bw * 0.8, (len(groups) - 1) * (2 * bw + gap) + bw * 1.8)
    if ylim: ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    if leg:
        ax.legend(handles=[mpatches.Patch(facecolor='#424242', label=A),
                           mpatches.Patch(facecolor=TMS_BOX_PAL['No TMS'], alpha=0.65, label=B)],
                  frameon=True, fontsize=7, loc='upper right')
    sns.despine(ax=ax, offset=5)


def overlay_boxes(ax, piv_by_hemi, A, B, groups, ylim=None, ylabel=''):
    """V10 _draw_overlay_boxes: radial/tangential normalized values for inPF
    (solid) and outPF (hatched) per TMS group."""
    bw, inner_gap, mid_gap, tms_gap = 0.22, 0.04, 0.14, 0.30
    step = 4 * bw + inner_gap + mid_gap + tms_gap
    xt = []
    for i, t in enumerate(groups):
        base = i * step
        pos_map = {(A, 'inPF'): base, (B, 'inPF'): base + bw + inner_gap,
                   (A, 'outPF'): base + 2 * bw + inner_gap + mid_gap, (B, 'outPF'): base + 3 * bw + 2 * inner_gap + mid_gap}
        xt.append(base + 1.5 * bw + inner_gap + mid_gap / 2)
        for (cond, hemi), pos in pos_map.items():
            p = piv_by_hemi[hemi]
            vals = p[p['TMS'] == t].dropna(subset=[f'{A}_norm', f'{B}_norm'])[f'{cond}_norm'].values
            if len(vals) >= 3:
                _box(ax, vals, pos, bw, t, cond == A, hatch=None if hemi == 'inPF' else '///', s=9)
    ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks(xt); ax.set_xticklabels([TMS_TICK[t] for t in groups], fontsize=9)
    if ylim: ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    nc = TMS_BOX_PAL['No TMS']
    ax.legend(handles=[mpatches.Patch(facecolor=nc, label=f'inPF · {A}'),
                       mpatches.Patch(facecolor=nc, alpha=0.65, label=f'inPF · {B}'),
                       mpatches.Patch(facecolor=nc, hatch='///', label=f'outPF · {A}'),
                       mpatches.Patch(facecolor=nc, alpha=0.65, hatch='///', label=f'outPF · {B}')],
              frameon=False, fontsize=6, loc='upper right')
    sns.despine(ax=ax, offset=5)


def scatter_pair(ax, vals, p_col, A, B, met, groups, xlabel, ylabel, x_is_A=False, lims=None):
    piv = vals.pivot_table(index=['subjID', 'TMS'], columns=p_col, values=met, observed=False).reset_index()
    xc, yc = (A, B) if x_is_A else (B, A)
    for t in groups:
        sub = piv[piv['TMS'] == t].dropna(subset=[A, B])
        c = TMS_LINE_PAL[t]
        ax.scatter(sub[xc], sub[yc], color=c, alpha=0.4, s=25, edgecolor='white')
        ax.errorbar(sub[xc].mean(), sub[yc].mean(), xerr=sub[xc].sem(), yerr=sub[yc].sem(), fmt='o', color=c,
                    markersize=8, markeredgecolor='white', label=TMS_SHORT[t])
    if lims is None:
        v = piv[[A, B]].values.flatten(); lo, hi = np.nanmin(v), np.nanmax(v)
        pad = 0.05 * (hi - lo); lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, 'k--', alpha=0.6, lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal', 'box'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, frameon=True, loc='upper left')
    sns.despine(ax=ax, offset=5)


def hemi_raw_boxes(ax, tab, value, groups, ylabel, ref=0.0, ylim=None):
    """Per-subject parameter for inPF (solid) / outPF (lighter) per TMS group,
    with the paired inPF - outPF p under each pair."""
    bw, gap = 0.28, 0.15
    xt = []
    for i, t in enumerate(groups):
        base = i * (2 * bw + gap)
        xt.append(base + bw / 2)
        w = tab[tab['TMS'] == t].pivot(index='subjID', columns='Hemifield', values=value)
        for pos, hemi in [(base, 'inPF'), (base + bw, 'outPF')]:
            _box(ax, w[hemi].dropna().values, pos, bw, t, hemi == 'inPF')
        d = (w['inPF'] - w['outPF']).dropna().values
        ax.text(base + bw / 2, 0.01, f'in−out\n{fmt_p(run_permutation_test(d))}', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7)
    ax.axhline(ref, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks(xt); ax.set_xticklabels([TMS_TICK[t] for t in groups], fontsize=9)
    ax.set_xlim(-bw * 0.8, (len(groups) - 1) * (2 * bw + gap) + bw * 1.8)
    if ylim: ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[mpatches.Patch(facecolor='#424242', label='inPF'),
                       mpatches.Patch(facecolor=TMS_BOX_PAL['No TMS'], alpha=0.65, label='outPF')],
              frameon=True, fontsize=7, loc='upper right')
    sns.despine(ax=ax, offset=5)


def tms_paired(ax, tab, value, groups, ylabel, ref=0.0, ylim=None):
    """Per-subject value per TMS group joined by lines. Each group's test vs
    `ref` is printed above it; None minus each TMS group at the bottom."""
    w = tab.pivot(index='subjID', columns='TMS', values=value).reindex(columns=groups)
    for _, row in w.iterrows():
        ax.plot(range(len(groups)), row.values, color='0.75', lw=0.8, zorder=1)
    for i, t in enumerate(groups):
        v = w[t].dropna().values
        _box(ax, v, i, 0.35, t, True, s=12)
        ax.text(i, 1.0, f'vs {ref:g}: {fmt_p(run_permutation_test(v - ref))}', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7)
    lines = []
    for t in groups[1:]:
        d = (w[groups[0]] - w[t]).dropna().values
        lines.append(f'None−{TMS_SHORT[t].split()[0]}: {d.mean():+.3f}, {fmt_p(run_permutation_test(d))}')
    ax.text(0.5, 0.01, '\n'.join(lines), transform=ax.transAxes, ha='center', va='bottom', fontsize=7)
    ax.axhline(ref, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([TMS_TICK[t] for t in groups], fontsize=9)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    if ylim: ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax, offset=5)


def binned_curve(ax, curve, value, groups, ylabel, split_hemi=False, xlabel=None):
    """curve: per-subject means with columns subjID, TMS, mid (bin centre), value[, Hemifield]."""
    for t in groups:
        for h in (['inPF', 'outPF'] if split_hemi else [None]):
            c = curve[(curve['TMS'] == t) & ((curve['Hemifield'] == h) if h else True)]
            g = c.groupby('mid')[value].agg(['mean', 'sem'])
            off = (groups.index(t) - (len(groups) - 1) / 2) * 1.6 + ((0.6 if h == 'outPF' else -0.6) if h else 0)
            ax.errorbar(g.index + off, g['mean'], yerr=g['sem'], color=TMS_LINE_PAL[t], marker='o', ms=4, capsize=2,
                        ls='--' if h == 'outPF' else '-', lw=1.5, label=TMS_SHORT[t] + (f' · {h}' if h else ''))
    ax.set_xlabel(xlabel or 'Distance of target from radial axis (°)\n0 = radial, 90 = tangential')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, frameon=False)
    sns.despine(ax=ax, offset=5)


def column_titles(axes, titles):
    for a, t in zip(axes, titles):
        a.annotate(t, xy=(0.5, 1.18), xycoords='axes fraction', ha='center', va='bottom', fontsize=12, fontweight='bold')


def row_label(ax, text):
    ax.annotate(text, xy=(-0.42, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=12,
                fontweight='bold', rotation=90)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}')
