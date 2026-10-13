"""
R08_radial_pull.py

Close look at the one TMS pattern that recurred in both datasets in the effect
atlas (R07): the pull of reports toward the radial axis at the phosphene-field
(inPF) location.

Measure. For each trial, phi = signed angle of the target orientation from the
radial axis at its recorded stimulus position, e = report - target (deg).
    pull = -e * sign(sin 2*phi)          (+ = report rotated toward the radial axis)
Targets are 'radial' if |phi| < 45 (closer to the radial axis) and
'tangential' otherwise. Per subject x TMS x location x type: mean pull.

Effect. E = [(R - T) under TMS] - [(R - T) under None]; negative = the radial /
tangential difference in pull shrinks under TMS.

Outputs (COMPARE_DIR/radial_pull):
  P0_Schematic, P1_MainResult, P2_PullVsDistance, P3_SubjectChanges,
  P4_Robustness, Stats_RadialPull.csv, Stats_RadialPull_Robustness.csv

Usage: python R08_radial_pull.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

from biasrep_common import COMPARE_DIR, STUDIES
import panels as P
from study_results import load_study

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
OUT = os.path.join(COMPARE_DIR, 'radial_pull')
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
COL_R, COL_T = '#1F4E9A', '#7FA7DC'
COL_NONE, COL_EARLY, COL_MID = '#333333', P.TMS_LINE_PAL['Early TMS'], P.TMS_LINE_PAL['Middle TMS']


def perm_p(v, n=20000):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    s = RNG.choice([-1.0, 1.0], size=(n, len(v)))
    return max(np.mean(np.abs((s * v).mean(1)) >= abs(v.mean())), 1 / n)


def summarize(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    h = stats.t.ppf(.975, len(v) - 1) * stats.sem(v)
    return dict(N=len(v), Mean=v.mean(), CI_low=v.mean() - h, CI_high=v.mean() + h, dz=v.mean() / v.std(ddof=1),
                n_negative=int((v < 0).sum()), p=perm_p(v))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def trials(study):
    d = load_study(study)
    d['pull'] = -d['errOri_scaled'] * np.sign(np.sin(np.radians(2 * d['PhiRadial'])))
    d['Study'] = STUDIES[study]['label']
    d['sid'] = d['Study'] + '_' + d['subjID'].astype(str)
    d['inPF_VF'] = d['sid'].map(d[d['Hemifield'] == 'inPF'].groupby('sid')['VF'].first())
    day1 = d[d['day'] == 1].groupby('sid')['TMS'].agg(lambda s: s.iloc[0] if s.nunique() == 1 else 'mixed')
    d['order'] = d['sid'].map(day1.map({'No TMS': 'No-TMS day first', 'Middle TMS': 'TMS day first', 'mixed': 'mixed',
                                        'Early TMS': 'interleaved'}))
    if study == 'orig':
        d['order'] = 'interleaved'
    return d


D = pd.concat([trials('orig'), trials('rep')], ignore_index=True)


def cell_means(d, type_col='GeoType', stat='mean'):
    return (d.groupby(['Study', 'sid', 'TMS', 'Hemifield', type_col])['pull'].agg(stat)
            .unstack(type_col).rename(columns={'Radial': 'R', 'Tangential': 'T'}).reset_index())


def effect(cm, tms, loc, sids=None):
    """Per-subject E = (R-T)_tms - (R-T)_None at one location."""
    w = cm[cm['Hemifield'] == loc].assign(RT=lambda x: x['R'] - x['T']).pivot(index='sid', columns='TMS', values='RT')
    e = (w[tms] - w['No TMS']).dropna()
    return e if sids is None else e[e.index.isin(sids)]


CM = cell_means(D)
S1 = sorted(D.loc[D['Study'] == 'Study 1', 'sid'].unique())
REP = sorted(D.loc[D['Study'] == 'Replication', 'sid'].unique())
CONTRASTS = [('Study 1 · Early', 'Early TMS', S1, COL_EARLY), ('Study 1 · Middle', 'Middle TMS', S1, COL_MID),
             ('Replication · Middle', 'Middle TMS', REP, '#0B5394'), ('Both datasets · Middle', 'Middle TMS', S1 + REP, '#5B2C83')]

# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------
rows = []
for name, tms, sids, _ in CONTRASTS:
    for loc in ['inPF', 'outPF']:
        rows.append({'Contrast': name, 'Quantity': f'E at {loc} (change in R−T pull)', **summarize(effect(CM, tms, loc, sids))})
        for typ in ['R', 'T']:
            w = CM[CM['Hemifield'] == loc].pivot(index='sid', columns='TMS', values=typ)
            v = (w[tms] - w['No TMS']).dropna(); v = v[v.index.isin(sids)]
            rows.append({'Contrast': name, 'Quantity': f'{"radial" if typ == "R" else "tangential"} targets at {loc}: TMS − None', **summarize(v)})
    ei, eo = effect(CM, tms, 'inPF', sids), effect(CM, tms, 'outPF', sids)
    rows.append({'Contrast': name, 'Quantity': 'location specificity: E(inPF) − E(outPF)', **summarize((ei - eo).dropna())})
# Baseline: tangential pull inPF vs outPF without TMS; R-T at each location without TMS
for name, sids in [('Study 1', S1), ('Replication', REP), ('Both datasets', S1 + REP)]:
    base = CM[(CM['TMS'] == 'No TMS') & CM['sid'].isin(sids)]
    w = base.pivot(index='sid', columns='Hemifield', values='T')
    rows.append({'Contrast': f'{name} · No TMS', 'Quantity': 'tangential pull: inPF − outPF', **summarize((w['inPF'] - w['outPF']).dropna())})
    for loc in ['inPF', 'outPF']:
        b = base[base['Hemifield'] == loc]
        rows.append({'Contrast': f'{name} · No TMS', 'Quantity': f'R − T pull at {loc}', **summarize((b['R'] - b['T']).values)})
ST = pd.DataFrame(rows)
ST.to_csv(os.path.join(OUT, 'Stats_RadialPull.csv'), index=False)

# ---------------------------------------------------------------------------
# Robustness of E at inPF
# ---------------------------------------------------------------------------
rob = []


def add(variant, e_by_contrast):
    for name, e in e_by_contrast.items():
        rob.append({'Variant': variant, 'Contrast': name, **summarize(e)})


def per_contrast(cm, subset=None):
    out = {}
    for name, tms, sids, _ in CONTRASTS:
        s = sids if subset is None else [x for x in sids if x in subset]
        if len(s) >= 6: out[name] = effect(cm, tms, 'inPF', s)
    return out


add('Main: geometry, |φ| < 45° = radial', per_contrast(CM))
add('Paper category labels', per_contrast(cell_means(D, 'DiagType')))
strict = D[(D['DistRadial'] < 30) | (D['DistRadial'] > 60)]
add('Strict geometry: radial < 30°, tangential > 60°', per_contrast(cell_means(strict)))
add('Subject medians instead of means', per_contrast(cell_means(D, stat='median')))
for vf, lab in [('LVF', 'phosphene field on the LEFT'), ('RVF', 'phosphene field on the RIGHT')]:
    add(f'Only subjects with {lab}', per_contrast(CM, set(D.loc[D['inPF_VF'] == vf, 'sid'])))
for o in ['No-TMS day first', 'TMS day first']:
    s = set(D.loc[D['order'] == o, 'sid'])
    e = effect(CM, 'Middle TMS', 'inPF', [x for x in REP if x in s])
    if len(e) >= 6:
        rob.append({'Variant': f'Replication, {o}', 'Contrast': 'Replication · Middle', **summarize(e)})
    elif len(e) >= 1:   # too few subjects for a test: report the mean only
        rob.append({'Variant': f'Replication, {o}', 'Contrast': 'Replication · Middle', 'N': len(e), 'Mean': e.mean(),
                    'CI_low': np.nan, 'CI_high': np.nan, 'dz': np.nan, 'n_negative': int((e < 0).sum()), 'p': np.nan})

# Trial-level mixed model at inPF: pull ~ type x TMS (+ study), random intercept and type slope per subject
inpf = D[D['Hemifield'] == 'inPF'].copy()
inpf['Rad'] = (inpf['GeoType'] == 'Radial').astype(float) - 0.5
for name, tms, sids, _ in CONTRASTS:
    dd = inpf[inpf['sid'].isin(sids) & inpf['TMS'].isin(['No TMS', tms])].copy()
    dd['TMSon'] = (dd['TMS'] == tms).astype(float)
    f = 'pull ~ Rad * TMSon' + (' + C(Study)' if dd['Study'].nunique() > 1 else '')
    best = None
    for meth in ('lbfgs', 'bfgs', 'powell'):
        try:
            m = smf.mixedlm(f, dd, groups=dd['sid'], re_formula='~Rad').fit(reml=True, method=meth)
            if best is None or m.llf > best.llf: best = m
        except Exception:
            pass
    est, se = best.fe_params['Rad:TMSon'], best.bse['Rad:TMSon']
    rob.append({'Variant': 'Trial-level mixed model (Rad × TMS)', 'Contrast': name, 'N': dd['sid'].nunique(), 'Mean': est,
                'CI_low': est - 1.96 * se, 'CI_high': est + 1.96 * se, 'dz': np.nan, 'n_negative': np.nan,
                'p': best.pvalues['Rad:TMSon']})
RB = pd.DataFrame(rob)
RB.to_csv(os.path.join(OUT, 'Stats_RadialPull_Robustness.csv'), index=False)

# ---------------------------------------------------------------------------
# P0: schematic of the measure
# ---------------------------------------------------------------------------
def draw_line(ax, x, y, ori, L, **kw):
    dx, dy = np.cos(np.radians(ori)) * L / 2, np.sin(np.radians(ori)) * L / 2
    ax.plot([x - dx, x + dx], [y - dy, y + dy], solid_capstyle='round', **kw)


fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
elev = 22
sx, sy = -np.cos(np.radians(elev)) * 9, -np.sin(np.radians(elev)) * 9      # stimulus in the lower-left
rad_ori = elev                                                            # radial axis orientation (deg, CCW)
for ax, (title, rad_shift, tan_shift) in zip(axs, [('What is measured', 6, 6),
                                                   ('Without TMS (inPF)', 7, 1),
                                                   ('With TMS (inPF)', 5, 5)]):
    ax.plot(0, 0, 'k+', ms=14, mew=2.5)
    ax.plot([0, 1.55 * sx], [0, 1.55 * sy], ls=(0, (2, 2)), color='0.45', lw=1.2)
    ax.add_patch(Circle((sx, sy), 2.6, fill=False, color='0.6', lw=1))
    if title == 'What is measured':
        draw_line(ax, sx, sy, rad_ori + 25, 5.2, color=COL_R, lw=3.5)
        draw_line(ax, sx, sy, rad_ori + 25 - rad_shift * 2.5, 5.2, color=COL_R, lw=1.6, ls='--')
        ax.add_patch(FancyArrowPatch((sx + 2.9 * np.cos(np.radians(rad_ori + 25)), sy + 2.9 * np.sin(np.radians(rad_ori + 25))),
                                     (sx + 2.9 * np.cos(np.radians(rad_ori + 25 - rad_shift * 2.5)), sy + 2.9 * np.sin(np.radians(rad_ori + 25 - rad_shift * 2.5))),
                                     connectionstyle='arc3,rad=0.3', arrowstyle='-|>', mutation_scale=14, color='crimson', lw=1.8))
        ax.text(sx, sy - 4.0, 'target (solid) → report (dashed)\nrotated toward the radial line\n= positive "pull"', ha='center', va='top', fontsize=9)
        ax.text(0.6 * sx + 0.8, 0.6 * sy + 0.8, 'radial axis\n(line through fixation)', fontsize=8, color='0.35', rotation=elev)
    else:
        for ori, shift, col, lab in [(rad_ori + 20, rad_shift, COL_R, 'radial target'), (rad_ori + 70, tan_shift, COL_T, 'tangential target')]:
            draw_line(ax, sx, sy, ori, 5.2, color=col, lw=3.5, alpha=0.9)
            draw_line(ax, sx, sy, ori - shift * 2.5, 5.2, color=col, lw=1.6, ls='--')
        ax.text(sx, sy - 4.0, 'radial targets pulled toward the radial axis;\ntangential targets barely' if 'Without' in title
                else 'tangential targets now pulled\nas much as radial targets', ha='center', va='top', fontsize=9)
    ax.set_xlim(-16, 4); ax.set_ylim(-9.5, 2.5); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
fig.suptitle('Pull toward the radial axis: how far the remembered orientation is rotated toward the line joining fixation and the stimulus\n'
             '(rotations exaggerated ×2.5 for display; real effects are ~1–2°)', fontsize=10, y=1.03)
P.save(fig, OUT, 'P0_Schematic')

# ---------------------------------------------------------------------------
# P1: main result — pull for radial and tangential targets, inPF vs outPF
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
for r, (study, sids) in enumerate([('orig', S1), ('rep', REP)]):
    G = STUDIES[study]['groups']
    for c, loc in enumerate(['inPF', 'outPF']):
        ax = axs[r, c]
        sub = CM[(CM['Hemifield'] == loc) & CM['sid'].isin(sids)]
        for typ, col, lab in [('R', COL_R, 'radial targets'), ('T', COL_T, 'tangential targets')]:
            g = sub.groupby('TMS')[typ].agg(['mean', 'sem']).reindex(G)
            ax.errorbar(np.arange(len(G)) + (-0.04 if typ == 'R' else 0.04), g['mean'], yerr=g['sem'], color=col, marker='o',
                        ms=8, lw=2.5, capsize=4, label=lab)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(range(len(G))); ax.set_xticklabels([P.TMS_TICK[t] for t in G])
        ax.set_title(f"{STUDIES[study]['label']} · {'phosphene-field location (inPF)' if loc == 'inPF' else 'mirror location (outPF)'}", fontsize=10)
        if c == 0: ax.set_ylabel('Pull toward radial axis (°)')
        e = effect(CM, 'Middle TMS', loc, sids)
        ax.text(0.98, 0.04, f'Change in radial−tangential gap (Middle):\n{e.mean():+.2f}°, p = {perm_p(e):.3f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8)
        sns.despine(ax=ax)
axs[0, 0].legend(frameon=False, fontsize=9, loc='upper right')
fig.suptitle('Without TMS, radial targets are pulled toward the radial axis much more than tangential targets at the phosphene-field location.\n'
             'Under TMS that gap closes there (tangential targets get pulled too); the mirror location shows no consistent change. Mean ± SEM.',
             fontsize=10, y=1.01)
plt.tight_layout()
P.save(fig, OUT, 'P1_MainResult')

# ---------------------------------------------------------------------------
# P2: pull vs distance from the radial axis (continuous)
# ---------------------------------------------------------------------------
edges = np.arange(0, 91, 15); mids = edges[:-1] + 7.5
D['dbin'] = pd.cut(D['DistRadial'], edges, labels=mids, include_lowest=True)
fig, axs = plt.subplots(3, 2, figsize=(11, 12), sharey=True, sharex=True)
for r, (name, sids, groups) in enumerate([('Study 1', S1, STUDIES['orig']['groups']), ('Replication', REP, STUDIES['rep']['groups']),
                                          ('Both datasets (None vs Middle)', S1 + REP, ['No TMS', 'Middle TMS'])]):
    for c, loc in enumerate(['inPF', 'outPF']):
        ax = axs[r, c]
        sub = D[(D['Hemifield'] == loc) & D['sid'].isin(sids)]
        for t in groups:
            g = sub[sub['TMS'] == t].groupby(['sid', 'dbin'], observed=True)['pull'].mean().reset_index()
            s = g.groupby('dbin', observed=True)['pull'].agg(['mean', 'sem', 'count'])
            col = {'No TMS': COL_NONE, 'Early TMS': COL_EARLY, 'Middle TMS': COL_MID}[t]
            ax.errorbar(s.index.astype(float) + (groups.index(t) - 1) * 1.2, s['mean'], yerr=s['sem'], color=col, marker='o',
                        lw=2, capsize=3, label=P.TMS_SHORT[t])
        ax.axhline(0, color='k', lw=0.8); ax.axvline(45, color='0.6', ls=':', lw=1)
        ax.set_title(f'{name} · {loc}', fontsize=10)
        if r == 2: ax.set_xlabel('Distance of target from the radial axis (°)\n0 = radial · 90 = tangential')
        if c == 0: ax.set_ylabel('Pull toward radial axis (°)')
        ax.set_xticks(range(0, 91, 15))
        ax.legend(frameon=False, fontsize=8)
        sns.despine(ax=ax)
fig.suptitle('Pull toward the radial axis as a function of where the target sits relative to it (15° bins; mean ± SEM over subjects)', fontsize=10, y=1.0)
plt.tight_layout()
P.save(fig, OUT, 'P2_PullVsDistance')

# ---------------------------------------------------------------------------
# P3: per-subject changes
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for ax, loc in zip(axs, ['inPF', 'outPF']):
    for i, (name, tms, sids, col) in enumerate(CONTRASTS):
        e = effect(CM, tms, loc, sids)
        x = i + RNG.uniform(-0.12, 0.12, len(e))
        ax.scatter(x, e.values, color=col, alpha=0.45, s=22, edgecolor='none')
        s = summarize(e)
        ax.errorbar(i + 0.3, s['Mean'], yerr=[[s['Mean'] - s['CI_low']], [s['CI_high'] - s['Mean']]], fmt='D', color=col, ms=8, capsize=4)
        ax.text(i + 0.3, 0.02, f"{s['Mean']:+.2f}°\np={s['p']:.3f}\n{s['n_negative']}/{s['N']} ↓", transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7)
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(range(len(CONTRASTS))); ax.set_xticklabels([c[0].replace(' · ', '\n') for c in CONTRASTS], fontsize=8)
    ax.set_title('phosphene-field location (inPF)' if loc == 'inPF' else 'mirror location (outPF)', fontsize=11)
    if loc == 'inPF': ax.set_ylabel('Change under TMS in the radial − tangential\ndifference in pull (°); negative = gap closes')
    sns.despine(ax=ax)
fig.suptitle('Per-subject change in the radial − tangential pull difference (dots = subjects; diamonds = mean ± 95% CI; "↓" = subjects with a decrease)',
             fontsize=10, y=1.02)
plt.tight_layout()
P.save(fig, OUT, 'P3_SubjectChanges')

# ---------------------------------------------------------------------------
# P4: robustness forest
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 0.33 * len(RB) + 2))
y = 0; yt, yl = [], []
colmap = {c[0]: c[3] for c in CONTRASTS}
for variant, g in RB.groupby('Variant', sort=False):
    for _, r in g.iterrows():
        ax.errorbar(r['Mean'], y, xerr=[[r['Mean'] - r['CI_low']], [r['CI_high'] - r['Mean']]] if not np.isnan(r['CI_low']) else None,
                    fmt='o', color=colmap[r['Contrast']], ms=6, capsize=2)
        ax.text(1.01, y, f"{r['Contrast']}: {r['Mean']:+.2f}" + (f", p={r['p']:.3f}" if not np.isnan(r['p']) else f" (n={int(r['N'])})"),
                transform=ax.get_yaxis_transform(), va='center', fontsize=7, color=colmap[r['Contrast']])
        y += 1
    yt.append(y - len(g) / 2 - 0.5); yl.append(variant)
    y += 0.7
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.invert_yaxis()
ax.axvline(0, color='k', lw=1)
ax.set_xlabel('Change under TMS in the radial − tangential pull difference at inPF (°), 95% CI')
sns.despine(ax=ax)
fig.suptitle('Robustness: the effect under alternative definitions, subsets and models', fontsize=10, y=1.0)
plt.tight_layout()
P.save(fig, OUT, 'P4_Robustness')

pd.set_option('display.width', 220)
print(ST.round(3).to_string(index=False))
print()
print(RB.round(3).to_string(index=False))
