"""
R07_effect_atlas.py

Exhaustive, uncorrected screen of TMS effects in Study 1 and the replication:
every measure x every split x every contrast form. Replaces R05.

Measures (per subject x TMS condition x cell; see MEASURES below).
Cells: all trials; inPF / outPF; Radial / Tangential by the paper's category
labels and by actual geometry (target closer to the radial or the tangential
axis at its recorded position); and the Radial / Tangential cells within inPF
and within outPF.

Splits and forms. For a split with cells A and B:
    diff = A - B,  ratio = A / B,  norm = (A - B) / (A + B)
ratio and norm only for measures that are always positive (rates with many
zero cells and signed measures get diff only). Each split is tested
  - per TMS condition, contrast vs no difference (0 for diff/norm, 1 for ratio)
  - TMS - None: paired difference of the contrast
For the radial splits within hemifields, 'inPF - outPF' rows test whether the
radial contrast differs between locations. The overall TMS effect (no split)
uses the same three forms on (TMS, None).

Tests: sign-flip permutation across subjects (10,000; two-sided), dz = mean
relative to the null / SD. No multiple-comparison correction.

Usage: python R07_effect_atlas.py
"""
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from biasrep_common import COMPARE_DIR, STUDIES, md
import panels as P
from study_results import load_study

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
OUT = os.path.join(COMPARE_DIR, 'effect_atlas')
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
RESP_WINDOW = 3.0

MEASURES = OrderedDict([
    # name: (description, positive-valued -> all three forms)
    ('AbsError', ('mean |report − target| (°); lower = more accurate', True)),
    ('CircSD', ('circular SD of signed errors (°); spread', True)),
    ('Kappa', ('precision of the von Mises part of the mixture; higher = more precise', True)),
    ('CardinalRepulsion', ('mean error away from the nearest cardinal (°); + = pushed toward the diagonals', False)),
    ('RadialPull', ('mean error toward the radial axis (°); + = pulled toward radial', False)),
    ('SB_prevTarget', ('serial bias: error toward the previous target (°), |Δ| ≤ 45°', False)),
    ('SB_prevResponse', ('serial bias: error toward the previous response (°)', False)),
    ('SB_sameLocation', ('serial bias toward previous target when it was at the same location (°)', False)),
    ('SB_diffLocation', ('serial bias toward previous target when it was at the other location (°)', False)),
    ('ErrorCarryover', ('slope of current error on previous error; + = errors repeat', False)),
    ('RT', ('response time (s)', True)),
    ('RT_SD', ('within-subject SD of response time (s)', True)),
    ('TimeoutRate', ('fraction of trials hitting the 3 s response limit', False)),
    ('LargeErrorRate', ('fraction of trials beyond the 3 SD error cutoff', False)),
    ('StartPull', ('slope of error on starting offset (|offset| < 30°); 0.1 = 10% of the offset remains', False)),
    ('WrongDirection', ('fraction of trials (start 15–30° away) whose first dial turn went the long way', False)),
    ('ExcessTravel', ('dial travel beyond the start-to-response distance (°); corrections', True)),
    ('Reversals', ('changes of dial direction per trial', True)),
])
FORMS = ['diff', 'ratio', 'norm']
SHORT = {'No TMS': 'No TMS', 'Early TMS': 'Early', 'Middle TMS': 'Middle'}


def wrap90(x):
    return (x + 90) % 180 - 90


def fast_perm(vals, null=0.0, n_perms=10000):
    v = np.asarray(vals, float); v = v[~np.isnan(v)] - null
    if len(v) < 6: return np.nan
    signs = RNG.choice([-1.0, 1.0], size=(n_perms, len(v)))
    p = np.mean(np.abs((signs * v).mean(1)) >= abs(v.mean()))
    return max(p, 1 / n_perms)


def fourier(theta_deg):
    th = np.radians(theta_deg)
    return np.column_stack([np.ones(len(th))] + [f(k * th) for k in (2, 4, 8, 12) for f in (np.sin, np.cos)])


def dial_features(arr, init_offset):
    if len(arr) < 2: return np.nan, 0.0, 0
    d = wrap90(np.diff(arr)); nz = d[d != 0]
    if len(nz) == 0: return np.nan, 0.0, 0
    wrong = float(np.sign(nz[0]) == np.sign(init_offset)) if 15 <= abs(init_offset) <= 30 else np.nan
    return wrong, np.abs(d).sum() - abs(wrap90(arr[-1] - arr[0])), int(np.sum(np.diff(np.sign(nz)) != 0))


def prepare(study):
    d = load_study(study, drop_outliers=False)
    e = d['errOri_scaled'].values
    d['resid'] = np.nan
    for _, idx in d.groupby(['subjID', 'TMS']).groups.items():
        sub = d.loc[idx]; ok = (sub['LargeError'] == 0).values
        X = fourier(sub['targOriActual'].values)
        beta = np.linalg.lstsq(X[ok], sub['errOri_scaled'].values[ok], rcond=None)[0]
        d.loc[idx, 'resid'] = sub['errOri_scaled'].values - X @ beta
    blk = d.groupby(['subjID', 'day', 'block'], sort=False)
    d['prevResid'] = blk['resid'].shift(1)
    d.loc[d['prevLargeError'] == 1, 'prevResid'] = np.nan
    d['dPrevT'] = wrap90(d['prevTarg'] - d['targOriActual'])
    d['dPrevR'] = wrap90(d['prevResp'] - d['targOriActual'])
    d['samePrevLoc'] = d['prevStimPF'] == d['stimPF']
    f = d.apply(lambda r: dial_features(r['oriArray'], r['initOffset']), axis=1, result_type='expand')
    d[['WrongDir', 'Excess', 'Reversals_trial']] = f.values
    return d


def serial(c, col):
    s = c[(c[col].abs() > 0) & (c[col].abs() <= 45)]
    return np.mean(s['resid'] * np.sign(s[col])) if len(s) >= 8 else np.nan


def cell_measures(sub):
    clean = sub[sub['LargeError'] == 0]
    if len(clean) < 8: return None
    e = clean['errOri_scaled'].values
    z = np.exp(1j * np.radians(2 * e)).mean()
    kappa = np.nan
    if len(e) >= 10:
        try: kappa = md.fit_model(e, 'mixtureModel', max_trials=100)[1]
        except ValueError: pass
    cp = clean.dropna(subset=['prevResid'])
    near = clean[clean['AbsInitOffset'] < 30]
    wd = clean['WrongDir'].dropna()
    return {
        'AbsError': np.abs(e).mean(),
        'CircSD': np.degrees(np.sqrt(-2 * np.log(np.abs(z)))) / 2,
        'Kappa': kappa,
        'CardinalRepulsion': np.mean(e * np.sign(np.sin(np.radians(4 * clean['targOriActual'].values)))),
        'RadialPull': np.mean(-e * np.sign(np.sin(np.radians(2 * clean['PhiRadial'].values)))),
        'SB_prevTarget': serial(clean, 'dPrevT'),
        'SB_prevResponse': serial(clean, 'dPrevR'),
        'SB_sameLocation': serial(clean[clean['samePrevLoc']], 'dPrevT'),
        'SB_diffLocation': serial(clean[~clean['samePrevLoc']], 'dPrevT'),
        'ErrorCarryover': np.polyfit(cp['prevResid'], cp['resid'], 1)[0] if len(cp) >= 10 else np.nan,
        'RT': clean['RT'].mean(), 'RT_SD': clean['RT'].std(),
        'TimeoutRate': (sub['RT'] >= RESP_WINDOW).mean(),
        'LargeErrorRate': sub['LargeError'].mean(),
        'StartPull': np.polyfit(near['initOffset'], near['errOri_scaled'], 1)[0] if len(near) >= 10 else np.nan,
        'WrongDirection': wd.mean() if len(wd) >= 5 else np.nan,
        'ExcessTravel': clean['Excess'].mean(), 'Reversals': clean['Reversals_trial'].mean()}


CELLS = OrderedDict([
    ('all', lambda d: d),
    ('inPF', lambda d: d[d['Hemifield'] == 'inPF']), ('outPF', lambda d: d[d['Hemifield'] == 'outPF']),
    ('R_cat', lambda d: d[d['DiagType'] == 'Radial']), ('T_cat', lambda d: d[d['DiagType'] == 'Tangential']),
    ('R_geo', lambda d: d[d['GeoType'] == 'Radial']), ('T_geo', lambda d: d[d['GeoType'] == 'Tangential']),
])
for h in ['inPF', 'outPF']:
    for lab, col in [('cat', 'DiagType'), ('geo', 'GeoType')]:
        for typ, short in [('Radial', 'R'), ('Tangential', 'T')]:
            CELLS[f'{short}_{lab}|{h}'] = (lambda h_, c_, t_: (lambda d: d[(d['Hemifield'] == h_) & (d[c_] == t_)]))(h, col, typ)

SPLITS = OrderedDict([
    ('Hemifield (inPF vs outPF)', ('inPF', 'outPF')),
    ('Radial vs Tangential (paper labels)', ('R_cat', 'T_cat')),
    ('Radial vs Tangential (geometry)', ('R_geo', 'T_geo')),
    ('Radial vs Tangential (paper labels) · inPF', ('R_cat|inPF', 'T_cat|inPF')),
    ('Radial vs Tangential (paper labels) · outPF', ('R_cat|outPF', 'T_cat|outPF')),
    ('Radial vs Tangential (geometry) · inPF', ('R_geo|inPF', 'T_geo|inPF')),
    ('Radial vs Tangential (geometry) · outPF', ('R_geo|outPF', 'T_geo|outPF')),
])


def form(a, b, f):
    if f == 'diff': return a - b
    ok = (a > 0) & (b > 0)
    return np.where(ok, a / b if f == 'ratio' else (a - b) / (a + b), np.nan)


def table_for(study):
    d = prepare(study)
    rows = []
    for cell, sel in CELLS.items():
        dc = sel(d)
        for (s, t), sub in dc.groupby(['subjID', 'TMS']):
            m = cell_measures(sub)
            if m: rows.append({'subjID': s, 'TMS': t, 'cell': cell, **m})
    return d, pd.DataFrame(rows)


def test_row(vals, null):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    if len(v) < 6: return dict(N=len(v), Mean=np.nan, SEM=np.nan, dz=np.nan, p=np.nan)
    return dict(N=len(v), Mean=v.mean(), SEM=stats.sem(v), dz=(v.mean() - null) / v.std(ddof=1), p=fast_perm(v, null))


def analyse(study, T):
    G = STUDIES[study]['groups']; lab = STUDIES[study]['label']
    out = []
    W = {m: T.pivot_table(index='subjID', columns=['cell', 'TMS'], values=m) for m in MEASURES}
    # overall TMS effect, three forms on (TMS, None)
    for m, (_, pos) in MEASURES.items():
        for t in G[1:]:
            for f in (FORMS if pos else ['diff']):
                c = form(W[m][('all', t)].values, W[m][('all', 'No TMS')].values, f)
                out.append({'Study': lab, 'Split': 'Overall (TMS vs None)', 'Measure': m, 'Form': f,
                            'Test': f'{SHORT[t]}−None', **test_row(c, 1.0 if f == 'ratio' else 0.0)})
    # splits
    contrasts = {}
    for sname, (A, B) in SPLITS.items():
        for m, (_, pos) in MEASURES.items():
            for f in (FORMS if pos else ['diff']):
                null = 1.0 if f == 'ratio' else 0.0
                C = {t: pd.Series(form(W[m][(A, t)].values, W[m][(B, t)].values, f), index=W[m].index) for t in G
                     if (A, t) in W[m] and (B, t) in W[m]}
                contrasts[(sname, m, f)] = C
                for t in G:
                    out.append({'Study': lab, 'Split': sname, 'Measure': m, 'Form': f, 'Test': SHORT[t], **test_row(C[t], null)})
                for t in G[1:]:
                    out.append({'Study': lab, 'Split': sname, 'Measure': m, 'Form': f, 'Test': f'{SHORT[t]}−None',
                                **test_row(C[t] - C['No TMS'], 0.0)})
    # radial contrast: inPF minus outPF
    for kind in ['paper labels', 'geometry']:
        si, so = f'Radial vs Tangential ({kind}) · inPF', f'Radial vs Tangential ({kind}) · outPF'
        for m, (_, pos) in MEASURES.items():
            for f in (FORMS if pos else ['diff']):
                Ci, Co = contrasts[(si, m, f)], contrasts[(so, m, f)]
                D = {t: Ci[t] - Co[t] for t in G}
                for t in G:
                    out.append({'Study': lab, 'Split': f'Radial vs Tangential ({kind}) · inPF−outPF', 'Measure': m, 'Form': f,
                                'Test': SHORT[t], **test_row(D[t], 0.0)})
                for t in G[1:]:
                    out.append({'Study': lab, 'Split': f'Radial vs Tangential ({kind}) · inPF−outPF', 'Measure': m, 'Form': f,
                                'Test': f'{SHORT[t]}−None', **test_row(D[t] - D['No TMS'], 0.0)})
    return pd.DataFrame(out)


print('Study 1...')
D1, T1 = table_for('orig')
print('Replication...')
D2, T2 = table_for('rep')
R = pd.concat([analyse('orig', T1), analyse('rep', T2)], ignore_index=True)
R.to_csv(os.path.join(OUT, 'EffectAtlas_AllTests.csv'), index=False)
pd.concat([T1.assign(Study='Study 1'), T2.assign(Study='Replication')]).to_csv(os.path.join(OUT, 'EffectAtlas_SubjectCells.csv'), index=False)
pd.DataFrame([(k, v[0], 'diff, ratio, norm' if v[1] else 'diff') for k, v in MEASURES.items()],
             columns=['Measure', 'Description', 'Forms']).to_csv(os.path.join(OUT, 'EffectAtlas_Glossary.csv'), index=False)

# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------
def heatmap(split_names, fname, title, overall=False):
    cols = ([('Study 1', 'Early−None'), ('Study 1', 'Middle−None'), ('Replication', 'Middle−None')] if overall else
            [('Study 1', 'No TMS'), ('Study 1', 'Early'), ('Study 1', 'Middle'), ('Study 1', 'Early−None'), ('Study 1', 'Middle−None'),
             ('Replication', 'No TMS'), ('Replication', 'Middle'), ('Replication', 'Middle−None')])
    row_keys = [(m, f) for m, (_, pos) in MEASURES.items() for f in (FORMS if pos else ['diff'])]
    fig, axs = plt.subplots(1, len(split_names), figsize=(4.2 + 3.6 * len(split_names) * len(cols) / 8, 0.32 * len(row_keys) + 2.5),
                            squeeze=False)
    for ax, sname in zip(axs[0], split_names):
        sub = R[R['Split'] == sname]
        Z = np.full((len(row_keys), len(cols)), np.nan); Pv = Z.copy()
        for i, (m, f) in enumerate(row_keys):
            for j, (st, te) in enumerate(cols):
                r = sub[(sub['Measure'] == m) & (sub['Form'] == f) & (sub['Study'] == st) & (sub['Test'] == te)]
                if len(r): Z[i, j], Pv[i, j] = r['dz'].iloc[0], r['p'].iloc[0]
        cm = matplotlib.colormaps['RdBu_r'].copy(); cm.set_bad('0.9')
        ax.imshow(np.clip(Z, -1.2, 1.2), cmap=cm, vmin=-1.2, vmax=1.2, aspect='auto')
        for i in range(len(row_keys)):
            for j in range(len(cols)):
                p = Pv[i, j]
                if not np.isnan(p) and p < 0.05:
                    ax.text(j, i, '***' if p < .001 else '**' if p < .01 else '*', ha='center', va='center', fontsize=8)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f'{st}\n{te}' for st, te in cols], rotation=90, fontsize=7)
        ax.set_yticks(range(len(row_keys)))
        ax.set_yticklabels([f'{m}  [{f}]' for m, f in row_keys] if ax is axs[0, 0] else [], fontsize=7)
        for y in np.cumsum([3 if MEASURES[m][1] else 1 for m in MEASURES])[:-1] - 0.5:
            ax.axhline(y, color='w', lw=0.6)
        if not overall: ax.axvline(4.5, color='k', lw=1)
        ax.set_title(sname, fontsize=9)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=matplotlib.colors.Normalize(-1.2, 1.2))
    cb = fig.colorbar(sm, ax=axs[0].tolist(), fraction=0.02, pad=0.01); cb.set_label('effect size dz', fontsize=8)
    fig.suptitle(title, fontsize=10, y=1.0)
    P.save(fig, OUT, fname)


heatmap(['Overall (TMS vs None)'], 'A0_Overall',
        'Overall TMS effect: diff = TMS − None, ratio = TMS / None, norm = (TMS − None)/(TMS + None)\n'
        'red = higher under TMS; * p<.05, ** p<.01, *** p<.001 (uncorrected)', overall=True)
heatmap(['Hemifield (inPF vs outPF)'], 'A1_Hemifield',
        'inPF vs outPF: diff = in − out, ratio = in/out, norm = (in − out)/(in + out)\n'
        'per-condition columns test the contrast vs no difference; "−None" columns test the change under TMS. Uncorrected.')
heatmap(['Radial vs Tangential (paper labels)', 'Radial vs Tangential (geometry)'], 'A2_Radial',
        'Radial vs Tangential: diff = R − T, ratio = R/T, norm = (R − T)/(R + T)\n'
        'e.g. AbsError diff < 0 (blue) = radial more accurate; Kappa diff > 0 (red) = radial more precise. Uncorrected.')
for kind in ['paper labels', 'geometry']:
    heatmap([f'Radial vs Tangential ({kind}) · inPF', f'Radial vs Tangential ({kind}) · outPF',
             f'Radial vs Tangential ({kind}) · inPF−outPF'], f"A3_Radial_byHemifield_{'labels' if kind == 'paper labels' else 'geometry'}",
            f'Radial vs Tangential ({kind}) within each location, and inPF − outPF of the radial contrast. Uncorrected.')

# ---------------------------------------------------------------------------
# History: serial-dependence curves
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
edges = np.arange(-90, 91, 15); mids = edges[:-1] + 7.5
for r, (d, study) in enumerate([(D1, 'orig'), (D2, 'rep')]):
    d = d[(d['LargeError'] == 0)]
    for c, (col, mask, ttl) in enumerate([('dPrevT', None, 'previous target (all)'), ('dPrevT', True, 'previous target, same location'),
                                          ('dPrevT', False, 'previous target, other location'), ('dPrevR', None, 'previous response')]):
        ax = axs[r, c]
        dd = d if mask is None else d[d['samePrevLoc'] == mask]
        dd = dd.dropna(subset=[col]).assign(bin=lambda x: pd.cut(x[col], edges, labels=mids))
        for t in STUDIES[study]['groups']:
            g = dd[dd['TMS'] == t].groupby(['subjID', 'bin'], observed=True)['resid'].mean().reset_index()
            s = g.groupby('bin', observed=True)['resid'].agg(['mean', 'sem'])
            off = (STUDIES[study]['groups'].index(t) - 1) * 1.5
            ax.errorbar(s.index.astype(float) + off, s['mean'], yerr=s['sem'], color=P.TMS_LINE_PAL[t], marker='o', ms=3,
                        capsize=2, lw=1.5, label=P.TMS_SHORT[t])
        ax.axhline(0, color='k', lw=0.8); ax.axvline(0, color='k', lw=0.8, ls=':')
        ax.set_title(f"{STUDIES[study]['label']} · {ttl}", fontsize=10)
        ax.set_xlabel('previous − current (°)'); ax.set_xticks(range(-90, 91, 30))
        if c == 0: ax.set_ylabel('current error, orientation bias removed (°)')
        ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax)
fig.suptitle('Serial dependence: positive slope through 0 = attraction toward the previous trial', y=0.99)
plt.tight_layout()
P.save(fig, OUT, 'H1_SerialDependenceCurves')

hits = R[R['p'] < 0.05].sort_values(['Split', 'Study', 'p'])
hits.to_csv(os.path.join(OUT, 'EffectAtlas_Uncorrected_p05.csv'), index=False)
n_tms = (R['Test'].str.contains('−None')).sum()
print(f'\n{len(R)} tests; TMS-change tests: {n_tms}; uncorrected p < .05 among TMS-change tests: '
      f"{(R['Test'].str.contains('−None') & (R['p'] < .05)).sum()} (≈{0.05 * n_tms:.0f} expected by chance)")
pd.set_option('display.width', 250)
print(hits[hits['Test'].str.contains('−None')][['Study', 'Split', 'Measure', 'Form', 'Test', 'N', 'Mean', 'dz', 'p']].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# Focus: pull toward the radial axis by location x orientation type x TMS
# ---------------------------------------------------------------------------
def radial_pull_figure(cells_tables):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (study, T) in zip(axs, cells_tables):
        G = STUDIES[study]['groups']
        for cell, col, ls, lab in [('R_geo|inPF', '#1F4E9A', '-', 'inPF · radial'), ('T_geo|inPF', '#8FB3E3', '-', 'inPF · tangential'),
                                   ('R_geo|outPF', '#1F4E9A', '--', 'outPF · radial'), ('T_geo|outPF', '#8FB3E3', '--', 'outPF · tangential')]:
            g = T[T['cell'] == cell].groupby('TMS')['RadialPull'].agg(['mean', 'sem']).reindex(G)
            ax.errorbar(range(len(G)), g['mean'], yerr=g['sem'], color=col, ls=ls, marker='o', capsize=3, lw=2, label=lab)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(range(len(G))); ax.set_xticklabels([P.TMS_TICK[t] for t in G])
        ax.set_title(STUDIES[study]['label'], fontsize=11)
        ax.set_ylabel('Pull toward the radial axis (°)\n+ = report drawn toward radial')
        sns.despine(ax=ax)
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Pull toward the radial axis (targets split by actual geometry); mean ± SEM over subjects', y=1.02)
    P.save(fig, OUT, 'F1_RadialPull_byLocation')


radial_pull_figure([('orig', T1), ('rep', T2)])
