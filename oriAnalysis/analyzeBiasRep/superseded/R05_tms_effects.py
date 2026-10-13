"""
R05_tms_effects.py

Screens for any effect of TMS beyond the radial index, in Study 1 (Early and
Middle vs None) and the replication (Middle vs None). Per subject x TMS
condition (both hemifields, and inPF / outPF separately):

  Memory     AbsError, CircSD (circular SD of errors), Kappa and GuessRate
             (von Mises + uniform mixture, modeller.fit_model)
  Biases     CardinalBias (mc: + = away from cardinals) and RadialBias (m:
             - = toward the radial axis) from the geometry mixture model;
             SerialBias: attraction of the error toward the previous trial's
             target (mean of error x sign(prev - current) for 0 < |diff| <= 45),
             after removing each subject's orientation-dependent bias
  Response   RT, RT_SD, TimeoutRate, LargeErrorRate (3-SD outliers), StartPull
             (slope of signed error on signed probe offset, |offset| < 30),
             WrongDirection (first dial movement away from the target, 15-30 deg
             offsets), ExcessTravel (dial path beyond the start-to-response
             distance), Reversals (direction changes per trial)

Contrasts (paired sign-flip permutation, 10,000; Cohen's dz):
  overall  TMS - None, both hemifields
  spatial  (inPF - outPF) under TMS minus (inPF - outPF) under None
FDR (Benjamini-Hochberg) across all metrics x contrasts within each study.
In the replication TMS condition is confounded with testing day (11/12
subjects); flagged effects are also shown split by session order.

Usage: python R05_tms_effects.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from biasrep_common import COMPARE_DIR, STUDIES, md, run_permutation_test
import panels as P
from study_results import compute, load_study

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
np.random.seed(0)
OUT = COMPARE_DIR
RESP_WINDOW = 3.0   # s (parameters.respDuration)


def wrap90(x):
    return (x + 90) % 180 - 90


def dial_features(arr, init_offset):
    """First-movement direction error, excess travel and reversals from the dial trace."""
    if len(arr) < 2:
        return np.nan, 0.0, 0
    d = wrap90(np.diff(arr))
    nz = d[d != 0]
    if len(nz) == 0:
        return np.nan, 0.0, 0
    # positive offset (start counter-clockwise of target) -> shortest path is clockwise = negative steps
    wrong = float(np.sign(nz[0]) == np.sign(init_offset)) if 15 <= abs(init_offset) <= 30 else np.nan
    excess = np.abs(d).sum() - abs(wrap90(arr[-1] - arr[0]))
    reversals = int(np.sum(np.diff(np.sign(nz)) != 0))
    return wrong, excess, reversals


def orientation_residuals(sub):
    """Errors minus each subject's orientation-dependent bias (Fourier terms of the
    target orientation up to 12*theta, which also captures attraction to the
    replication's 45/135 category centres)."""
    th = np.radians(sub['targOriActual'].values)
    X = np.column_stack([np.ones(len(th))] + [f(k * th) for k in (2, 4, 8, 12) for f in (np.sin, np.cos)])
    y = sub['errOri_scaled'].values
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def subject_metrics(study):
    df = load_study(study)
    raw = pd.read_pickle(STUDIES[study]['pkl'])
    raw['Hemifield'] = raw['stimPF_label']; raw['TMS'] = raw['TMScond_label']
    feats = df.apply(lambda r: dial_features(r['oriArray'], r['initOffset']), axis=1, result_type='expand')
    df[['WrongDir', 'Excess', 'Reversals']] = feats.values
    # serial dependence on orientation-bias residuals (per subject x TMS)
    df['resid'] = np.nan
    for _, idx in df.groupby(['subjID', 'TMS']).groups.items():
        df.loc[idx, 'resid'] = orientation_residuals(df.loc[idx])
    df['dPrev'] = wrap90(df['prevTarg'] - df['targOriActual'])

    an = compute(study)['an']; anh = compute(study)['an_hemi']
    rows = []
    for hemi in ['both', 'inPF', 'outPF']:
        d = df if hemi == 'both' else df[df['Hemifield'] == hemi]
        r_ = raw if hemi == 'both' else raw[raw['Hemifield'] == hemi]
        a_ = an.assign(Hemifield='both') if hemi == 'both' else anh[anh['Hemifield'] == hemi]
        for (s, t), sub in d.groupby(['subjID', 'TMS']):
            e = sub['errOri_scaled'].values
            z = np.exp(1j * np.radians(2 * e)).mean()
            g, k = md.fit_model(e, 'mixtureModel', max_trials=100)
            sp = sub[(sub['dPrev'].abs() > 0) & (sub['dPrev'].abs() <= 45)]
            near = sub[sub['AbsInitOffset'] < 30]
            rr = r_[(r_['subjID'] == s) & (r_['TMS'] == t)]
            am = a_[(a_['subjID'] == s) & (a_['TMS'] == t)]
            rows.append({'Study': STUDIES[study]['label'], 'subjID': s, 'TMS': t, 'Hemifield': hemi,
                         'AbsError': np.abs(e).mean(),
                         'CircSD': np.degrees(np.sqrt(-2 * np.log(np.abs(z)))) / 2,
                         'Kappa': k, 'GuessRate': g,
                         'CardinalBias': am['mc'].iloc[0] if len(am) else np.nan,
                         'RadialBias': am['m'].iloc[0] if len(am) else np.nan,
                         'SerialBias': np.mean(sp['resid'] * np.sign(sp['dPrev'])),
                         'RT': sub['RT'].mean(), 'RT_SD': sub['RT'].std(),
                         'TimeoutRate': (rr['RT'] >= RESP_WINDOW).mean(),
                         'LargeErrorRate': rr['LargeError'].mean(),
                         'StartPull': np.polyfit(near['initOffset'], near['errOri_scaled'], 1)[0],
                         'WrongDirection': sub['WrongDir'].mean(),
                         'ExcessTravel': sub['Excess'].mean(), 'Reversals': sub['Reversals'].mean()})
    return pd.DataFrame(rows), df


METRICS = ['AbsError', 'CircSD', 'Kappa', 'GuessRate', 'CardinalBias', 'RadialBias', 'SerialBias',
           'RT', 'RT_SD', 'TimeoutRate', 'LargeErrorRate', 'StartPull', 'WrongDirection', 'ExcessTravel', 'Reversals']


def bh_fdr(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def contrasts(M, groups):
    out = []
    for met in METRICS:
        w = M.pivot_table(index='subjID', columns=['Hemifield', 'TMS'], values=met)
        for t in groups[1:]:
            d_all = (w[('both', t)] - w[('both', 'No TMS')]).dropna().values
            d_sp = ((w[('inPF', t)] - w[('outPF', t)]) - (w[('inPF', 'No TMS')] - w[('outPF', 'No TMS')])).dropna().values
            for kind, d in [('overall', d_all), ('spatial', d_sp)]:
                out.append({'Metric': met, 'Contrast': f'{P.TMS_SHORT[t].split()[0]} − None', 'Kind': kind, 'N': len(d),
                            'Mean': d.mean(), 'SEM': stats.sem(d), 'dz': d.mean() / d.std(ddof=1),
                            'p': run_permutation_test(d)})
    out = pd.DataFrame(out)
    out['q_FDR'] = bh_fdr(out['p'].values)
    return out


print('Computing metrics...')
M1, D1 = subject_metrics('orig')
M2, D2 = subject_metrics('rep')
C1, C2 = contrasts(M1, STUDIES['orig']['groups']), contrasts(M2, STUDIES['rep']['groups'])
C1['Study'], C2['Study'] = 'Study 1', 'Replication'
C = pd.concat([C1, C2], ignore_index=True)
C.to_csv(os.path.join(OUT, 'Stats_TMSEffects.csv'), index=False)
pd.concat([M1, M2]).to_csv(os.path.join(OUT, 'SubjectValues_TMSEffects.csv'), index=False)

# Replication: session-order split for every metric (TMS is confounded with day)
d1 = D2[D2['day'] == 1].groupby('subjID')['TMS'].agg(lambda s: s.iloc[0] if s.nunique() == 1 else 'mixed')
order = d1.map({'No TMS': 'NoTMS first', 'Middle TMS': 'TMS first', 'mixed': 'mixed'})
w2 = M2[M2['Hemifield'] == 'both'].pivot_table(index='subjID', columns='TMS', values=METRICS)
split = []
for met in METRICS:
    dd = (w2[(met, 'Middle TMS')] - w2[(met, 'No TMS')])
    sign = order.map({'NoTMS first': 1, 'TMS first': -1, 'mixed': 0})
    X = np.column_stack([np.ones(len(dd)), sign.loc[dd.index].values])
    b, res, *_ = np.linalg.lstsq(X, dd.values, rcond=None)
    split.append({'Metric': met, 'NoTMS_first_mean': dd[order == 'NoTMS first'].mean(),
                  'TMS_first_mean': dd[order == 'TMS first'].mean(), 'TMS_effect_day_adjusted': b[0], 'Day2_minus_Day1': b[1]})
split = pd.DataFrame(split)
split.to_csv(os.path.join(OUT, 'Stats_TMSEffects_ReplicationOrderSplit.csv'), index=False)

# ---------------------------------------------------------------------------
# Figure: effect atlas (dz with 95% CI)
# ---------------------------------------------------------------------------
cols = [('Study 1', 'Early − None', P.TMS_LINE_PAL['Early TMS']), ('Study 1', 'Middle − None', P.TMS_LINE_PAL['Middle TMS']),
        ('Replication', 'Middle − None', '#0B5394')]
fig, axs = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
for ax, kind, ttl in [(axs[0], 'overall', 'Overall (TMS − None)'), (axs[1], 'spatial', 'Spatially specific\n(inPF−outPF under TMS) − (inPF−outPF under None)')]:
    for j, (st, con, col) in enumerate(cols):
        sub = C[(C['Study'] == st) & (C['Contrast'] == con) & (C['Kind'] == kind)].set_index('Metric').reindex(METRICS)
        y = np.arange(len(METRICS))[::-1] + (1 - j) * 0.22
        ci = stats.t.ppf(.975, sub['N'] - 1) / np.sqrt(sub['N'])
        ax.errorbar(sub['dz'], y, xerr=ci, fmt='none', ecolor=col, alpha=0.7, capsize=2)
        sig = sub['q_FDR'] < 0.05
        ax.scatter(sub['dz'][~sig], y[~sig.values], facecolor='white', edgecolor=col, s=35, zorder=3,
                   label=f'{st}: {con}' if kind == 'overall' else None)
        ax.scatter(sub['dz'][sig], y[sig.values], color=col, s=45, zorder=4)
        for yy, (m, r) in zip(y, sub.iterrows()):
            if r['p'] < 0.05:
                ax.text(1.55, yy, f"{'*' if r['q_FDR'] < 0.05 else '·'} p={r['p']:.3f}", fontsize=6, color=col, va='center')
    ax.axvline(0, color='k', lw=1)
    ax.set_xlim(-1.6, 1.9); ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("Effect size dz (± 95% CI); filled = FDR q < .05")
    sns.despine(ax=ax, left=True)
axs[0].set_yticks(np.arange(len(METRICS))[::-1]); axs[0].set_yticklabels(METRICS)
axs[0].legend(fontsize=8, frameon=False, loc='lower left')
fig.suptitle('Screen for TMS effects beyond radial sensitivity (per-subject paired contrasts; p listed when uncorrected p < .05)', y=0.98)
plt.tight_layout()
P.save(fig, OUT, 'T1_TMSEffectAtlas')

pd.set_option('display.width', 220)
show = C.copy()
for c in ['Mean', 'SEM', 'dz']: show[c] = show[c].map(lambda x: f'{x:+.4f}')
print(show[show['p'] < 0.1].sort_values(['Kind', 'p']).to_string(index=False))
print('\nReplication, session-order split (TMS − None):')
print(split.round(4).to_string(index=False))
