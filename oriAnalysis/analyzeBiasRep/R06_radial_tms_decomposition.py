"""
R06_radial_tms_decomposition.py

Where does the "TMS reduces radial sensitivity" result come from? Estimates the
radial advantage separately for each TMS condition, and the None - TMS
difference, under each analysis choice:

  Binary index    the paper's (R-T)/(R+T) of per-subject means (category labels),
                  all trials / |probe offset| < 45 deg (paper Fig 5) / < 30 deg
  Mixed model     trial-level absolute error with a separate radial slope per
                  TMS condition, random intercept + radial slope per subject,
                  cardinal and probe-offset covariates;
                    'category'   Radial = +1 / -1 from the paper's labels
                    'geometry'   Radial = cos(2*phi) from the recorded position
                  all trials / < 45 deg
For the mixed models, 'TMS-dependence' is the likelihood-ratio test of separate
radial slopes per TMS condition against one common slope.

Usage: python R06_radial_tms_decomposition.py
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
import statsmodels.formula.api as smf

from biasrep_common import COMPARE_DIR, STUDIES
import panels as P
from study_results import compute, load_study

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
OUT = COMPARE_DIR


def fit(formula, d):
    best = None
    for meth in ('lbfgs', 'bfgs', 'powell'):
        try:
            m = smf.mixedlm(formula, d, groups=d['subjID'], re_formula='~Radial').fit(reml=False, method=meth)
        except Exception:
            continue
        if best is None or m.llf > best.llf:
            best = m
    return best


def mixed_variant(d, groups, radial):
    d = d.copy()
    d['Radial'] = (np.where(d['DiagType'] == 'Radial', 1.0, -1.0) if radial == 'category'
                   else np.cos(np.radians(2 * d['PhiRadial'])))
    d['Cardinal'] = np.cos(np.radians(4 * d['targOriActual']))
    d['Off'] = d['AbsInitOffset'] / 30
    base = "AbsError ~ C(TMS, Treatment('No TMS')) + C(Hemifield) + Cardinal + Off + I(Off**2)"
    common = fit(base + ' + Radial', d)
    sep = fit(base + " + Radial + Radial:C(TMS, Treatment('No TMS'))", d)
    lr = 2 * (sep.llf - common.llf)
    out = {'TMS-dependence chi2': lr, 'df': len(groups) - 1, 'TMS-dependence p': stats.chi2.sf(lr, len(groups) - 1)}
    rows = []
    for t in groups:
        if t == 'No TMS':
            est, se = sep.fe_params['Radial'], sep.bse['Radial']
            diff_p = np.nan
        else:
            k = f"Radial:C(TMS, Treatment('No TMS'))[T.{t}]"
            cov = sep.cov_params()
            est = sep.fe_params['Radial'] + sep.fe_params[k]
            se = np.sqrt(cov.loc['Radial', 'Radial'] + cov.loc[k, k] + 2 * cov.loc['Radial', k])
            diff_p = sep.pvalues[k]
        rows.append({'TMS': t, 'Estimate': est, 'SE': se, 'p_vs0': 2 * stats.norm.sf(abs(est / se)), 'p_None_minus_TMS': diff_p})
    return rows, out


records, tests = [], []
for study in ['orig', 'rep']:
    R = compute(study)
    G = STUDIES[study]['groups']
    lab = STUDIES[study]['label']
    st = R['stats']
    # Binary index (paper) from the cached contrasts
    for sub in (['all', '<45', '<30'] if study == 'orig' else ['all']):
        s = st[(st['Metric'] == 'Abs Error') & (st['Contrast'] == 'Radial-Tangential') & (st['Subset'] == sub)]
        for t in G:
            r = s[s['TMS'] == t].iloc[0]
            dp = s[s['TMS'] == f'No TMS minus {t}']['p_perm']
            records.append({'Study': lab, 'Analysis': f'Binary index · {sub}', 'Kind': 'index', 'TMS': t,
                            'Estimate': r['CI_mean'], 'SE': r['CI_sem'], 'p_vs0': r['p_perm'],
                            'p_None_minus_TMS': dp.iloc[0] if len(dp) else np.nan})
    # Mixed models
    d = load_study(study)
    for radial in ['category', 'geometry']:
        for sub, dd in [('all', d)] + ([('<45', d[d['AbsInitOffset'] < 45])] if study == 'orig' else []):
            print(f'  {lab}: mixed model, {radial}, {sub}')
            rows, test = mixed_variant(dd, G, radial)
            name = f'Mixed model · {radial} · {sub}'
            records += [{'Study': lab, 'Analysis': name, 'Kind': 'deg', **r} for r in rows]
            tests.append({'Study': lab, 'Analysis': name, **test})

rec, tests = pd.DataFrame(records), pd.DataFrame(tests)
rec.to_csv(os.path.join(OUT, 'Stats_RadialByTMS_Decomposition.csv'), index=False)
tests.to_csv(os.path.join(OUT, 'Stats_RadialByTMS_TMSdependence.csv'), index=False)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1]})
for ax, kind, xlab in [(axs[0], 'index', 'Radial index (R−T)/(R+T), abs error\n(negative = radial more accurate)'),
                       (axs[1], 'deg', 'Radial slope (° abs error per unit; negative = radial more accurate)')]:
    sub = rec[rec['Kind'] == kind]
    labels, y = [], 0
    for (study, analysis), g in sub.groupby(['Study', 'Analysis'], sort=False):
        for _, r in g.iloc[::-1].iterrows():
            ax.errorbar(r['Estimate'], y, xerr=1.96 * r['SE'], fmt='o', color=P.TMS_LINE_PAL[r['TMS']], ms=6, capsize=2)
            note = '' if np.isnan(r['p_None_minus_TMS']) else f"   None−this p={r['p_None_minus_TMS']:.3f}"
            ax.text(1.01, y, f"{P.TMS_SHORT[r['TMS']]}: p vs 0 = {r['p_vs0']:.3f}{note}",
                    transform=ax.get_yaxis_transform(), va='center', fontsize=7)
            y += 1
        t = tests[(tests['Study'] == study) & (tests['Analysis'] == analysis)]
        lab = f'{study} · {analysis}' + (f"\nTMS-dependence χ²({int(t['df'].iloc[0])}) = {t['TMS-dependence chi2'].iloc[0]:.2f}, p = {t['TMS-dependence p'].iloc[0]:.3f}" if len(t) else '')
        labels.append((y - len(g) / 2 - 0.5, lab))
        y += 0.8
    ax.set_yticks([p for p, _ in labels]); ax.set_yticklabels([l for _, l in labels], fontsize=8)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel(xlab)
    ax.invert_yaxis()
    sns.despine(ax=ax)
fig.suptitle('Radial advantage per TMS condition under each analysis choice (95% CI)', y=0.99)
plt.tight_layout()
P.save(fig, OUT, 'T2_RadialByTMS_Decomposition')

pd.set_option('display.width', 220)
print(rec.round(4).to_string(index=False))
print(tests.round(4).to_string(index=False))
