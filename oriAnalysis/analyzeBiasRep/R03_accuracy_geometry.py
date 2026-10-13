"""
R03_accuracy_geometry.py

Accuracy as a function of each target's actual geometry relative to fixation,
in Study 1 and the bias replication. Every trial is placed by its recorded
screen position: phi = angle of the target orientation from the line joining
fixation and the stimulus (0 = radial, 90 = tangential). Each subject's
stimulus sits at a different polar angle, so the fixed orientation sets land at
different distances from the radial axis.

Figures (in COMPARE_DIR):
  G1_StimulusGeometry      where each subject's orientation categories fall
                           relative to the radial axis (paper labels in colour)
  G2_ErrorDistributions    signed-error distributions, radial vs tangential by
                           geometry (closer to the radial vs tangential axis),
                           inPF vs outPF, per TMS condition
  G3_AccuracyVsGeometry    absolute error vs distance from the radial axis,
                           inPF vs outPF, per dataset and pooled (None, Middle)
  G4_Anisotropy            per-subject anisotropy a from the mixture model with
                           kappa = exp(b + a*cos2phi + c*cos4theta) (cardinal
                           axes controlled), per dataset and pooled
Model (Stats_GeometryMixedModel.csv): trial-level linear mixed model of
absolute error on cos(2*phi) (+1 radial, -1 tangential) with cardinal and probe-
offset covariates and per-subject random intercepts and radial slopes, fitted to
each dataset and to both together; terms are tested by likelihood ratio.

Usage: python R03_accuracy_geometry.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

from biasrep_common import COMPARE_DIR, run_permutation_test
import panels as P
from study_results import compute, load_study, ERR_GRID

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
np.random.seed(0)
OUT = COMPARE_DIR

R1, R2 = compute('orig'), compute('rep')
T1, T2 = f"Study 1 (N={R1['N']})", f"Replication (N={R2['N']})"
POOL_GROUPS = ['No TMS', 'Middle TMS']
TP = f"Both datasets pooled (N={R1['N'] + R2['N']}; None and Middle)"
COL_R, COL_T = '#1F4E9A', '#8FB3E3'


def pooled(key):
    """Concatenate a per-subject table from both datasets, keeping None and Middle."""
    a = R1[key].assign(subjID='S1_' + R1[key]['subjID'].astype(str))
    b = R2[key].assign(subjID='R_' + R2[key]['subjID'].astype(str))
    out = pd.concat([a, b], ignore_index=True)
    return out[out['TMS'].isin(POOL_GROUPS)] if 'TMS' in out else out


def letter(ax, s):
    ax.set_title(s, loc='left', fontweight='bold', fontsize=13)


# ---------------------------------------------------------------------------
# G1: stimulus geometry per subject
# ---------------------------------------------------------------------------
print('G1 stimulus geometry...')


def circ_segments(phi):
    """Range of angles on the 180-deg orientation circle as plotting segments in
    [-90, 90), split in two where the range wraps past +-90. Returns (segments, centre)."""
    z = np.exp(1j * np.radians(2 * phi)).mean()
    centre = np.degrees(np.angle(z)) / 2
    off = (phi - centre + 90) % 180 - 90
    lo, hi = centre + off.min(), centre + off.max()
    if lo < -90: return [(lo + 180, 90), (-90, hi)], centre
    if hi >= 90: return [(lo, 90), (-90, hi - 180)], centre
    return [(lo, hi)], centre


fig, axs = plt.subplots(1, 2, figsize=(15, 11), gridspec_kw={'width_ratios': [R1['N'], R2['N'] + 4]})
for ax, st, R, title in [(axs[0], 'orig', R1, T1), (axs[1], 'rep', R2, T2)]:
    d = load_study(st)
    d = d[d['Hemifield'] == 'inPF']   # outPF is the mirror image: identical in this frame
    order = d.groupby('subjID')['Elevation'].first().sort_values()
    for y, (s, elev) in enumerate(order.items()):
        for (ori, dtype), g in d[d['subjID'] == s].groupby(['OriCat', 'DiagType']):
            segs, centre = circ_segments(g['PhiAligned'].values)
            for lo, hi in segs:
                ax.plot([lo, hi], [y, y], lw=6, solid_capstyle='butt', color=COL_R if dtype == 'Radial' else COL_T)
            ax.text(centre, y + 0.32, f'{ori:.0f}', ha='center', fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'sub{s:02d}  {e:.0f}°' for s, e in order.items()], fontsize=8)
    for x in (-45, 45): ax.axvline(x, color='0.5', ls=':', lw=1)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlim(-90, 90); ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_xlabel('Signed angle of target from the radial axis (°)\n0 = radial, ±90 = tangential; + = rotated toward vertical')
    ax.set_title(f'{title}\nrows: subjects by stimulus angle below the horizontal meridian', fontsize=10)
    sns.despine(ax=ax, left=True)
axs[0].plot([], [], lw=6, color=COL_R, label='labelled Radial (paper rule)')
axs[0].plot([], [], lw=6, color=COL_T, label='labelled Tangential')
axs[0].legend(loc='lower left', fontsize=8, frameon=True)
fig.suptitle('Where the orientation categories fall relative to each stimulus\'s radial axis (bars = range of targets incl. jitter; numbers = category)', y=0.98)
plt.tight_layout()
P.save(fig, OUT, 'G1_StimulusGeometry')

# ---------------------------------------------------------------------------
# G2: error distributions, radial vs tangential (by geometry) x inPF/outPF x TMS
# ---------------------------------------------------------------------------
print('G2 error distributions...')
rows = [(R1, t, T1) for t in R1['groups']] + [(R2, t, T2) for t in R2['groups']]
fig, axs = plt.subplots(len(rows), 2, figsize=(11, 3.3 * len(rows)), sharex=True, sharey=True)
dist_stats = []
for r, (R, t, title) in enumerate(rows):
    for c, h in enumerate(['inPF', 'outPF']):
        ax = axs[r, c]
        dd = R['err_density']
        cells = R['geo_cells'][(R['geo_cells']['TMS'] == t) & (R['geo_cells']['Hemifield'] == h)]
        for gt, col in [('Radial', COL_R), ('Tangential', COL_T)]:
            m = np.stack(dd[(dd['TMS'] == t) & (dd['Hemifield'] == h) & (dd['GeoType'] == gt)]['density'].values)
            mu, se = m.mean(0), m.std(0, ddof=1) / np.sqrt(len(m))
            ax.plot(ERR_GRID, mu, color=col, lw=2, label=gt)
            ax.fill_between(ERR_GRID, mu - se, mu + se, color=col, alpha=0.25)
        w = cells.pivot(index='subjID', columns='GeoType', values='AbsError').dropna()
        k = cells.pivot(index='subjID', columns='GeoType', values='Precision').dropna()
        de, dk = (w['Tangential'] - w['Radial']).values, (k['Radial'] - k['Tangential']).values
        pe, pk = run_permutation_test(de), run_permutation_test(dk)
        dist_stats.append({'Study': R['label'], 'TMS': t, 'Hemifield': h, 'N': len(w),
                           'AbsErr_Radial': w['Radial'].mean(), 'AbsErr_Tangential': w['Tangential'].mean(),
                           'AbsErr_T_minus_R': de.mean(), 'AbsErr_SEM': stats.sem(de), 'AbsErr_p': pe,
                           'Kappa_Radial': k['Radial'].mean(), 'Kappa_Tangential': k['Tangential'].mean(),
                           'Kappa_R_minus_T': dk.mean(), 'Kappa_SEM': stats.sem(dk), 'Kappa_p': pk})
        ax.text(0.98, 0.95, f"|err| R {w['Radial'].mean():.2f}° vs T {w['Tangential'].mean():.2f}°, {P.fmt_p(pe)}\n"
                            f"κ R {k['Radial'].mean():.1f} vs T {k['Tangential'].mean():.1f}, {P.fmt_p(pk)}",
                transform=ax.transAxes, ha='right', va='top', fontsize=7)
        ax.axvline(0, color='k', lw=0.6, ls=':')
        ax.set_title(f'{title} · {P.TMS_SHORT[t]} · {h}', fontsize=9)
        if r == len(rows) - 1: ax.set_xlabel('Signed error (°)')
        if c == 0: ax.set_ylabel('Density')
        sns.despine(ax=ax)
axs[0, 0].legend(fontsize=8, frameon=False, loc='upper left', title='by geometry', title_fontsize=7)
fig.suptitle('Error distributions: targets closer to the radial vs the tangential axis (recorded stimulus positions)\n'
             'mean ± SEM of per-subject densities; p = paired sign-flip permutation across subjects', y=1.0)
plt.tight_layout()
P.save(fig, OUT, 'G2_ErrorDistributions')
pd.DataFrame(dist_stats).to_csv(os.path.join(OUT, 'Stats_ErrorDistributions.csv'), index=False)

# ---------------------------------------------------------------------------
# G3: accuracy vs distance from the radial axis
# ---------------------------------------------------------------------------
print('G3 accuracy vs geometry...')
fig, axs = plt.subplots(3, 2, figsize=(12, 15), sharey=True)
for r, (curve, groups, title) in enumerate([(R1['curve_err_hemi'], R1['groups'], T1), (R2['curve_err_hemi'], R2['groups'], T2),
                                            (pooled('curve_err_hemi'), POOL_GROUPS, TP)]):
    for c, h in enumerate(['inPF', 'outPF']):
        ax = axs[r, c]
        P.binned_curve(ax, curve[curve['Hemifield'] == h], 'AbsError', groups, 'Absolute error (°)')
        n = curve[curve['Hemifield'] == h].groupby(['TMS', 'mid'])['subjID'].nunique().groupby('mid').min()
        for x, k in n.items():
            ax.text(x, 0.02, f'n={k}', transform=ax.get_xaxis_transform(), ha='center', fontsize=6, color='0.4')
        ax.set_title(f'{title} · {h}', fontsize=10); ax.set_ylim(4, 10); ax.set_xticks(range(0, 91, 15))
fig.suptitle('Accuracy vs distance of the target from the radial axis (15° bins; mean ± SEM over subjects with data in the bin)', y=0.995)
plt.tight_layout()
P.save(fig, OUT, 'G3_AccuracyVsGeometry')

# ---------------------------------------------------------------------------
# G4: anisotropy (mixture model, cardinal axes controlled)
# ---------------------------------------------------------------------------
print('G4 anisotropy...')
fig = plt.figure(figsize=(12, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.4)
an_stats = []
for r, (an, anh, groups, title) in enumerate([(R1['an'], R1['an_hemi'], R1['groups'], T1),
                                              (R2['an'], R2['an_hemi'], R2['groups'], T2),
                                              (pooled('an'), pooled('an_hemi'), POOL_GROUPS, TP)]):
    a0, a1 = fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])
    P.tms_paired(a0, an, 'a', groups, 'Anisotropy a\n(>0 = radial more precise)', ylim=(-0.8, 1.0))
    P.hemi_raw_boxes(a1, anh, 'a', groups, 'Anisotropy a', ylim=(-1.2, 1.5))
    a0.set_title(f'{title}\nboth hemifields', fontsize=10, pad=16); a1.set_title(f'{title}\ninPF vs outPF', fontsize=10)
    for t in groups:
        for h in ['both', 'inPF', 'outPF']:
            v = (an if h == 'both' else anh[anh['Hemifield'] == h])
            v = v[v['TMS'] == t]['a'].dropna().values
            an_stats.append({'Dataset': title, 'TMS': t, 'Hemifield': h, 'N': len(v), 'Mean_a': v.mean(), 'SEM': stats.sem(v),
                             'kappa_ratio_R_over_T': np.exp(2 * v.mean()), 'p_perm_vs0': run_permutation_test(v)})
fig.suptitle('Precision anisotropy from the mixture model: κ = exp(b + a·cos2φ + c·cos4θ)\n'
             'a > 0: radial targets more precise; κ_radial/κ_tangential = exp(2a); cardinal (oblique) effect absorbed by c', y=0.95)
P.save(fig, OUT, 'G4_Anisotropy')
pd.DataFrame(an_stats).to_csv(os.path.join(OUT, 'Stats_Anisotropy.csv'), index=False)

# ---------------------------------------------------------------------------
# Trial-level mixed model across datasets
# ---------------------------------------------------------------------------
print('Mixed models...')


def model_data(exclude=()):
    parts = []
    for st, lab in [('orig', 'Study1'), ('rep', 'Replication')]:
        d = load_study(st)
        d = d[~d['subjID'].isin(exclude)] if st == 'orig' else d
        parts.append(pd.DataFrame({
            'subj': lab + '_' + d['subjID'].astype(str), 'Study': lab, 'TMS': d['TMS'].values,
            'Hemifield': d['Hemifield'].values, 'AbsError': d['AbsError'].values,
            'Radial': np.cos(np.radians(2 * d['PhiRadial'].values)),
            'Cardinal': np.cos(np.radians(4 * d['targOriActual'].values)),
            'Off': d['AbsInitOffset'].values / 30}))
    return pd.concat(parts, ignore_index=True)


BASE = "AbsError ~ C(TMS, Treatment('No TMS')) + C(Hemifield, Treatment('outPF')) + Cardinal + Off + I(Off**2)"
STEPS = [('Radial (main effect)', ' + Radial'),
         ('Radial x TMS', " + Radial:C(TMS, Treatment('No TMS'))"),
         ('Radial x Hemifield', " + Radial:C(Hemifield, Treatment('outPF'))"),
         ('Radial x TMS x Hemifield', " + Radial:C(TMS, Treatment('No TMS')):C(Hemifield, Treatment('outPF'))")]


def fit(formula, d):
    """ML fit; tries several optimizers and keeps the highest likelihood, so
    nested likelihood-ratio tests are not distorted by a stalled optimizer."""
    best = None
    for meth in ('lbfgs', 'bfgs', 'powell'):
        try:
            m = smf.mixedlm(formula, d, groups=d['subj'], re_formula='~Radial').fit(reml=False, method=meth)
        except Exception:
            continue
        if best is None or m.llf > best.llf:
            best = m
    return best


def run_models(d, label):
    rows, coefs = [], []
    multi = d['Study'].nunique() > 1
    f = BASE + (' + C(Study)' if multi else '')
    prev = fit(f, d)
    for name, term in STEPS:
        f_new = f + term
        m = fit(f_new, d)
        lr = 2 * (m.llf - prev.llf)
        dfd = len(m.fe_params) - len(prev.fe_params)
        rows.append({'Data': label, 'Test': name, 'chi2': lr, 'df': dfd, 'p': stats.chi2.sf(lr, dfd)})
        for k, v in m.fe_params.items():
            if 'Radial' in k:
                coefs.append({'Data': label, 'Model': name, 'Term': k, 'Estimate': v, 'SE': m.bse[k], 'p': m.pvalues[k]})
        f, prev = f_new, m
    if multi:   # does the radial effect differ between datasets? (added to the Radial x Hemifield model)
        base3 = BASE + ' + C(Study)' + ''.join(t for _, t in STEPS[:3])
        m3, m5 = fit(base3, d), fit(base3 + ' + Radial:C(Study)', d)
        lr = 2 * (m5.llf - m3.llf)
        rows.append({'Data': label, 'Test': 'Radial x Study', 'chi2': lr, 'df': 1, 'p': stats.chi2.sf(lr, 1)})
        k = [k for k in m5.fe_params.index if k.startswith('Radial:C(Study)')][0]
        coefs.append({'Data': label, 'Model': 'Radial x Study', 'Term': k, 'Estimate': m5.fe_params[k], 'SE': m5.bse[k], 'p': m5.pvalues[k]})
    return rows, coefs


all_d = model_data()
lrt, coef = [], []
for d, label in [(all_d[all_d['Study'] == 'Study1'], 'Study 1'), (all_d[all_d['Study'] == 'Replication'], 'Replication'),
                 (all_d, 'Both datasets'), (model_data(exclude=(17,)), 'Both datasets, without sub17')]:
    print(f'  {label} ({d["subj"].nunique()} subjects, {len(d)} trials)')
    r_, c_ = run_models(d, label)
    lrt += r_; coef += c_
lrt, coef = pd.DataFrame(lrt), pd.DataFrame(coef)
lrt.to_csv(os.path.join(OUT, 'Stats_GeometryMixedModel_LRT.csv'), index=False)
coef.to_csv(os.path.join(OUT, 'Stats_GeometryMixedModel_Coefficients.csv'), index=False)

pd.set_option('display.width', 220)
print('\nLikelihood-ratio tests (each term added to the previous model):')
print(lrt.round(4).to_string(index=False))
print('\nRadial coefficients (deg of abs error per unit cos2phi; negative = radial more accurate):')
print(coef.round(4).to_string(index=False))
print('\nError distributions (paired tests, radial vs tangential by geometry):')
print(pd.DataFrame(dist_stats).round(3).to_string(index=False))
print('\nAnisotropy:')
print(pd.DataFrame(an_stats).round(3).to_string(index=False))
print(f'\nOutputs in {OUT}')
