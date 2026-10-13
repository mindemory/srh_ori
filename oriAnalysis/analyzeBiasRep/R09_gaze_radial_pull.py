"""
R09_gaze_radial_pull.py

Can gaze behaviour explain the radial-pull effect (R08)? Uses the gaze epochs
built by gaze_epochs.py (iEye-preprocessed X/Y in deg, +y up, 10-ms bins,
-500..4250 ms around grating onset), read only.

Per trial, relative to the pre-stimulus baseline (-300..0 ms):
  toward    gaze displacement along the fixation -> stimulus direction (deg;
            + = toward the remembered location), late delay 2700-4200 ms
            (after the middle-delay pulse window); also early delay 700-2200 ms
  axial     r^2 - t^2 of that displacement (deg^2; + = gaze displaced along the
            radial line, - = across it), the axial feature that carries
            orientation information in the decoders
  mag       displacement magnitude (deg)
  maxdev    largest deviation from baseline between 0 and 4250 ms (deg)
  sample_missing  fraction of missing samples while the grating was shown

Tests
  1. Does TMS change gaze the way it changes pull (R - T difference at inPF)?
  2. Does gaze predict pull trial by trial (within-subject regressions)?
  3. Does the TMS effect on pull survive gaze covariates (trial-level mixed model)?
  4. Does it survive excluding fixation breaks (maxdev > 2, 1.5, 1 deg) and
     trials with missing data during the grating?
  5. Across subjects, does the TMS change in pull track the TMS change in gaze?

Outputs: COMPARE_DIR/radial_pull/G1_GazeAndPull.png, Stats_GazePull_*.csv

Usage: python R09_gaze_radial_pull.py
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

from biasrep_common import COMPARE_DIR, STUDIES, ORIG_ROOT, DATA_ROOT
import panels as P
from study_results import load_study

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
OUT = os.path.join(COMPARE_DIR, 'radial_pull')
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
EPOCHS = {'orig': os.path.join(ORIG_ROOT, 'analysis', 'gaze_epochs_orig.npz'),
          'rep': os.path.join(DATA_ROOT, 'analysis', 'gaze_epochs_rep.npz')}
BASE, EARLY, LATE, SAMPLE = (-300, 0), (700, 2200), (2700, 4200), (0, 250)


def perm_p(v, n=20000):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    s = RNG.choice([-1.0, 1.0], size=(n, len(v)))
    return max(np.mean(np.abs((s * v).mean(1)) >= abs(v.mean())), 1 / n)


def summarize(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return dict(N=len(v), Mean=v.mean(), SEM=stats.sem(v), d=v.mean() / v.std(), p=perm_p(v))


def gaze_features(study):
    z = np.load(EPOCHS[study], allow_pickle=True)
    t, X, Y = z['times'], z['X'], z['Y']
    win = lambda a, b: (t >= a) & (t < b)
    with np.errstate(invalid='ignore', divide='ignore'):
        bx, by = np.nanmean(X[:, win(*BASE)], 1), np.nanmean(Y[:, win(*BASE)], 1)
        f = pd.DataFrame({k[5:]: z[k] for k in z.files if k.startswith('meta_')})
        for name, (a, b) in [('late', LATE), ('early', EARLY)]:
            f[f'dx_{name}'] = np.nanmean(X[:, win(a, b)], 1) - bx
            f[f'dy_{name}'] = np.nanmean(Y[:, win(a, b)], 1) - by
        dev = np.hypot(X[:, win(0, 4250)] - bx[:, None], Y[:, win(0, 4250)] - by[:, None])
        f['maxdev'] = np.nanmax(dev, 1)
        f['sample_missing'] = np.mean(np.isnan(X[:, win(*SAMPLE)]), 1)
    return f


def trials(study):
    d = load_study(study)
    d['trial'] = d['trialInBlock'] + 1
    d = d.merge(gaze_features(study), on=['subjID', 'day', 'block', 'trial'], how='left')
    d['pull'] = -d['errOri_scaled'] * np.sign(np.sin(np.radians(2 * d['PhiRadial'])))
    ux, uy = (d['targX'] - d['xCenter']), -(d['targY'] - d['yCenter'])
    n = np.hypot(ux, uy); ux, uy = ux / n, uy / n
    for w in ['late', 'early']:
        r = d[f'dx_{w}'] * ux + d[f'dy_{w}'] * uy
        tg = -d[f'dx_{w}'] * uy + d[f'dy_{w}'] * ux
        d[f'toward_{w}'], d[f'axial_{w}'], d[f'mag_{w}'] = r, r ** 2 - tg ** 2, np.hypot(r, tg)
    d['Study'] = STUDIES[study]['label']
    d['sid'] = d['Study'] + '_' + d['subjID'].astype(str)
    return d


D = pd.concat([trials('orig'), trials('rep')], ignore_index=True)
S1 = sorted(D.loc[D['Study'] == 'Study 1', 'sid'].unique())
REP = sorted(D.loc[D['Study'] == 'Replication', 'sid'].unique())
CONTRASTS = [('Study 1 · Early', 'Early TMS', S1), ('Study 1 · Middle', 'Middle TMS', S1),
             ('Replication · Middle', 'Middle TMS', REP), ('Both datasets · Middle', 'Middle TMS', S1 + REP)]


def cell_means(d, value):
    return (d.groupby(['sid', 'TMS', 'Hemifield', 'GeoType'])[value].mean()
            .unstack('GeoType').rename(columns={'Radial': 'R', 'Tangential': 'T'}).reset_index())


def effect(cm, tms, loc, sids, part='RT'):
    c = cm[cm['Hemifield'] == loc].copy()
    c['v'] = c['R'] - c['T'] if part == 'RT' else c[part]
    w = c.pivot(index='sid', columns='TMS', values='v')
    e = (w[tms] - w['No TMS']).dropna()
    return e[e.index.isin(sids)]


rows = []
# ---------------------------------------------------------------------------
# 1. TMS change in gaze, mirroring the pull analysis
# ---------------------------------------------------------------------------
for value, label in [('toward_late', 'gaze toward stimulus, late delay (deg)'), ('toward_early', 'gaze toward stimulus, early delay (deg)'),
                     ('axial_late', 'gaze along radial line, late delay (deg²)'), ('mag_late', 'gaze displacement, late delay (deg)'),
                     ('pull', 'report pull toward radial axis (deg)')]:
    cm = cell_means(D.dropna(subset=[value]), value)
    for name, tms, sids in CONTRASTS:
        for loc in ['inPF', 'outPF']:
            for part, plab in [('RT', 'radial − tangential'), ('T', 'tangential targets'), ('R', 'radial targets')]:
                rows.append({'Test': '1. TMS change', 'Measure': label, 'Contrast': name, 'Location': loc, 'Part': plab,
                             **summarize(effect(cm, tms, loc, sids, part))})

# ---------------------------------------------------------------------------
# 2. Trial-level association of gaze with pull (within subject, standardized predictors)
# ---------------------------------------------------------------------------
assoc = []
for sid, g in D.dropna(subset=['toward_late', 'axial_late', 'mag_late', 'pull']).groupby('sid'):
    for subset, gg in [('all targets', g), ('tangential targets', g[g['GeoType'] == 'Tangential']), ('radial targets', g[g['GeoType'] == 'Radial'])]:
        if len(gg) < 30: continue
        Xm = gg[['toward_late', 'axial_late', 'mag_late']].apply(lambda c: (c - c.mean()) / c.std())
        Xm.insert(0, 'const', 1.0)
        b = np.linalg.lstsq(Xm.values, gg['pull'].values, rcond=None)[0]
        assoc.append({'sid': sid, 'Study': gg['Study'].iloc[0], 'subset': subset, 'toward': b[1], 'axial': b[2], 'mag': b[3]})
assoc = pd.DataFrame(assoc)
for subset, g in assoc.groupby('subset'):
    for k in ['toward', 'axial', 'mag']:
        rows.append({'Test': '2. Trial-level association', 'Measure': f'pull per 1 SD of gaze {k}', 'Contrast': 'Both datasets, all conditions',
                     'Location': 'both', 'Part': subset, **summarize(g[k].values)})

# ---------------------------------------------------------------------------
# 3. Mixed model at inPF with and without gaze covariates (same trials)
# ---------------------------------------------------------------------------
def mixed(dd, covars):
    f = 'pull ~ Rad * TMSon' + (' + C(Study)' if dd['Study'].nunique() > 1 else '') + ''.join(f' + {c}' for c in covars)
    best = None
    for meth in ('lbfgs', 'bfgs', 'powell'):
        try:
            m = smf.mixedlm(f, dd, groups=dd['sid'], re_formula='~Rad').fit(reml=True, method=meth)
            if best is None or m.llf > best.llf: best = m
        except Exception:
            pass
    return best.fe_params['Rad:TMSon'], best.bse['Rad:TMSon'], best.pvalues['Rad:TMSon']


inpf = D[(D['Hemifield'] == 'inPF')].dropna(subset=['toward_late', 'axial_late', 'mag_late', 'toward_early']).copy()
inpf['Rad'] = (inpf['GeoType'] == 'Radial').astype(float) - 0.5
for c in ['toward_late', 'axial_late', 'mag_late', 'toward_early']:
    inpf[c + '_z'] = inpf.groupby('sid')[c].transform(lambda x: (x - x.mean()) / x.std())
COV = ['toward_late_z', 'axial_late_z', 'mag_late_z', 'toward_early_z']
COV_INT = COV + [f'Rad:{c}' for c in COV]
for name, tms, sids in CONTRASTS:
    dd = inpf[inpf['sid'].isin(sids) & inpf['TMS'].isin(['No TMS', tms])].copy()
    dd['TMSon'] = (dd['TMS'] == tms).astype(float)
    for lab, cov in [('no gaze covariates', []), ('+ gaze covariates', COV), ('+ gaze covariates × target class', COV_INT)]:
        est, se, p = mixed(dd, cov)
        rows.append({'Test': '3. Mixed model, inPF', 'Measure': 'Rad × TMS (deg)', 'Contrast': name, 'Location': 'inPF', 'Part': lab,
                     'N': dd['sid'].nunique(), 'Mean': est, 'SEM': se, 'd': np.nan, 'p': p})

# ---------------------------------------------------------------------------
# 4. Excluding fixation breaks / missing data during the grating
# ---------------------------------------------------------------------------
excl = []
for thr in [None, 2.0, 1.5, 1.0]:
    keep = D['maxdev'].notna() & (D['sample_missing'] <= 0.5)
    if thr is not None: keep &= D['maxdev'] <= thr
    lab = 'gaze available, grating seen' if thr is None else f'+ no deviation > {thr:g}° during the trial'
    dk = D[keep]
    for st in ['Study 1', 'Replication']:
        excl.append({'criterion': lab, 'Study': st, 'kept_fraction': keep[D['Study'] == st].mean()})
    cm = cell_means(dk, 'pull')
    for name, tms, sids in CONTRASTS:
        rows.append({'Test': '4. Fixation exclusion', 'Measure': 'change in R − T pull (deg)', 'Contrast': name, 'Location': 'inPF', 'Part': lab,
                     **summarize(effect(cm, tms, 'inPF', sids))})
excl = pd.DataFrame(excl)

# ---------------------------------------------------------------------------
# 5. Across subjects: TMS change in pull vs TMS change in gaze (both datasets, Middle, inPF)
# ---------------------------------------------------------------------------
e_pull = effect(cell_means(D, 'pull'), 'Middle TMS', 'inPF', S1 + REP)
across = {}
for value in ['toward_late', 'axial_late']:
    e_g = effect(cell_means(D.dropna(subset=[value]), value), 'Middle TMS', 'inPF', S1 + REP)
    j = pd.concat([e_pull.rename('pull'), e_g.rename('gaze')], axis=1).dropna()
    r, p = stats.spearmanr(j['pull'], j['gaze'])
    across[value] = (j, r, p)
    rows.append({'Test': '5. Across subjects', 'Measure': f'Spearman ρ: TMS change in pull vs in {value}', 'Contrast': 'Both datasets · Middle',
                 'Location': 'inPF', 'Part': 'radial − tangential', 'N': len(j), 'Mean': r, 'SEM': np.nan, 'd': np.nan, 'p': p})

# ---------------------------------------------------------------------------
# 6. TMS shift in the gaze orientation code (gaze_timeresolved.py decoders) and
#    whether it tracks the change in pull across subjects
# ---------------------------------------------------------------------------
GAZE_DIR = os.path.join(COMPARE_DIR, 'gaze')
DECODERS = {'mahal_target': 'Mahalanobis', 'tafkap_target': 'TAFKAP', 'ridge_target': 'ridge'}
GZ_WINDOWS = {'late delay (2.4–4.2 s)': (2600, 4000), 'cluster window (2.9–3.7 s)': (3100, 3500)}   # window centres; windows span ±200 ms


def gaze_code_shift(study, decoder, centres_range):
    with open(os.path.join(GAZE_DIR, f'timeresolved_{study}.pkl'), 'rb') as fh:
        T = pickle.load(fh)
    cen = np.asarray(T['centres']); wsel = (cen >= centres_range[0]) & (cen <= centres_range[1])
    tr = T['trials'].reset_index(drop=True).copy()
    tr['fid'] = np.nan
    for s, res in T['results'].items():
        v = res['standard']
        with np.errstate(invalid='ignore'):
            f = np.nanmean((v['real'][decoder] - v['null'][decoder])[:, wsel].astype(float), axis=1)
        tr.loc[res['rows'], 'fid'] = f
    tr['sid'] = STUDIES[study]['label'] + '_' + tr['subjID'].astype(str)
    m = tr.groupby(['sid', 'TMS', 'Hemifield'])['fid'].mean().unstack('Hemifield')
    io = (m['inPF'] - m['outPF']).unstack('TMS')
    return {t: (io[t] - io['No TMS']).dropna() for t in io.columns if t != 'No TMS'}, io


e_pull_all = {name: effect(cell_means(D, 'pull'), tms, 'inPF', sids) for name, tms, sids in CONTRASTS}
gz_shift = {}
for dec, dlab in DECODERS.items():
    for wlab, wr in GZ_WINDOWS.items():
        sh = {st: gaze_code_shift(st, dec, wr)[0] for st in ['orig', 'rep']}
        per = {'Study 1 · Early': sh['orig'].get('Early TMS'), 'Study 1 · Middle': sh['orig'].get('Middle TMS'),
               'Replication · Middle': sh['rep'].get('Middle TMS')}
        per['Both datasets · Middle'] = pd.concat([per['Study 1 · Middle'], per['Replication · Middle']])
        gz_shift[(dec, wlab)] = per
        for name, g in per.items():
            rows.append({'Test': '6a. TMS shift in gaze orientation code', 'Measure': f'{dlab} fidelity, inPF − outPF, {wlab}',
                         'Contrast': name, 'Location': 'inPF − outPF', 'Part': 'TMS − None', **summarize(g.values)})
            j = pd.concat([e_pull_all[name].rename('pull'), g.rename('gaze')], axis=1).dropna()
            r, p = stats.spearmanr(j['pull'], j['gaze'])
            rows.append({'Test': '6b. Gaze-code shift vs pull change', 'Measure': f'Spearman ρ, {dlab}, {wlab}', 'Contrast': name,
                         'Location': 'inPF', 'Part': 'across subjects', 'N': len(j), 'Mean': r, 'SEM': np.nan, 'd': np.nan, 'p': p})

ST = pd.DataFrame(rows)
ST.to_csv(os.path.join(OUT, 'Stats_GazePull.csv'), index=False)
excl.to_csv(os.path.join(OUT, 'Stats_GazePull_TrialsKept.csv'), index=False)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(13, 15))
# A: gaze toward stimulus, R and T targets at inPF / outPF, None vs Middle (both datasets)
ax = axs[0, 0]
cm = cell_means(D.dropna(subset=['toward_late']), 'toward_late')
cm = cm[cm['sid'].isin(S1 + REP) & cm['TMS'].isin(['No TMS', 'Middle TMS'])]
for i, loc in enumerate(['inPF', 'outPF']):
    for typ, col in [('R', '#1F4E9A'), ('T', '#7FA7DC')]:
        g = cm[cm['Hemifield'] == loc].groupby('TMS')[typ].agg(['mean', 'sem']).reindex(['No TMS', 'Middle TMS'])
        ax.errorbar(np.array([0, 1]) + i * 2.6, g['mean'], yerr=g['sem'], color=col, marker='o', capsize=3, lw=2,
                    label=('radial targets' if typ == 'R' else 'tangential targets') if i == 0 else None)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks([0, 1, 2.6, 3.6]); ax.set_xticklabels(['None', 'Middle', 'None', 'Middle'])
ax.text(0.5, 1.0, 'inPF', transform=ax.get_xaxis_transform(), ha='center', va='bottom'); ax.text(3.1, 1.0, 'outPF', transform=ax.get_xaxis_transform(), ha='center', va='bottom')
ax.set_ylabel('Gaze toward stimulus, late delay (°)'); ax.legend(frameon=False, fontsize=8)
ax.set_title('A. Gaze drift toward the remembered location (both datasets)', loc='left', fontsize=10, pad=16)
# B: trial-level association
ax = axs[0, 1]
order = ['toward', 'axial', 'mag']
for j, subset in enumerate(['all targets', 'tangential targets', 'radial targets']):
    g = assoc[assoc['subset'] == subset]
    m = g[order].mean(); se = g[order].sem()
    ax.errorbar(np.arange(3) + (j - 1) * 0.22, m, yerr=1.96 * se, fmt='o', capsize=3, label=subset,
                color=['0.2', '#7FA7DC', '#1F4E9A'][j])
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels(['toward\nstimulus', 'along vs across\nradial line', 'displacement\nsize'])
ax.set_ylabel('Change in pull (°) per 1 SD of gaze measure'); ax.legend(frameon=False, fontsize=8)
ax.set_title('B. Does gaze predict pull on single trials? (95% CI)', loc='left', fontsize=10)
# C: effect with gaze covariates and fixation exclusions
ax = axs[1, 0]
sub = ST[ST['Test'].isin(['3. Mixed model, inPF', '4. Fixation exclusion']) & (ST['Contrast'] == 'Both datasets · Middle')]
for i, (_, r) in enumerate(sub.iterrows()):
    ci = 1.96 * r['SEM'] if r['Test'].startswith('3') else stats.t.ppf(.975, r['N'] - 1) * r['SEM']
    ax.errorbar(r['Mean'], i, xerr=ci, fmt='o', color='#5B2C83', capsize=3)
    ax.text(1.02, i, f"{r['Mean']:+.2f}°, p={r['p']:.3f}", transform=ax.get_yaxis_transform(), va='center', fontsize=8)
ax.set_yticks(range(len(sub)))
ax.set_yticklabels([('Mixed model: ' if r['Test'].startswith('3') else 'Cell means: ') + r['Part'] for _, r in sub.iterrows()], fontsize=8)
ax.invert_yaxis(); ax.axvline(0, color='k', lw=1)
ax.set_xlabel('TMS change in radial − tangential pull at inPF (°), Middle TMS, both datasets')
ax.set_title('C. The pull effect after accounting for gaze', loc='left', fontsize=10)
# D: across subjects
ax = axs[1, 1]
j, r, p = across['toward_late']
ax.scatter(j['gaze'], j['pull'], color='#5B2C83', alpha=0.7)
ax.axhline(0, color='k', lw=0.8); ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('TMS change in gaze toward stimulus, R − T at inPF (°)'); ax.set_ylabel('TMS change in pull, R − T at inPF (°)')
ax.set_title(f'D. Across subjects: Spearman ρ = {r:+.2f}, p = {p:.2f}', loc='left', fontsize=10)
# E: TMS shift in the gaze orientation code at inPF relative to outPF
ax = axs[2, 0]
cols = {'Study 1 · Early': P.TMS_LINE_PAL['Early TMS'], 'Study 1 · Middle': P.TMS_LINE_PAL['Middle TMS'],
        'Replication · Middle': '#0B5394', 'Both datasets · Middle': '#5B2C83'}
for i, dec in enumerate(DECODERS):
    per = gz_shift[(dec, 'late delay (2.4–4.2 s)')]
    for j, (name, g) in enumerate(per.items()):
        g = g.values; z = (g - 0) / g.std(ddof=1) if g.std(ddof=1) > 0 else g * 0
        m, ci = g.mean() / g.std(ddof=1), stats.t.ppf(.975, len(g) - 1) / np.sqrt(len(g))   # standardized (dz) so decoders share an axis
        ax.errorbar(i + (j - 1.5) * 0.18, m, yerr=ci, fmt='o', color=cols[name], capsize=3, label=name if i == 0 else None)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(range(len(DECODERS))); ax.set_xticklabels(list(DECODERS.values()))
ax.set_ylabel('TMS change in gaze orientation code,\ninPF − outPF, late delay (dz ± 95% CI)')
ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.set_title('E. Gaze orientation code shifts toward inPF under TMS', loc='left', fontsize=10)
# F: does that gaze shift track the pull change across subjects?
ax = axs[2, 1]
g = gz_shift[('mahal_target', 'late delay (2.4–4.2 s)')]['Both datasets · Middle']
j2 = pd.concat([e_pull_all['Both datasets · Middle'].rename('pull'), g.rename('gaze')], axis=1).dropna()
r2, p2 = stats.spearmanr(j2['pull'], j2['gaze'])
ax.scatter(j2['gaze'], j2['pull'], color='#5B2C83', alpha=0.7)
ax.axhline(0, color='k', lw=0.8); ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('TMS change in gaze orientation code, inPF − outPF\n(Mahalanobis, late delay; Middle TMS)')
ax.set_ylabel('TMS change in pull, R − T at inPF (°)')
ax.set_title(f'F. Gaze-code shift vs pull change: ρ = {r2:+.2f}, p = {p2:.2f}', loc='left', fontsize=10)
for a in axs.flat: sns.despine(ax=a)
plt.tight_layout()
P.save(fig, OUT, 'G1_GazeAndPull')

pd.set_option('display.width', 230)
show = ST.copy()
print(show[(show['Test'] != '1. TMS change') | (show['Contrast'] == 'Both datasets · Middle')].round(4).to_string(index=False))
print(); print(excl.round(3).to_string(index=False))
