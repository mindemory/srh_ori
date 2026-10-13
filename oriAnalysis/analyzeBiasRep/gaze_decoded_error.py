"""
gaze_decoded_error.py   (srh conda env)

Does the orientation decoded from gaze follow the subject's report (the
remembered orientation, errors included) rather than the target, beyond what
shared systematic biases would produce?

gaze_timeresolved.py found that on the least accurate trials the ridge-decoded
angle is closer to the report than to the target. That can arise without any
trial-specific memory signal in gaze: if the decoder mostly recovers the
orientation category (its output pulled to the category centre) and reports are
also biased towards the category centre or the radial axis, decoded angles and
reports share a bias that depends only on the target. This script separates
the two.

Per subject (same 400-ms windows, features, folds and location centring as the
'standard' variant of gaze_timeresolved.py; ridge decoder; N_SHUFFLE shuffles of
targets and reports together within block x location):
  e      memory error, report - target (deg, wrapped to +-90)
  e_hat  error predicted from the target alone: cubic polynomial of the
         target's offset from its category centre, fitted within TMS x
         location x category cells (central tendency, radial pull, ...)
  Measures, each real minus the mean of the shuffled decoders; every decoder is
  scored against each trial's own target and report (scoring the shuffled
  decoders against the shuffled labels instead produces a spurious positive
  'report - target' on high-error trials, already present before the grating):
    report - target       cos 2(dec - report) - cos 2(dec - target)
    report - predicted    cos 2(dec - report) - cos 2(dec - target - e_hat):
                          > 0 if decoded angles follow the trial's own error
                          beyond what the target-dependent biases predict
    r(residuals)          correlation of sin 2(dec - target) with e - e_hat,
                          after removing the same cubic target dependence from
                          sin 2(dec - target) within cells
Subsets: all trials and |error| quartiles (within subject x TMS x location x
category, as in gaze_timeresolved.py).

Output: <COMPARE_DIR>/gaze/decoded_error_<study>.pkl, GZ11_DecodedError.png,
        Stats_DecodedError.csv
Usage:  python gaze_decoded_error.py [orig|rep ...]   then   python gaze_decoded_error.py --figures
"""
import os
import sys
import time
import pickle
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from biasrep_common import COMPARE_DIR
from gaze_decoder_zoo import load, ALPHAS, _cossin, _angle
from gaze_timeresolved import (window_features, _within_perm, window_centres, cell_quartile, signflip_p, window_mask,
                               WINDOWS, STUDY_LABELS, MIN_TRIALS, N_FOLDS, N_QUART, MIN_FRACTION, MIN_SUBSET, CELLS)

warnings.filterwarnings('ignore')
OUT = os.path.join(COMPARE_DIR, 'gaze')
N_SHUFFLE = 20


def wrap90(a):
    return (a + 90) % 180 - 90


def _ridge_angles(F, theta, loc, groups):
    ang = np.full(len(theta), np.nan)
    for tr, te in GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups)))).split(F, groups=groups):
        A, B = F[tr].copy(), F[te].copy()
        for v in np.unique(loc[tr]):
            mu = A[loc[tr] == v].mean(0)
            A[loc[tr] == v] -= mu; B[loc[te] == v] -= mu
        sd = A.std(0) + 1e-9
        ang[te] = _angle(RidgeCV(alphas=ALPHAS).fit(A / sd, _cossin(theta[tr])).predict(B / sd))
    return ang


def run_subject(s, D):
    rows = np.where(D['trials']['subjID'].values == s)[0]
    tr = D['trials'].iloc[rows]
    theta = tr['targOriActual'].values.astype(float) % 180
    loc = (tr['VF'].values == 'RVF').astype(int)
    groups = (tr['day'].astype(str) + '_' + tr['block'].astype(str)).values
    keys = np.array([f'{g}_{c}' for g, c in zip(groups, loc)])
    F, OK = window_features(D['X'][rows], D['Y'][rows])
    rng = np.random.default_rng(int(s))
    perms = [np.arange(len(rows))] + [_within_perm(keys, rng) for _ in range(N_SHUFFLE)]
    dec = np.full((len(perms), len(rows), len(F)), np.nan, np.float32)      # [0] real, [1:] shuffles
    for w, (f, ok) in enumerate(zip(F, OK)):
        if ok.sum() < MIN_TRIALS or np.bincount(loc[ok], minlength=2).min() < 5:
            continue
        idx = np.where(ok)[0]
        for k, p in enumerate(perms):
            dec[k, idx, w] = _ridge_angles(f[idx], theta[p[idx]], loc[idx], groups[idx])
    return s, {'rows': rows, 'perms': np.array(perms), 'dec': dec}


def main(study):
    D = load(study)
    subs = np.unique(D['trials']['subjID'])
    t0 = time.time()
    res = Parallel(n_jobs=int(os.environ.get('N_JOBS', 6)))(delayed(run_subject)(s, D) for s in subs)
    with open(os.path.join(OUT, f'decoded_error_{study}.pkl'), 'wb') as fh:
        pickle.dump({'centres': window_centres(D['centres']), 'trials': D['trials'], 'results': dict(res)}, fh, protocol=4)
    print(f'[{study}] {len(subs)} subjects in {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------
def _cubic_fit(y, j, cells):
    """Within-cell least-squares fit of y on [1, j, j^2, j^3] (columns of y fitted separately)."""
    out = np.full(y.shape, np.nan)
    for c in np.unique(cells):
        m = cells == c
        X = np.column_stack([np.ones(m.sum()), j[m], j[m] ** 2, j[m] ** 3])
        Y = y[m]
        ok = ~np.isnan(Y)
        for col in range(Y.shape[1]):
            o = ok[:, col]
            if o.sum() >= 8:
                beta = np.linalg.lstsq(X[o], Y[o, col], rcond=None)[0]
                out[np.where(m)[0][o], col] = X[o] @ beta
    return out


def _fid(a, b):
    return np.cos(np.radians(2 * (a - b)))


def measures(tr, perms, dec):
    """Per trial x window, real and null-mean values of the three measures."""
    theta = tr['targOriActual'].values.astype(float) % 180
    report = tr['finalOri'].values.astype(float) % 180
    jit = wrap90(theta - tr['OriCat'].values.astype(float))
    cells = tr[CELLS].astype(str).agg('_'.join, axis=1).values
    out = {k: [] for k in ['report − target', 'report − predicted', 'r(residuals)']}
    e = wrap90(report - theta)
    e_hat = _cubic_fit(e[:, None], jit, cells)[:, 0]
    for k in range(len(perms)):                       # real decoder (k = 0) and shuffled decoders, all scored
        a = dec[k].astype(float)                      # against each trial's own target and report
        out['report − target'].append(_fid(a, report[:, None]) - _fid(a, theta[:, None]))
        out['report − predicted'].append(_fid(a, report[:, None]) - _fid(a, (theta + e_hat)[:, None]))
        sd = np.sin(np.radians(2 * (a - theta[:, None])))
        sd_res = sd - _cubic_fit(sd, jit, cells)
        out['r(residuals)'].append((sd_res, (e - e_hat)))
    return out


def subject_curves(R, s, groups_q=True):
    r = R['results'][s]
    tr = R['trials'].iloc[r['rows']]
    M = measures(tr, r['perms'], r['dec'])
    usable = ~np.isnan(r['dec'][0])
    sparse = usable.mean(0) < MIN_FRACTION
    q = cell_quartile(tr, 'AbsError')
    masks = {'All': np.ones(len(tr), bool), **{f'Err Q{i + 1}': q == i for i in range(N_QUART)}}
    res = {}
    for name, mk in masks.items():
        drop = sparse | (usable[mk].sum(0) < MIN_SUBSET)
        for meas in ['report − target', 'report − predicted']:
            vals = [np.nanmean(v[mk], 0) for v in M[meas]]
            d = vals[0] - np.nanmean(vals[1:], 0)
            d[drop] = np.nan
            res[(meas, name)] = d
        rs = []
        for sd_res, e_res in M['r(residuals)']:
            a, b = sd_res[mk], np.broadcast_to(e_res[mk, None], sd_res[mk].shape)
            ok = ~np.isnan(a) & ~np.isnan(b)
            a = np.where(ok, a, np.nan); b = np.where(ok, b, np.nan)
            with np.errstate(invalid='ignore', divide='ignore'):
                a = a - np.nanmean(a, 0); b = b - np.nanmean(b, 0)
                rs.append(np.nansum(a * b, 0) / np.sqrt(np.nansum(a ** 2, 0) * np.nansum(b ** 2, 0)))
        d = rs[0] - np.nanmean(rs[1:], 0)
        d[drop] = np.nan
        res[('r(residuals)', name)] = d
    # behaviour: central tendency (slope of error on offset from the category centre)
    theta = tr['targOriActual'].values.astype(float) % 180
    e = wrap90(tr['finalOri'].values.astype(float) % 180 - theta)
    jit = wrap90(theta - tr['OriCat'].values.astype(float))
    res['slope'] = np.polyfit(jit, e, 1)[0]
    return res


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import panels as P
    from gaze_figures import cluster_test, decorate, tc_plot
    sns.set_theme(style='white', context='notebook')
    matplotlib.rcParams['pdf.fonttype'] = 42
    qpal = dict(zip([f'Err Q{i + 1}' for i in range(N_QUART)], sns.color_palette('viridis', N_QUART)))
    ylabs = {'report − target': 'cos 2(dec − report) −\ncos 2(dec − target)',
             'report − predicted': 'cos 2(dec − report) −\ncos 2(dec − bias-predicted report)',
             'r(residuals)': 'r(decoded deviation,\nmemory error) after\nremoving target-dependent bias'}
    fig, axs = plt.subplots(3, 2, figsize=(15, 10.5), sharex=True)
    rows, clusters = [], []
    for c, (st, lab) in enumerate(STUDY_LABELS):
        with open(os.path.join(OUT, f'decoded_error_{st}.pkl'), 'rb') as fh:
            R = pickle.load(fh)
        x = R['centres']
        per = [subject_curves(R, s) for s in R['results']]
        slopes = np.array([p['slope'] for p in per])
        rows.append(dict(Study=lab, Measure='behaviour: slope of error on offset from category centre', Subset='All',
                         Window='', N=len(slopes), Mean=slopes.mean(), SEM=slopes.std(ddof=1) / np.sqrt(len(slopes)),
                         p=signflip_p(slopes)))
        for r, meas in enumerate(['report − target', 'report − predicted', 'r(residuals)']):
            ax = axs[r, c]
            for name in ['All'] + list(qpal):
                data = np.array([p[(meas, name)] for p in per])
                col = '0.4' if name == 'All' else qpal[name]
                tc_plot(ax, x, data, col, name + (' (most accurate)' if name == 'Err Q1' else ' (least accurate)' if name == f'Err Q{N_QUART}' else ''))
                for a, b, pv in cluster_test(data):
                    mean = np.nanmean(data[:, a:b + 1])
                    clusters.append(dict(Study=lab, Measure=meas, Subset=name, window_centre_from=x[a], window_centre_to=x[b],
                                         direction='+' if mean > 0 else '−', mean=mean, p=pv))
                    if pv < 0.05:
                        slot = ['All'] + list(qpal)
                        y = 0.97 - 0.035 * slot.index(name) if mean > 0 else 0.03 + 0.035 * slot.index(name)
                        ax.plot([x[a], x[b]], [y, y], color=col, lw=3, solid_capstyle='butt',
                                transform=ax.get_xaxis_transform(), clip_on=False)
                for wname, (lo, hi) in WINDOWS.items():
                    vals = np.nanmean(data[:, window_mask(x, lo, hi, [])], 1)
                    k = int(np.sum(~np.isnan(vals)))
                    rows.append(dict(Study=lab, Measure=meas, Subset=name, Window=wname, N=k, Mean=np.nanmean(vals),
                                     SEM=np.nanstd(vals, ddof=1) / np.sqrt(k), p=signflip_p(vals)))
            decorate(ax, [], ylabs[meas] + '\n(real − null)', 0)
            if r < 2:
                ax.set_xlabel('')
            if r == 0:
                ax.set_title(f'{lab}: does decoded gaze follow the report?', fontsize=11, fontweight='bold')
                ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper left')
        axs[0, c].text(0.99, 0.02, f'behaviour: error vs offset slope {slopes.mean():+.2f} '
                                   f'(central tendency {"yes" if slopes.mean() < 0 else "no"})',
                       transform=axs[0, c].transAxes, ha='right', fontsize=8)
    fig.suptitle('Ridge-decoded orientation vs memory reports (|error| quartiles within subject x TMS x location x category; '
                 'bars: cluster p < .05, top = positive, bottom = negative)', y=0.997)
    plt.tight_layout(); P.save(fig, OUT, 'GZ11_DecodedError')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'Stats_DecodedError.csv'), index=False)
    pd.DataFrame(clusters).to_csv(os.path.join(OUT, 'Stats_DecodedError_clusters.csv'), index=False)


if __name__ == '__main__':
    if '--figures' in sys.argv:
        figures()
    else:
        for st in [a for a in sys.argv[1:] if not a.startswith('--')] or ['orig', 'rep']:
            main(st)
