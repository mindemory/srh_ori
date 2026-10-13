"""
gaze_decoding.py   (run with the srh conda env: ~/.conda/envs/srh/bin/python)

Time-resolved decoding of stimulus location and remembered orientation from
eye data (gaze position, axial gaze displacement, pupil) in Study 1 and the bias
replication, using the epochs from gaze_epochs.py.

Preprocessing
  - per-trial baseline: mean of the 200 ms before grating onset subtracted
    from X, Y and pupil (as in Mostert et al. 2018; Kang & Woodman 2014)
  - trials dropped if: behavioural outlier (LargeError), < 50% valid gaze in the
    baseline or in the grating + delay period, or gaze beyond 2.5 deg of the
    baseline position at any point in grating + delay (fixation breaks)
  - features averaged in 100-ms windows stepped by 50 ms

Decoders (per subject, cross-validated with GroupKFold over blocks)
  location     stimulus visual field (left/right; = inPF/outPF within a subject)
               from gaze X, Y; shrinkage LDA; decision values -> AUC
  orientation  ridge regression of [cos 2theta, sin 2theta] on features, after
               removing the stimulus-location mean of each feature (training
               folds only); decoded angle = atan2 / 2; trial fidelity =
               cos 2(decoded - true), 0 = no information
               feature sets:
                 'gaze'    X, Y, X^2 - Y^2, 2XY (position + axial displacement)
                 'pupil'   pupil size (control)
                 'mirror'  gaze features after mirroring left-field trials
                           (x -> -x, theta -> 180 - theta), i.e. orientation
                           relative to the radial axis, pooled across locations
  null         the same decoders with orientations shuffled within each
               block x location (keeps every fold's label composition, so the
               cross-validation bias matches the real decoder); N_SHUFFLE times;
               null fidelity is kept separately for every trial subset used later
               (all, each TMS condition, each TMS condition x hemifield)

Output: <COMPARE_DIR>/gaze/decoding_<study>.pkl  (trial table, baseline-
corrected epochs, per-trial per-window cross-validated predictions, nulls)

Usage: python gaze_decoding.py [orig|rep]
"""
import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from biasrep_common import COMPARE_DIR, ORIG_ROOT, DATA_ROOT
from study_results import load_study

warnings.filterwarnings('ignore')
OUT_DIR = os.path.join(COMPARE_DIR, 'gaze')
os.makedirs(OUT_DIR, exist_ok=True)
ROOTS = {'orig': ORIG_ROOT, 'rep': DATA_ROOT}
BASELINE = (-200, 0)        # ms
TASK = (0, 4250)            # grating + delay
FIX_LIMIT = 2.5             # deg
WIN, STEP = 100, 50         # ms
N_SHUFFLE = 20
N_FOLDS = 5
MIN_TRIALS = 40
ALPHAS = np.logspace(-2, 4, 13)
FEATURE_SETS = ['gaze', 'pupil', 'mirror']


def prepare(study):
    z = np.load(os.path.join(ROOTS[study], 'analysis', f'gaze_epochs_{study}.npz'))
    t = z['times']
    meta = pd.DataFrame({k[5:]: z[k] for k in z.files if k.startswith('meta_')})
    beh = load_study(study)                      # outliers removed; index = row in the raw table
    X, Y, P, V = z['X'], z['Y'], z['Pupil'], z['valid']
    base = (t >= BASELINE[0]) & (t < BASELINE[1])
    task = (t >= TASK[0]) & (t < TASK[1])
    with np.errstate(invalid='ignore'):
        Xb = X - np.nanmean(X[:, base], axis=1, keepdims=True)
        Yb = Y - np.nanmean(Y[:, base], axis=1, keepdims=True)
        Pb = P - np.nanmean(P[:, base], axis=1, keepdims=True)
        excursion = np.nanmax(np.hypot(Xb[:, task], Yb[:, task]), axis=1)
    reasons = pd.Series('kept', index=meta.index)
    reasons[~meta['beh_row'].isin(beh.index)] = 'behavioural outlier'
    reasons[(reasons == 'kept') & (V[:, base].mean(1) < 0.5)] = 'baseline missing'
    reasons[(reasons == 'kept') & (V[:, task].mean(1) < 0.5)] = 'task period missing'
    reasons[(reasons == 'kept') & ~(excursion <= FIX_LIMIT)] = 'fixation break'
    keep = (reasons == 'kept').values
    trials = beh.loc[meta.loc[keep, 'beh_row']].reset_index(drop=True)
    trials['pulse_ms'] = meta.loc[keep, 'pulse_ms'].values
    trials['excursion'] = excursion[keep]
    qc = meta[['subjID', 'beh_row']].assign(reason=reasons.values)
    qc = qc.merge(beh[['TMS']], left_on='beh_row', right_index=True, how='left')
    return dict(t=t, X=Xb[keep], Y=Yb[keep], P=Pb[keep], V=V[keep], trials=trials, qc=qc, study=study)


def window_features(D):
    t = D['t']
    centres = np.arange(t[0] + WIN / 2, t[-1] - WIN / 2 + 1, STEP)
    idx = [np.where((t >= c - WIN / 2) & (t < c + WIN / 2))[0] for c in centres]
    with np.errstate(invalid='ignore'):
        f = {k: np.stack([np.nanmean(D[k][:, i], axis=1) for i in idx], axis=1) for k in ['X', 'Y', 'P']}
    return centres, f


def _features(x, y, p, mirror):
    return {'gaze': np.column_stack([x, y, x ** 2 - y ** 2, 2 * x * y]),
            'pupil': p[:, None],
            'mirror': np.column_stack([x * mirror, y, x ** 2 - y ** 2, 2 * x * mirror * y])}


def _shuffle_within(theta, keys, rng):
    th = theta.copy()
    for k in np.unique(keys):
        m = keys == k
        th[m] = rng.permutation(th[m])
    return th


def _ridge_cv(F, theta, loc, groups):
    """Cross-validated decoded orientation (deg); NaN where features are missing."""
    out = np.full(len(theta), np.nan)
    ok = ~np.isnan(F).any(axis=1)
    if ok.sum() < MIN_TRIALS:
        return out
    Fo, th, lo, gr = F[ok], theta[ok], loc[ok], groups[ok]
    T = np.column_stack([np.cos(np.radians(2 * th)), np.sin(np.radians(2 * th))])
    pred = np.full(len(th), np.nan)
    for tr, te in GroupKFold(n_splits=min(N_FOLDS, len(np.unique(gr)))).split(Fo, groups=gr):
        A, B = Fo[tr].copy(), Fo[te].copy()
        for v in np.unique(lo[tr]):           # remove the location mean (training folds only)
            mu = A[lo[tr] == v].mean(0)
            A[lo[tr] == v] -= mu
            B[lo[te] == v] -= mu
        sd = A.std(0) + 1e-9
        c, s = RidgeCV(alphas=ALPHAS).fit(A / sd, T[tr]).predict(B / sd).T
        pred[te] = np.degrees(np.arctan2(s, c)) / 2 % 180
    out[np.where(ok)[0]] = pred
    return out


def _lda_cv(F, loc, groups):
    out = np.full(len(loc), np.nan)
    ok = ~np.isnan(F).any(axis=1)
    if ok.sum() < MIN_TRIALS:
        return out
    Fo, lo, gr = F[ok], loc[ok], groups[ok]
    dv = np.full(len(lo), np.nan)
    for tr, te in GroupKFold(n_splits=min(N_FOLDS, len(np.unique(gr)))).split(Fo, groups=gr):
        dv[te] = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Fo[tr], lo[tr]).decision_function(Fo[te])
    out[np.where(ok)[0]] = dv
    return out


def subsets(tr):
    """Named boolean masks over a subject's trials used for condition-wise metrics."""
    m = {'All': np.ones(len(tr), bool)}
    for g in tr['TMS'].unique():
        m[g] = tr['TMS'].values == g
        for h in ['inPF', 'outPF']:
            m[f'{g} · {h}'] = m[g] & (tr['Hemifield'].values == h)
    for gt in ['Radial', 'Tangential']:
        m[f'Geo {gt}'] = tr['GeoType'].values == gt
    return m


def decode_subject(s, D, centres, f):
    sel = np.where(D['trials']['subjID'].values == s)[0]
    tr = D['trials'].iloc[sel]
    theta = tr['targOriActual'].values.astype(float)
    theta_m = np.where(tr['VF'].values == 'LVF', 180 - theta, theta) % 180     # mirrored (radial-relative) frame
    mirror = np.where(tr['VF'].values == 'LVF', -1.0, 1.0)
    loc = (tr['VF'].values == 'RVF').astype(int)
    groups = (tr['day'].astype(str) + '_' + tr['block'].astype(str)).values
    shuffle_keys = groups + '_' + loc.astype(str)
    targets = {'gaze': theta, 'pupil': theta, 'mirror': theta_m}
    masks = subsets(tr)
    fs = {k: v[sel] for k, v in f.items()}
    rng = np.random.default_rng(int(s))
    nW = len(centres)
    res = {'rows': sel, 'loc_dv': np.full((len(sel), nW), np.nan), 'theta_m': theta_m}
    for name in FEATURE_SETS:
        res[f'ori_{name}'] = np.full((len(sel), nW), np.nan)
        res[f'null_{name}'] = {k: np.full((N_SHUFFLE, nW), np.nan) for k in masks}
    for w in range(nW):
        F = _features(fs['X'][:, w], fs['Y'][:, w], fs['P'][:, w], mirror)
        res['loc_dv'][:, w] = _lda_cv(np.column_stack([fs['X'][:, w], fs['Y'][:, w]]), loc, groups)
        for name in FEATURE_SETS:
            res[f'ori_{name}'][:, w] = _ridge_cv(F[name], targets[name], loc, groups)
            for k in range(N_SHUFFLE):
                th_sh = _shuffle_within(targets[name], shuffle_keys, rng)
                pred = _ridge_cv(F[name], th_sh, loc, groups)
                fid = np.cos(np.radians(2 * (pred - th_sh)))
                for mk, mm in masks.items():
                    res[f'null_{name}'][mk][k, w] = np.nanmean(fid[mm]) if np.any(mm & ~np.isnan(fid)) else np.nan
    return s, res


def main(study):
    D = prepare(study)
    centres, f = window_features(D)
    subs = np.unique(D['trials']['subjID'])
    print(f'[{study}] {len(D["trials"])} trials kept from {len(D["qc"])}; decoding {len(subs)} subjects x {len(centres)} windows')
    print(D['qc'].groupby(['TMS', 'reason']).size().unstack(fill_value=0).to_string())
    out = Parallel(n_jobs=6)(delayed(decode_subject)(s, D, centres, f) for s in subs)
    with open(os.path.join(OUT_DIR, f'decoding_{study}.pkl'), 'wb') as fh:
        pickle.dump({'centres': centres, 'trials': D['trials'], 'qc': D['qc'], 'results': dict(out),
                     'X': D['X'], 'Y': D['Y'], 'P': D['P'], 'V': D['V'], 't': D['t']}, fh)
    print(f'saved {os.path.join(OUT_DIR, f"decoding_{study}.pkl")}')


if __name__ == '__main__':
    for st in (sys.argv[1:] or ['orig', 'rep']):
        main(st)
