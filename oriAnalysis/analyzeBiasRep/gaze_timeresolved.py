"""
gaze_timeresolved.py   (srh conda env)

Time-resolved orientation decoding from gaze with ridge regression, Mahalanobis
distance tuning and a TAFKAP-style Bayesian decoder, split by TMS condition,
TMS x hemifield and by memory accuracy (|error| and RT quartiles), in Study 1
and the replication.

Data: 100-ms binned, baseline-corrected gaze and the trial selection of
gaze_decoding.py (via gaze_decoder_zoo.load).

Features  window-mean x, y, x^2 - y^2, 2xy (the zoo's 'gaze_mean', its best
          feature set for ridge and Mahalanobis) in 400-ms windows stepped by
          100 ms (window centres -200 ... 4000 ms)
Decoders  per subject, trained on all of the subject's trials, cross-validated
          with GroupKFold over blocks; features centred within each stimulus
          location ('standard') or within location x radial/tangential
          ('control', see gaze_decoder_zoo.py), training folds only
  Ridge        ridge regression to [cos 2theta, sin 2theta] -> decoded angle
  Mahalanobis  Mahalanobis distance tuning (Wolff et al. 2017), Ledoit-Wolf
               covariance, over orientation bins = category x side of the
               category centre (Study 1: 8 bins, replication: 4), so the
               within-category jitter is used too
  TAFKAP       TAFKAP-style decoder (van Bergen & Jehee 2021; jeheelab/TAFKAP):
               generative model features = W c(theta) + Gaussian noise;
               c = 8 half-wave-rectified cos^5 channels on the doubled angle,
               channel phase drawn per bag from 4 offsets; W by least squares;
               noise covariance = Ledoit-Wolf-shrunk residual covariance (TAFKAP
               cross-validates a shrinkage towards a W W'-structured target,
               which matters for many voxels, not for 4 features); likelihood on
               a 100-point grid, flat prior, posteriors summed over N_BAG
               bootstrap fits; decoded angle = circular mean, uncertainty =
               circular SD of the posterior (deg), as in TAFKAP.py
Scores    per trial: fidelity cos 2(decoded - target) and cos 2(decoded -
          report) for ridge and TAFKAP; tuning to the target and to the report
          for Mahalanobis; TAFKAP posterior SD and |decoded - target|
Null      N_SHUFFLE decoders trained on targets (and bins) shuffled within
          block x location (x radial/tangential for the control), each scored
          against every trial's OWN target and report. The per-trial null mean
          is kept, so any trial subset is compared with the null on the same
          trials with the same labels; this matters for subsets defined by
          behaviour or by the target (error/RT quartiles, radial/tangential),
          where scoring shuffled decoders against the shuffled labels gives a
          biased null (e.g. a spurious 'decoded angle closer to the report' on
          high-error trials, present even before the grating). TAFKAP
          uncertainty and decoding error are kept for every shuffle (float16).

Subject values: a subject's window is dropped if < 50% of its trials have usable
gaze there (e.g. post-stimulus blinks), and a subset value if < 10 of the
subset's trials are usable.

Splits: all trials; each TMS condition; TMS x hemifield; radial / tangential;
quartiles of |error| and of RT defined within subject x TMS x location x
orientation category (every quartile has the same condition mix; Q1 = most
accurate / fastest), alone and within each TMS condition.

TAFKAP uncertainty analyses: within-subject correlations of posterior SD with
|decoding error| (calibration), |memory error| and RT, computed on ranks
z-scored within TMS x location x orientation cells, real minus the mean of the
shuffled decoders (partial correlations control the norm of the trial's
feature vector).

Usage:  python gaze_timeresolved.py [orig|rep ...]     (decoding)
        python gaze_timeresolved.py --figures          (statistics + figures)
Output: <COMPARE_DIR>/gaze/timeresolved_<study>.pkl, GZ7-GZ10 figures,
        Stats_TimeResolved.csv, Stats_TimeResolved_clusters.csv
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

from biasrep_common import COMPARE_DIR, STUDIES
from gaze_decoder_zoo import load, ALPHAS, _cossin, _angle

warnings.filterwarnings('ignore')
OUT = os.path.join(COMPARE_DIR, 'gaze')
WIN_BINS = 4                  # 4 x 100-ms bins = 400-ms windows, stepped by 100 ms
HALF_WIN = WIN_BINS * 100 / 2
N_SHUFFLE = 20
N_FOLDS = 5
MIN_TRIALS = 40
VARIANTS = ['standard', 'control']
DECODED = ['ridge_target', 'ridge_report', 'mahal_target', 'mahal_report',
           'tafkap_target', 'tafkap_report', 'tafkap_unc', 'tafkap_abserr']
N_BAG = 50
TF_EXP, TF_GRID = 5, 100                                       # cos^5 channels, 100-point grid
TF_CENTRES = np.arange(8) * 2 * np.pi / 8
TF_PHASES = np.linspace(0, 2 * np.pi / 8, 5)[:4]
GRID = np.linspace(0, 2 * np.pi, TF_GRID, endpoint=False)       # doubled-angle grid
N_QUART = 4


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------
def _lw_precision(R):
    """Inverse of the Ledoit-Wolf shrunk covariance (as sklearn.covariance.ledoit_wolf)."""
    R = R - R.mean(0)
    n, p = R.shape
    S = R.T @ R / n
    mu = np.trace(S) / p
    delta = ((S - mu * np.eye(p)) ** 2).sum() / p
    R2 = R ** 2
    beta = min(((R2.T @ R2) / n - S ** 2).sum() / (p * n), delta)
    shrink = beta / delta if delta > 0 else 0.0
    return np.linalg.inv((1 - shrink) * S + shrink * mu * np.eye(p))


def _mahal(A, lab, th, B, th_tests):
    labs = np.unique(lab)
    means = np.stack([A[lab == c].mean(0) for c in labs])
    centre = np.array([np.angle(np.exp(2j * np.radians(th[lab == c])).mean()) for c in labs])   # doubled angle
    P = _lw_precision(A - means[np.searchsorted(labs, lab)])
    diff = B[:, None, :] - means[None]
    d = np.sqrt(np.einsum('ncf,fg,ncg->nc', diff, P, diff))
    d -= d.mean(1, keepdims=True)
    return [-(np.cos(centre[None, :] - np.radians(2 * t)[:, None]) * d).mean(1) for t in th_tests]


def _basis(s, ph):
    c = np.maximum(np.cos(s[:, None] - TF_CENTRES[None, :] - ph), 0)
    c2 = c * c
    return c2 * c2 * c                                                  # cos^5 (TF_EXP)


GRID_BASIS = [_basis(GRID, ph) for ph in TF_PHASES]


def _tafkap(A, theta, B, rng):
    s, n = np.radians(2 * theta), len(A)
    post = np.zeros((len(B), TF_GRID))
    for _ in range(N_BAG):
        i = rng.integers(0, n, n)
        k = rng.integers(len(TF_PHASES))
        C = _basis(s[i], TF_PHASES[k])
        W = np.linalg.lstsq(C, A[i], rcond=None)[0]                     # channels x features
        P = _lw_precision(A[i] - C @ W)
        M = GRID_BASIS[k] @ W                                           # predicted pattern per grid orientation
        PM = P @ M.T
        ll = B @ PM - 0.5 * (M * PM.T).sum(1)                           # Gaussian log-likelihood up to a per-trial constant
        lik = np.exp(ll - ll.max(1, keepdims=True))
        post += lik / lik.sum(1, keepdims=True)
    z = (post / post.sum(1, keepdims=True)) @ np.exp(1j * GRID)
    return np.degrees(np.angle(z)) / 2 % 180, np.degrees(np.sqrt(-2 * np.log(np.abs(z)))) / 2


def _fid(a, b):
    return np.cos(np.radians(2 * (a - b)))


def _cv(F, theta, report, bins, centring, groups, rng, score=None):
    """Cross-validated decoders trained on (theta, bins); scored against score = (target, report)
    per trial (default: the training labels)."""
    s_th, s_rp = (theta, report) if score is None else score
    n = len(theta)
    ang, est = np.full(n, np.nan), np.full(n, np.nan)
    o = {k: np.full(n, np.nan) for k in ['mahal_target', 'mahal_report', 'tafkap_unc', 'norm']}
    for tr, te in GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups)))).split(F, groups=groups):
        A, B = F[tr].copy(), F[te].copy()
        for v in np.unique(centring[tr]):                                # location (x class) means, training folds only
            mu = A[centring[tr] == v].mean(0)
            A[centring[tr] == v] -= mu; B[centring[te] == v] -= mu
        sd = A.std(0) + 1e-9
        A, B = A / sd, B / sd
        o['norm'][te] = np.linalg.norm(B, axis=1)
        ang[te] = _angle(RidgeCV(alphas=ALPHAS).fit(A, _cossin(theta[tr])).predict(B))
        o['mahal_target'][te], o['mahal_report'][te] = _mahal(A, bins[tr], theta[tr], B, (s_th[te], s_rp[te]))
        est[te], o['tafkap_unc'][te] = _tafkap(A, theta[tr], B, rng)
    o.update(ridge_target=_fid(ang, s_th), ridge_report=_fid(ang, s_rp), tafkap_target=_fid(est, s_th),
             tafkap_report=_fid(est, s_rp), tafkap_abserr=np.abs((est - s_th + 90) % 180 - 90))
    return o


# ---------------------------------------------------------------------------
# One subject
# ---------------------------------------------------------------------------
def window_features(X, Y):
    F, OK = [], []
    for s0 in range(X.shape[1] - WIN_BINS + 1):
        xb, yb = X[:, s0:s0 + WIN_BINS], Y[:, s0:s0 + WIN_BINS]
        with np.errstate(invalid='ignore'):
            xm, ym = np.nanmean(xb, 1), np.nanmean(yb, 1)
        f = np.column_stack([xm, ym, xm ** 2 - ym ** 2, 2 * xm * ym])
        F.append(f)
        OK.append(((~np.isnan(xb)).sum(1) >= WIN_BINS // 2) & ~np.isnan(f).any(1))
    return F, OK


def _within_perm(keys, rng):
    p = np.arange(len(keys))
    for k in np.unique(keys):
        m = np.where(keys == k)[0]
        p[m] = rng.permutation(m)
    return p


def run_subject(s, D, windows=None):
    rows = np.where(D['trials']['subjID'].values == s)[0]
    tr = D['trials'].iloc[rows]
    theta = tr['targOriActual'].values.astype(float) % 180
    report = tr['finalOri'].values.astype(float) % 180
    oricat = tr['OriCat'].values.astype(float)
    bins = np.searchsorted(np.unique(oricat), oricat) * 2 + (((theta - oricat + 90) % 180 - 90) > 0)
    loc = (tr['VF'].values == 'RVF').astype(int)
    geo = (tr['GeoType'].values == 'Radial').astype(int)
    groups = (tr['day'].astype(str) + '_' + tr['block'].astype(str)).values
    F, OK = window_features(D['X'][rows], D['Y'][rows])
    n, W = len(rows), len(F)
    rng = np.random.default_rng(int(s))
    out = {'rows': rows}
    for variant in VARIANTS:
        centring = loc * 2 + geo if variant == 'control' else loc
        keys = np.array([f'{g}_{c}' for g, c in zip(groups, centring)])
        perms = [_within_perm(keys, rng) for _ in range(N_SHUFFLE)]
        real = {k: np.full((n, W), np.nan, np.float32) for k in DECODED + ['norm']}
        null = {k: np.full((n, W), np.nan, np.float32) for k in DECODED}
        null_unc = np.full((N_SHUFFLE, n, W), np.nan, np.float16)
        null_abserr = np.full((N_SHUFFLE, n, W), np.nan, np.float16)
        for w in (range(W) if windows is None else windows):
            ok = OK[w]
            if ok.sum() < MIN_TRIALS or np.bincount(centring[ok], minlength=centring.max() + 1).min() < 5:
                continue
            idx = np.where(ok)[0]
            acc = {k: np.zeros(len(idx)) for k in DECODED}
            for k, p in enumerate([None] + perms):
                q = idx if p is None else p[idx]                         # whose labels each usable trial is trained with
                sc = _cv(F[w][idx], theta[q], report[q], bins[q], centring[idx], groups[idx], rng,
                         score=(theta[idx], report[idx]))                # always scored against the trial's own labels
                if p is None:
                    for m, v in sc.items():
                        real[m][idx, w] = v
                else:
                    for m in DECODED:
                        acc[m] += sc[m] / N_SHUFFLE
                    null_unc[k - 1, idx, w] = sc['tafkap_unc']
                    null_abserr[k - 1, idx, w] = sc['tafkap_abserr']
            for m in DECODED:
                null[m][idx, w] = acc[m]
        out[variant] = dict(real=real, null=null, null_unc=null_unc, null_abserr=null_abserr)
    return s, out


def window_centres(centres):
    return np.array([centres[i:i + WIN_BINS].mean() for i in range(len(centres) - WIN_BINS + 1)])


def main(study):
    D = load(study)
    subs = np.unique(D['trials']['subjID'])
    t0 = time.time()
    res = Parallel(n_jobs=int(os.environ.get("N_JOBS", 7)))(delayed(run_subject)(s, D) for s in subs)
    with open(os.path.join(OUT, f'timeresolved_{study}.pkl'), 'wb') as fh:
        pickle.dump({'centres': window_centres(D['centres']), 'trials': D['trials'], 'results': dict(res)}, fh, protocol=4)
    print(f'[{study}] {len(subs)} subjects in {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------------------
# Subsets, time courses, uncertainty correlations
# ---------------------------------------------------------------------------
CELLS = ['TMS', 'VF', 'OriCat']
MIN_FRACTION = 0.5           # a subject's window is used only if >= 50% of its trials have gaze there
MIN_SUBSET = 10              # and a subset value only if >= 10 of the subset's trials are usable


def cell_quartile(tr, col):
    return tr.groupby(CELLS)[col].transform(
        lambda v: np.floor((v.rank(method='first') - 0.5) / len(v) * N_QUART)).values


def subsets(tr, groups):
    tms, hemi = tr['TMS'].values, tr['Hemifield'].values
    m = {'All': np.ones(len(tr), bool)}
    for g in groups:
        m[g] = tms == g
        for h in ['inPF', 'outPF']:
            m[f'{g} · {h}'] = m[g] & (hemi == h)
    for gt in ['Radial', 'Tangential']:
        m[gt] = tr['GeoType'].values == gt
    for col, tag in [('AbsError', 'Err'), ('RT', 'RT')]:
        q = cell_quartile(tr, col)
        for i in range(N_QUART):
            m[f'{tag} Q{i + 1}'] = q == i
            for g in groups:
                m[f'{g} · {tag} Q{i + 1}'] = (q == i) & (tms == g)
    return m


def _rank_z(V, cells):
    """Ranks along the trial axis (-2), z-scored within each cell; NaN kept."""
    V = V.astype(float)
    Z = np.full(V.shape, np.nan)
    for c in np.unique(cells):
        m = cells == c
        sub = V[..., m, :]
        r = np.argsort(np.argsort(np.where(np.isnan(sub), np.inf, sub), axis=-2), axis=-2).astype(float)
        r[np.isnan(sub)] = np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            Z[..., m, :] = (r - np.nanmean(r, -2, keepdims=True)) / np.nanstd(r, -2, keepdims=True)
    return Z


def _corr(a, b, m):
    a, b = np.broadcast_arrays(a[..., m, :], b[..., m, :])
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = np.where(ok, a, np.nan), np.where(ok, b, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        a = a - np.nanmean(a, -2, keepdims=True); b = b - np.nanmean(b, -2, keepdims=True)
        return np.nansum(a * b, -2) / np.sqrt(np.nansum(a ** 2, -2) * np.nansum(b ** 2, -2))


def _partial(rab, rac, rbc):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))


UNC_ANALYSES = ['Calibration: |decoding error|', 'Memory |error|', 'Memory |error| | feature norm', 'RT']


def uncertainty_corr(tr, rv, groups):
    """Real - null within-subject correlations of TAFKAP posterior SD, per window."""
    cells = tr[CELLS].astype(str).agg('_'.join, axis=1).values
    z = {'unc': _rank_z(rv['real']['tafkap_unc'], cells), 'abserr': _rank_z(rv['real']['tafkap_abserr'], cells),
         'norm': _rank_z(rv['real']['norm'], cells),
         'err': _rank_z(tr['AbsError'].values[:, None], cells), 'rt': _rank_z(tr['RT'].values[:, None], cells)}
    zn = {'unc': _rank_z(rv['null_unc'], cells), 'abserr': _rank_z(rv['null_abserr'], cells)}
    out = {}
    for name in ['All'] + list(groups):
        m = np.ones(len(tr), bool) if name == 'All' else tr['TMS'].values == name
        r_en = _corr(z['err'], z['norm'], m)
        for label, zu, za in [('real', z['unc'], z['abserr']), ('null', zn['unc'], zn['abserr'])]:
            r = {UNC_ANALYSES[0]: _corr(zu, za, m), UNC_ANALYSES[1]: _corr(zu, z['err'], m),
                 UNC_ANALYSES[2]: _partial(_corr(zu, z['err'], m), _corr(zu, z['norm'], m), r_en),
                 UNC_ANALYSES[3]: _corr(zu, z['rt'], m)}
            for k, v in r.items():
                if label == 'real':
                    out[(k, name)] = v
                else:
                    out[(k, name)] = out[(k, name)] - np.nanmean(v, 0)
    return out


def time_courses(study):
    with open(os.path.join(OUT, f'timeresolved_{study}.pkl'), 'rb') as fh:
        R = pickle.load(fh)
    groups = STUDIES[study]['groups']
    TC, RHO = {}, {}
    for s, r in R['results'].items():
        tr = R['trials'].iloc[r['rows']]
        masks = subsets(tr, groups)
        for variant in VARIANTS:
            real, null = r[variant]['real'], r[variant]['null']
            usable = ~np.isnan(real['ridge_target'])
            sparse = usable.mean(0) < MIN_FRACTION                       # windows where most of the subject's gaze is missing
            for name, mk in masks.items():
                drop = sparse | (usable[mk].sum(0) < MIN_SUBSET)
                for metric in DECODED:
                    with np.errstate(invalid='ignore'):
                        d = np.nanmean(real[metric][mk], 0) - np.nanmean(null[metric][mk], 0)
                    d[drop] = np.nan
                    TC.setdefault((variant, metric, name), []).append(d)
            for key, v in uncertainty_corr(tr, r[variant], groups).items():
                mk = np.ones(len(tr), bool) if key[1] == 'All' else tr['TMS'].values == key[1]
                v[sparse | (usable[mk].sum(0) < MIN_SUBSET)] = np.nan
                RHO.setdefault((variant, *key), []).append(v)
    return R['centres'], {k: np.array(v) for k, v in TC.items()}, {k: np.array(v) for k, v in RHO.items()}, groups


# ---------------------------------------------------------------------------
# Statistics and figures
# ---------------------------------------------------------------------------
WINDOWS = {'Early delay (0.6–2.2 s)': (600, 2200), 'Late delay (2.4–4.2 s)': (2400, 4200),
           'Whole delay (0.6–4.2 s)': (600, 4200)}
STUDY_LABELS = [('orig', 'Study 1'), ('rep', 'Replication')]
LABEL = {'ridge_target': 'Ridge fidelity', 'ridge_report': 'Ridge fidelity to the report',
         'mahal_target': 'Mahalanobis tuning', 'mahal_report': 'Mahalanobis tuning to the report',
         'tafkap_target': 'TAFKAP fidelity', 'tafkap_report': 'TAFKAP fidelity to the report',
         'tafkap_unc': 'TAFKAP posterior SD (deg)', 'tafkap_abserr': 'TAFKAP |decoding error| (deg)'}


def signflip_p(v, n_perm=10000, seed=0):
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return np.nan
    signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, len(v)))
    return max(np.mean(np.abs((signs * v).mean(1)) >= abs(v.mean())), 1 / n_perm)


def window_mask(x, lo, hi, involved):
    """Window centres in [lo, hi), minus 400-ms windows overlapping a pulse train of an involved TMS condition."""
    from gaze_figures import PULSE, PULSE_MASK
    m = (x >= lo) & (x < hi)
    for g in involved:
        if g in PULSE:
            a, b = PULSE[g] + PULSE_MASK[0], PULSE[g] + PULSE_MASK[1]
            m &= ~((x + HALF_WIN > a) & (x - HALF_WIN < b))
    return m


def contrasts(groups):
    """name -> list of (sign, subset) for every contrast reported."""
    c = {}
    for g in groups[1:]:
        c[f'{g} − No TMS'] = [(1, g), (-1, 'No TMS')]
        c[f'({g} − No TMS) · inPF − outPF'] = [(1, f'{g} · inPF'), (-1, f'{g} · outPF'),
                                                (-1, 'No TMS · inPF'), (1, 'No TMS · outPF')]
        for tag in ['Err', 'RT']:
            c[f'({g} − No TMS) · {tag} Q1 − Q4'] = [(1, f'{g} · {tag} Q1'), (-1, f'{g} · {tag} Q4'),
                                                     (-1, f'No TMS · {tag} Q1'), (1, f'No TMS · {tag} Q4')]
    for g in groups:
        c[f'{g} · inPF − outPF'] = [(1, f'{g} · inPF'), (-1, f'{g} · outPF')]
        for tag in ['Err', 'RT']:
            c[f'{g} · {tag} Q1 − Q4'] = [(1, f'{g} · {tag} Q1'), (-1, f'{g} · {tag} Q4')]
    for tag in ['Err', 'RT']:
        c[f'{tag} Q1 − Q4'] = [(1, f'{tag} Q1'), (-1, f'{tag} Q4')]
    c['Radial − Tangential'] = [(1, 'Radial'), (-1, 'Tangential')]
    return c


def involved_groups(names, groups):
    return [g for g in groups if any(g in n for n in names)]


def _summary(vals, **info):
    k = int(np.sum(~np.isnan(vals)))
    return dict(**info, N=k, Mean=np.nanmean(vals), SEM=np.nanstd(vals, ddof=1) / np.sqrt(k) if k > 1 else np.nan,
                p=signflip_p(vals))


def stats_table(TCs, RHOs):
    rows = []
    for st, lab in STUDY_LABELS:
        x, tc, groups = TCs[st]
        for (variant, metric, name), v in tc.items():
            for wname, (lo, hi) in WINDOWS.items():
                vals = np.nanmean(v[:, window_mask(x, lo, hi, involved_groups([name], groups))], 1)
                rows.append(_summary(vals, Study=lab, Variant=variant, Measure=metric, Subset=name, Type='vs null', Window=wname))
        for variant in VARIANTS:
            for metric in DECODED:
                for cname, terms in contrasts(groups).items():
                    d = sum(sgn * tc[(variant, metric, name)] for sgn, name in terms)
                    m_inv = involved_groups([t[1] for t in terms], groups)
                    for wname, (lo, hi) in WINDOWS.items():
                        vals = np.nanmean(d[:, window_mask(x, lo, hi, m_inv)], 1)
                        rows.append(_summary(vals, Study=lab, Variant=variant, Measure=metric, Subset=cname,
                                             Type='contrast', Window=wname))
        for variant in VARIANTS:                                         # is the decoded angle closer to the report than to the target?
            for dec in ['ridge', 'mahal', 'tafkap']:
                for name in ['All'] + list(groups) + [f'Err Q{i + 1}' for i in range(N_QUART)]:
                    d = tc[(variant, f'{dec}_report', name)] - tc[(variant, f'{dec}_target', name)]
                    for wname, (lo, hi) in WINDOWS.items():
                        vals = np.nanmean(d[:, window_mask(x, lo, hi, involved_groups([name], groups))], 1)
                        rows.append(_summary(vals, Study=lab, Variant=variant, Measure=f'{dec}: report − target', Subset=name,
                                             Type='contrast', Window=wname))
        rho = RHOs[st]
        for (variant, analysis, name), v in rho.items():
            series = [(name, v, [name])]
            if name not in ('All', 'No TMS'):
                series.append((f'{name} − No TMS', v - rho[(variant, analysis, 'No TMS')], [name]))
            for sname, d, inv in series:
                for wname, (lo, hi) in WINDOWS.items():
                    vals = np.nanmean(d[:, window_mask(x, lo, hi, involved_groups(inv, groups))], 1)
                    rows.append(_summary(vals, Study=lab, Variant=variant, Measure=f'TAFKAP r(posterior SD, {analysis})',
                                         Subset=sname, Type='real − null r' if sname == name else 'contrast', Window=wname))
    return pd.DataFrame(rows)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import panels as P
    from gaze_figures import cluster_test, decorate, tc_plot
    sns.set_theme(style='white', context='notebook')
    matplotlib.rcParams['pdf.fonttype'] = 42
    TCs, RHOs = {}, {}
    for st, _ in STUDY_LABELS:
        x, tc, rho, groups = time_courses(st)
        TCs[st], RHOs[st] = (x, tc, groups), rho
    clusters = []

    def bars(ax, x, data, color, slot, tag):
        """Cluster bars: positive clusters along the top of the panel, negative ones along the bottom."""
        for a, b, p in cluster_test(data):
            mean = np.nanmean(data[:, a:b + 1])
            clusters.append(dict(Panel=tag, window_centre_from=x[a], window_centre_to=x[b],
                                 data_from_ms=x[a] - HALF_WIN, data_to_ms=x[b] + HALF_WIN,
                                 direction='+' if mean > 0 else '−', mean=mean, p=p))
            if p < 0.05:
                y = 0.97 - 0.04 * slot if mean > 0 else 0.03 + 0.04 * slot
                ax.plot([x[a], x[b]], [y, y], color=color, lw=3, solid_capstyle='butt',
                        transform=ax.get_xaxis_transform(), clip_on=False)

    def tms_lines(ax, x, series, groups, lab, tag, ylab):
        """series: name -> subjects x windows (All + each TMS condition)."""
        tc_plot(ax, x, series['All'], '0.6', 'All trials')
        bars(ax, x, series['All'], '0.6', 0, f'{lab} | {tag} | All')
        for i, g in enumerate(groups):
            tc_plot(ax, x, series[g], P.TMS_LINE_PAL[g], P.TMS_SHORT[g])
            if g != 'No TMS':
                bars(ax, x, series[g] - series['No TMS'], P.TMS_LINE_PAL[g], i, f'{lab} | {tag} | {g} − No TMS')
        decorate(ax, groups, ylab, 0)

    def sel(tc, variant, metric, names):
        return {n: tc[(variant, metric, n)] for n in names}

    # GZ7: by TMS condition ------------------------------------------------------
    spec = [(v, m) for v in VARIANTS for m in ['ridge_target', 'mahal_target', 'tafkap_target']]
    fig, axs = plt.subplots(len(spec), 2, figsize=(15, 3.3 * len(spec)), sharex=True)
    for c, (st, lab) in enumerate(STUDY_LABELS):
        x, tc, groups = TCs[st]
        for r, (variant, metric) in enumerate(spec):
            ylab = LABEL[metric] + ' − null' + ('\n(radial/tangential\nremoved)' if variant == 'control' else '')
            tms_lines(axs[r, c], x, sel(tc, variant, metric, ['All'] + groups), groups, lab, f'{variant} {metric}', ylab)
        axs[0, c].set_title(f'{lab}: decoding by TMS condition', fontsize=11, fontweight='bold')
        axs[0, c].legend(fontsize=7, frameon=False, ncol=2, loc='upper left')
    fig.suptitle('Time-resolved orientation decoding from gaze (400-ms windows, x = window centre). '
                 'Bars (cluster p < .05; top = positive, bottom = negative): grey = all trials vs null, coloured = TMS − None', y=0.997)
    [a.set_xlabel('') for a in axs[:-1].ravel()]
    plt.tight_layout(); P.save(fig, OUT, 'GZ7_TimeResolved_TMS')

    # GZ8: inPF - outPF ------------------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(15, 9.9), sharex=True)
    for c, (st, lab) in enumerate(STUDY_LABELS):
        x, tc, groups = TCs[st]
        for r, metric in enumerate(['ridge_target', 'mahal_target', 'tafkap_target']):
            ax = axs[r, c]
            key = ('standard', metric)
            for i, g in enumerate(groups):
                d = tc[(*key, f'{g} · inPF')] - tc[(*key, f'{g} · outPF')]
                tc_plot(ax, x, d, P.TMS_LINE_PAL[g], P.TMS_SHORT[g])
                if g == 'No TMS':
                    bars(ax, x, d, '0.3', 0, f'{lab} | standard {metric} | No TMS · inPF − outPF')
                else:
                    base = tc[(*key, 'No TMS · inPF')] - tc[(*key, 'No TMS · outPF')]
                    bars(ax, x, d - base, P.TMS_LINE_PAL[g], i, f'{lab} | standard {metric} | ({g} − No TMS) · inPF − outPF')
            decorate(ax, groups, LABEL[metric] + '\ninPF − outPF', 0)
        axs[0, c].set_title(f'{lab}: stimulated (inPF) minus non-stimulated location', fontsize=11, fontweight='bold')
        axs[0, c].legend(fontsize=7, frameon=False, loc='upper left')
    fig.suptitle('Spatially specific TMS effects on gaze decoding (bars: dark = None in − out vs 0; '
                 'coloured = (in − out) under TMS vs None; cluster p < .05)', y=0.997)
    [a.set_xlabel('') for a in axs[:-1].ravel()]
    plt.tight_layout(); P.save(fig, OUT, 'GZ8_TimeResolved_inPF_outPF')

    # GZ9: memory accuracy -----------------------------------------------------------
    qpal = dict(zip(range(1, N_QUART + 1), sns.color_palette('viridis', N_QUART)))
    rows_q = [('ridge_target', 'Err'), ('mahal_target', 'Err'), ('tafkap_target', 'Err'), ('ridge_report', 'Err'),
              ('ridge_target', 'Err by TMS'), ('ridge_target', 'RT')]
    fig, axs = plt.subplots(len(rows_q), 2, figsize=(15, 3.3 * len(rows_q)), sharex=True)
    for c, (st, lab) in enumerate(STUDY_LABELS):
        x, tc, groups = TCs[st]
        for r, (metric, kind) in enumerate(rows_q):
            ax = axs[r, c]
            key = ('standard', metric)
            if kind == 'Err by TMS':
                for i, g in enumerate(groups):
                    d = tc[(*key, f'{g} · Err Q1')] - tc[(*key, f'{g} · Err Q{N_QUART}')]
                    tc_plot(ax, x, d, P.TMS_LINE_PAL[g], P.TMS_SHORT[g])
                    if g != 'No TMS':
                        base = tc[(*key, 'No TMS · Err Q1')] - tc[(*key, f'No TMS · Err Q{N_QUART}')]
                        bars(ax, x, d - base, P.TMS_LINE_PAL[g], i, f'{lab} | standard {metric} | ({g} − No TMS) · Err Q1 − Q4')
                decorate(ax, groups, LABEL[metric] + '\nQ1 − Q4 of |error|', 0)
                ax.legend(fontsize=7, frameon=False, loc='upper left')
                continue
            for q, col in qpal.items():
                note = {1: ' (most accurate)' if kind == 'Err' else ' (fastest)',
                        N_QUART: ' (least accurate)' if kind == 'Err' else ' (slowest)'}.get(q, '')
                tc_plot(ax, x, tc[(*key, f'{kind} Q{q}')], col, f'Q{q}{note}')
            bars(ax, x, tc[(*key, f'{kind} Q1')] - tc[(*key, f'{kind} Q{N_QUART}')], 'k', 0,
                 f'{lab} | standard {metric} | {kind} Q1 − Q4')
            decorate(ax, groups, LABEL[metric] + ' − null\n' + ('|error| quartiles' if kind == 'Err' else 'RT quartiles'), 0)
            if r in (0, len(rows_q) - 1):
                ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper left')
        axs[0, c].set_title(f'{lab}: decoding by memory accuracy', fontsize=11, fontweight='bold')
    fig.suptitle('Orientation decoding from gaze by memory accuracy (quartiles within subject x TMS x location x orientation; '
                 'black bar: Q1 − Q4, coloured: TMS − None, cluster p < .05)', y=0.997)
    [a.set_xlabel('') for a in axs[:-1].ravel()]
    plt.tight_layout(); P.save(fig, OUT, 'GZ9_TimeResolved_Behaviour')

    # GZ10: TAFKAP uncertainty --------------------------------------------------------
    fig, axs = plt.subplots(4, 2, figsize=(15, 13.2), sharex=True)
    for c, (st, lab) in enumerate(STUDY_LABELS):
        x, tc, groups = TCs[st]
        rho = RHOs[st]
        tms_lines(axs[0, c], x, sel(tc, 'standard', 'tafkap_unc', ['All'] + groups), groups, lab,
                  'standard tafkap_unc', 'TAFKAP posterior SD (deg)\nreal − null')
        axs[0, c].set_title(f'{lab}: TAFKAP trial-wise uncertainty', fontsize=11, fontweight='bold')
        axs[0, c].legend(fontsize=7, frameon=False, ncol=2, loc='lower left')
        for r, analysis in enumerate(UNC_ANALYSES[:3], start=1):
            series = {n: rho[('standard', analysis, n)] for n in ['All'] + groups}
            tms_lines(axs[r, c], x, series, groups, lab, f'TAFKAP r | {analysis}',
                      f'r(posterior SD,\n{analysis})\nreal − null')
    fig.suptitle('TAFKAP uncertainty: within-subject rank correlations (within TMS x location x orientation cells), '
                 'real − shuffled decoders; bars: cluster p < .05', y=0.997)
    [a.set_xlabel('') for a in axs[:-1].ravel()]
    plt.tight_layout(); P.save(fig, OUT, 'GZ10_TimeResolved_TAFKAP_Uncertainty')

    pd.DataFrame(clusters).to_csv(os.path.join(OUT, 'Stats_TimeResolved_clusters.csv'), index=False)
    stats_table(TCs, RHOs).to_csv(os.path.join(OUT, 'Stats_TimeResolved.csv'), index=False)


if __name__ == '__main__':
    if '--figures' in sys.argv:
        figures()
    else:
        for st in [a for a in sys.argv[1:] if not a.startswith('--')] or ['orig', 'rep']:
            main(st)
