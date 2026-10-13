"""
gaze_decoder_zoo.py   (srh conda env)

Compares many orientation decoders on the same gaze data, trials, folds and
nulls, in Study 1 and the replication. Uses the baseline-corrected epochs and
trial selection saved by gaze_decoding.py.

Per subject, per time window and feature set, every decoder is cross-validated
with GroupKFold over blocks after removing each stimulus location's mean
(training folds only), and compared with the same decoder trained on
orientations shuffled within block x location (N_SHUFFLE times).

Feature sets
  gaze_mean   window-mean x, y, x^2 - y^2, 2xy           (4 features)
  gaze_bins   x, y in 100-ms bins across the window      (2 x bins)
  pupil_bins  pupil in 100-ms bins (control)             (bins)

Decoders
  continuous (target: doubled angle 2*theta; output: decoded angle; score:
  fidelity = mean cos 2(decoded - true))
    Ridge             ridge regression to [cos 2theta, sin 2theta]
    Kernel ridge      RBF kernel ridge to [cos 2theta, sin 2theta]
    Random forest     random-forest regression to [cos 2theta, sin 2theta]
    von Mises (JAX)   von Mises regression: linear predictor z = (zc, zs), mean
                      direction atan2(zs, zc), concentration |z|; maximum
                      likelihood with an L2 penalty, gradients from JAX
    IEM               inverted encoding model, 6 half-wave-rectified cos^5
                      channels over orientation, ridge-regularised inversion
  distance tuning (score: cosine-weighted, mean-centred distance to the
  training mean of each orientation category; > 0 = information)
    Mahalanobis       Ledoit-Wolf covariance (Wolff et al. 2017)
    Euclidean         as single-trial gaze RSA (Linde-Domingo & Spitzer 2024)
  categorical (target: orientation category within location; scores: balanced
  accuracy, and fidelity of the predicted category's angle)
    LDA, Logistic, Linear SVM, RBF SVM, Naive Bayes, Gradient boosting

Each decoder's score is expressed as z = (real - mean(null)) / sd(null) per
subject (and as real - mean(null)), then tested across subjects.

Radial/tangential control (--geocontrol): features are additionally centred
within radial and within tangential targets at each location (training folds),
so only orientation differences inside each geometric class can be decoded
(Study 1: 30 vs 60 and 120 vs 150; replication: the +-24 deg jitter only);
the null then shuffles orientations within block x location x radial/tangential.

Output: <COMPARE_DIR>/gaze/zoo[_geo]_<study>.pkl, Stats_DecoderZoo[_GeoControl].csv,
        Stats_DecoderAveraged.csv (z averaged over decoders within subject),
        GZ5_DecoderZoo.png / GZ6_DecoderZoo_GeoControl.png
Usage: python gaze_decoder_zoo.py [--geocontrol] [orig|rep ...]
       python gaze_decoder_zoo.py --figures [--geocontrol]
"""
import os
import sys
import time
import pickle
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

from biasrep_common import COMPARE_DIR

warnings.filterwarnings('ignore')
OUT = os.path.join(COMPARE_DIR, 'gaze')
N_SHUFFLE = 10
N_FOLDS = 5
MIN_TRIALS = 40
BIN = 100
WINDOWS = {'Encoding (0.1–0.6 s)': (100, 600), 'Early delay (0.6–2.2 s)': (600, 2200),
           'Late delay (2.4–4.2 s)': (2400, 4200), 'Whole delay (0.6–4.2 s)': (600, 4200)}
FSETS = ['gaze_mean', 'gaze_bins', 'pupil_bins']
ALPHAS = np.logspace(-2, 4, 13)

# ---------------------------------------------------------------------------
# von Mises regression in JAX (fixed padded shapes -> compiled once)
# ---------------------------------------------------------------------------
import jax
import jax.numpy as jnp
from jax.scipy.special import i0e
jax.config.update('jax_enable_x64', True)
N_PAD, F_PAD, VM_LAMBDA = 900, 80, 1e-2


def _vm_loss(params, X, c2, s2, w):
    Wc, Ws = params[:F_PAD], params[F_PAD:2 * F_PAD]
    zc = X @ Wc + params[-2]
    zs = X @ Ws + params[-1]
    kappa = jnp.sqrt(zc ** 2 + zs ** 2 + 1e-12)
    loglik = zc * c2 + zs * s2 - (jnp.log(i0e(kappa)) + kappa)
    return -jnp.sum(w * loglik) / jnp.sum(w) + VM_LAMBDA * (jnp.sum(Wc ** 2) + jnp.sum(Ws ** 2))


_vm_value_grad = jax.jit(jax.value_and_grad(_vm_loss))


class VonMisesRegression:
    def fit(self, X, theta):
        n, f = X.shape
        Xp = np.zeros((N_PAD, F_PAD)); Xp[:n, :f] = X
        w = np.zeros(N_PAD); w[:n] = 1
        c2 = np.zeros(N_PAD); s2 = np.zeros(N_PAD)
        c2[:n], s2[:n] = np.cos(np.radians(2 * theta)), np.sin(np.radians(2 * theta))
        args = tuple(jnp.asarray(a) for a in (Xp, c2, s2, w))

        def fun(p):
            v, g = _vm_value_grad(jnp.asarray(p), *args)
            return float(v), np.asarray(g, dtype=float)
        res = minimize(fun, np.zeros(2 * F_PAD + 2), jac=True, method='L-BFGS-B')
        self.p, self.f = res.x, f
        return self

    def predict_angle(self, X):
        Wc, Ws = self.p[:self.f], self.p[F_PAD:F_PAD + self.f]
        return np.degrees(np.arctan2(X @ Ws + self.p[-1], X @ Wc + self.p[-2])) / 2 % 180


# ---------------------------------------------------------------------------
# Inverted encoding model
# ---------------------------------------------------------------------------
IEM_CENTRES = np.arange(0, 180, 30)


def _iem_basis(theta):
    d = np.radians((theta[:, None] - IEM_CENTRES[None, :] + 90) % 180 - 90)   # orientation difference, wrapped to [-90, 90)
    return np.cos(d) ** 5


class IEM:
    def fit(self, X, theta, lam=1.0):
        C = _iem_basis(theta)
        self.W = np.linalg.lstsq(C, X, rcond=None)[0]                  # channels x features
        self.lam = lam
        return self

    def predict_angle(self, X):
        W = self.W
        Chat = X @ W.T @ np.linalg.inv(W @ W.T + self.lam * np.eye(len(W)))
        z = Chat @ np.exp(1j * np.radians(2 * IEM_CENTRES))
        return np.degrees(np.angle(z)) / 2 % 180


# ---------------------------------------------------------------------------
# Decoder registry
# ---------------------------------------------------------------------------
def _cossin(theta):
    return np.column_stack([np.cos(np.radians(2 * theta)), np.sin(np.radians(2 * theta))])


def _angle(cs):
    return np.degrees(np.arctan2(cs[:, 1], cs[:, 0])) / 2 % 180


CONTINUOUS = {
    'Ridge': lambda A, th, B: _angle(RidgeCV(alphas=ALPHAS).fit(A, _cossin(th)).predict(B)),
    'Kernel ridge (RBF)': lambda A, th, B: _angle(KernelRidge(kernel='rbf', alpha=1.0, gamma=1.0 / A.shape[1]).fit(A, _cossin(th)).predict(B)),
    'Random forest': lambda A, th, B: _angle(RandomForestRegressor(n_estimators=60, max_depth=5, min_samples_leaf=5,
                                                                   random_state=0).fit(A, _cossin(th)).predict(B)),
    'von Mises (JAX)': lambda A, th, B: VonMisesRegression().fit(A, th).predict_angle(B),
    'IEM (6 channels)': lambda A, th, B: IEM().fit(A, th).predict_angle(B),
}
CATEGORICAL = {
    'LDA (shrinkage)': lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    'Logistic (L2)': lambda: LogisticRegression(C=1.0, max_iter=2000),
    'Linear SVM': lambda: LinearSVC(C=0.1, max_iter=5000),
    'RBF SVM': lambda: SVC(kernel='rbf', C=1.0, gamma='scale'),
    'Naive Bayes': lambda: GaussianNB(),
    'Gradient boosting': lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=60, learning_rate=0.1, random_state=0),
}
DISTANCE = ['Mahalanobis', 'Euclidean']


def _distance_tuning(A, cat_tr, th_tr, B, th_te, metric):
    cats = np.unique(cat_tr)
    means = np.stack([A[cat_tr == c].mean(0) for c in cats])
    centre = np.array([np.degrees(np.angle(np.exp(1j * np.radians(2 * th_tr[cat_tr == c])).mean())) / 2 % 180 for c in cats])
    if metric == 'Mahalanobis':
        resid = A - means[np.searchsorted(cats, cat_tr)]
        P = LedoitWolf().fit(resid).precision_
        diff = B[:, None, :] - means[None, :, :]
        d = np.sqrt(np.einsum('ncf,fg,ncg->nc', diff, P, diff))
    else:
        d = np.linalg.norm(B[:, None, :] - means[None, :, :], axis=2)
    d = d - d.mean(1, keepdims=True)
    w = np.cos(np.radians(2 * (centre[None, :] - th_te[:, None])))
    return -(w * d).mean(1)                                                 # per-trial tuning accuracy


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(study):
    with open(os.path.join(OUT, f'decoding_{study}.pkl'), 'rb') as fh:
        R = pickle.load(fh)
    t = R['t']
    edges = np.arange(-400, 4250, BIN)
    def binned(a):
        out = np.full((len(a), len(edges) - 1), np.nan)
        for i in range(len(edges) - 1):
            m = (t >= edges[i]) & (t < edges[i + 1])
            with np.errstate(invalid='ignore'):
                v = np.nanmean(a[:, m], axis=1)
            v[np.isnan(a[:, m]).mean(1) > 0.5] = np.nan
            out[:, i] = v
        return out
    return dict(trials=R['trials'], centres=edges[:-1] + BIN / 2, X=binned(R['X']), Y=binned(R['Y']), P=binned(R['P']))


def features(D, rows, window, fset):
    lo, hi = window
    m = (D['centres'] >= lo) & (D['centres'] < hi)
    x, y, p = D['X'][rows][:, m], D['Y'][rows][:, m], D['P'][rows][:, m]
    if fset == 'gaze_mean':
        with np.errstate(invalid='ignore'):
            xm, ym = np.nanmean(x, 1), np.nanmean(y, 1)
        F = np.column_stack([xm, ym, xm ** 2 - ym ** 2, 2 * xm * ym])
        ok = ~np.isnan(F).any(1)
    else:
        F = np.column_stack([x, y]) if fset == 'gaze_bins' else p
        ok = np.isnan(F).mean(1) <= 0.5
    return F, ok


# ---------------------------------------------------------------------------
# One subject
# ---------------------------------------------------------------------------
def _cv_all(F, theta, cat, loc, groups, cont=None, categ=None, dist=None):
    """Cross-validated outputs of every decoder for one label assignment.
    loc: centring groups (stimulus location, or location x radial/tangential)."""
    cont = CONTINUOUS if cont is None else cont
    categ = CATEGORICAL if categ is None else categ
    dist = DISTANCE if dist is None else dist
    n = len(theta)
    ang = {k: np.full(n, np.nan) for k in cont}
    cls = {k: np.full(n, np.nan) for k in categ}
    tun = {k: np.full(n, np.nan) for k in dist}
    for tr, te in GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups)))).split(F, groups=groups):
        A, B = F[tr].copy(), F[te].copy()
        for v in np.unique(loc[tr]):                                   # remove location means (training folds only)
            with np.errstate(invalid='ignore'):
                mu = np.nanmean(A[loc[tr] == v], 0)
            A[loc[tr] == v] -= mu; B[loc[te] == v] -= mu
        A, B = np.nan_to_num(A), np.nan_to_num(B)                       # remaining gaps at the (centred) mean
        sd = A.std(0) + 1e-9
        A, B = A / sd, B / sd
        for k, f in cont.items():
            ang[k][te] = f(A, theta[tr], B)
        for k, mk in categ.items():
            cls[k][te] = mk().fit(A, cat[tr]).predict(B)
        for k in dist:
            tun[k][te] = _distance_tuning(A, cat[tr], theta[tr], B, theta[te], k)
    return ang, cls, tun


def _scores(ang, cls, tun, theta, cat, centre_of):
    s = {}
    for k, a in ang.items():
        s[(k, 'fidelity')] = np.nanmean(np.cos(np.radians(2 * (a - theta))))
    for k, c in cls.items():
        s[(k, 'balanced accuracy')] = balanced_accuracy_score(cat, c)
        s[(k, 'fidelity')] = np.nanmean(np.cos(np.radians(2 * (np.vectorize(centre_of.get)(c) - theta))))
    for k, v in tun.items():
        s[(k, 'tuning')] = np.nanmean(v)
    return s


GEO_CONT = ['Ridge', 'von Mises (JAX)', 'IEM (6 channels)']
GEO_CATEG = ['LDA (shrinkage)', 'Linear SVM']
GEO_WINDOWS = ['Early delay (0.6–2.2 s)', 'Late delay (2.4–4.2 s)', 'Whole delay (0.6–4.2 s)']


def run_subject(s, D, geo=False):
    tr_all = D['trials']
    rows_all = np.where(tr_all['subjID'].values == s)[0]
    tr = tr_all.iloc[rows_all]
    rng = np.random.default_rng(int(s))
    out = {}
    sel = dict(cont={k: CONTINUOUS[k] for k in GEO_CONT}, categ={k: CATEGORICAL[k] for k in GEO_CATEG}) if geo else {}
    for wname, window in WINDOWS.items():
        if geo and wname not in GEO_WINDOWS:
            continue
        for fset in FSETS:
            if geo and fset == 'pupil_bins':
                continue
            F, ok = features(D, rows_all, window, fset)
            if ok.sum() < MIN_TRIALS:
                continue
            sub = tr[ok]
            theta = sub['targOriActual'].values.astype(float)
            cat = sub['OriCat'].values.astype(int)
            loc = (sub['VF'].values == 'RVF').astype(int)
            groups = (sub['day'].astype(str) + '_' + sub['block'].astype(str)).values
            centre_of = {c: c for c in np.unique(cat)}
            centring = loc * 2 + (sub['GeoType'].values == 'Radial') if geo else loc
            # null: shuffle within block x location (x radial/tangential in the control, so the
            # null shares the centring structure and only within-class orientation is permuted)
            keys = groups + '_' + centring.astype(str)
            if geo and min(np.bincount(centring)) < 5:
                continue
            real = _scores(*_cv_all(F[ok], theta, cat, centring, groups, **sel), theta, cat, centre_of)
            null = {k: [] for k in real}
            for _ in range(N_SHUFFLE):
                perm = np.arange(len(theta))
                for k in np.unique(keys):
                    m = np.where(keys == k)[0]; perm[m] = rng.permutation(m)
                th_s, cat_s = theta[perm], cat[perm]
                for k, v in _scores(*_cv_all(F[ok], th_s, cat_s, centring, groups, **sel), th_s, cat_s, centre_of).items():
                    null[k].append(v)
            for k in real:
                out[(wname, fset, *k)] = (real[k], np.array(null[k]))
    return s, out


def main(study, geo=False):
    D = load(study)
    subs = np.unique(D['trials']['subjID'])
    t0 = time.time()
    res = Parallel(n_jobs=6)(delayed(run_subject)(s, D, geo) for s in subs)
    with open(os.path.join(OUT, f'zoo{"_geo" if geo else ""}_{study}.pkl'), 'wb') as fh:
        pickle.dump(dict(res), fh)
    print(f'[{study}] {len(subs)} subjects in {time.time() - t0:.0f} s')


# ---------------------------------------------------------------------------
# Group statistics and figure
# ---------------------------------------------------------------------------
def group_stats(geo=False):
    rows = []
    for st, lab in [('orig', 'Study 1'), ('rep', 'Replication')]:
        p = os.path.join(OUT, f'zoo{"_geo" if geo else ""}_{st}.pkl')
        if not os.path.exists(p):
            continue
        Z = pickle.load(open(p, 'rb'))
        keys = sorted({k for v in Z.values() for k in v})
        for key in keys:
            diffs, zs = [], []
            for s, v in Z.items():
                if key not in v: continue
                real, null = v[key]
                diffs.append(real - null.mean()); zs.append((real - null.mean()) / (null.std(ddof=1) + 1e-12))
            diffs, zs = np.array(diffs), np.array(zs)
            rows.append({'Study': lab, 'Window': key[0], 'Features': key[1], 'Decoder': key[2], 'Metric': key[3],
                         'N': len(zs), 'real_minus_null': diffs.mean(), 'SEM': stats.sem(diffs),
                         'mean_z': zs.mean(), 't': stats.ttest_1samp(zs, 0).statistic, 'p': stats.ttest_1samp(zs, 0).pvalue})
    return pd.DataFrame(rows)


def decoder_averaged():
    """Per subject, z vs own null averaged over all decoders (primary metric each), then tested
    across subjects; for the standard and the radial/tangential-control runs."""
    rows = []
    for tag, lab_tag in [('', 'All decoders, no control'), ('_geo', 'Radial/tangential control')]:
        for st, lab in [('orig', 'Study 1'), ('rep', 'Replication')]:
            p = os.path.join(OUT, f'zoo{tag}_{st}.pkl')
            if not os.path.exists(p):
                continue
            Z = pickle.load(open(p, 'rb'))
            recs = []
            for s, v in Z.items():
                for (win, fset, dec, met), (real, null) in v.items():
                    primary = (met == 'balanced accuracy') if dec in CATEGORICAL else (met in ('fidelity', 'tuning'))
                    if primary:
                        recs.append({'subj': s, 'Window': win.split(' (')[0], 'Features': fset,
                                     'z': (real - null.mean()) / null.std(ddof=1)})
            d = pd.DataFrame(recs).groupby(['Features', 'Window', 'subj'])['z'].mean().reset_index()
            for (f, w), g in d.groupby(['Features', 'Window']):
                t = stats.ttest_1samp(g['z'], 0)
                rows.append({'Analysis': lab_tag, 'Study': lab, 'Features': f, 'Window': w, 'N': len(g),
                             'decoder-averaged z': g['z'].mean(), 'SEM': g['z'].sem(), 't': t.statistic, 'p': t.pvalue})
    return pd.DataFrame(rows)


def figure(st, geo=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import panels as P
    primary = st[st['Metric'].isin(['fidelity', 'tuning']) & ~st['Decoder'].isin(list(CATEGORICAL))]
    primary = pd.concat([primary, st[st['Decoder'].isin(list(CATEGORICAL)) & (st['Metric'] == 'balanced accuracy')]])
    order = (GEO_CONT + DISTANCE + GEO_CATEG) if geo else (list(CONTINUOUS) + DISTANCE + list(CATEGORICAL))
    wins = GEO_WINDOWS if geo else list(WINDOWS)
    fsets = ['gaze_mean', 'gaze_bins'] if geo else FSETS
    fig, axs = plt.subplots(len(fsets), 2, figsize=(16, 5.2 * len(fsets)), squeeze=False)
    for c, study in enumerate(['Study 1', 'Replication']):
        for r, fset in enumerate(fsets):
            ax = axs[r, c]
            sub = primary[(primary['Study'] == study) & (primary['Features'] == fset)]
            if sub.empty:
                ax.axis('off'); continue
            M = sub.pivot_table(index='Decoder', columns='Window', values='mean_z').reindex(index=order, columns=wins)
            Pv = sub.pivot_table(index='Decoder', columns='Window', values='p').reindex(index=order, columns=wins)
            annot = M.round(2).astype(str) + Pv.map(lambda p: '\n*' if p < .05 else ('\n·' if p < .1 else ''))
            sns.heatmap(M, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=annot, fmt='', annot_kws={'fontsize': 7},
                        cbar_kws={'label': 'mean z vs own shuffle null'})
            ax.set_title(f'{study} · {fset}', fontsize=11, fontweight='bold'); ax.set_xlabel(''); ax.set_ylabel('')
            ax.set_xticklabels([w.replace(' (', '\n(') for w in wins], rotation=0, fontsize=8)
            n_cont = len(GEO_CONT) if geo else len(CONTINUOUS)
            ax.axhline(n_cont, color='k', lw=1); ax.axhline(n_cont + len(DISTANCE), color='k', lw=1)
    head = ('Radial/tangential control: features centred within radial and within tangential targets at each location\n'
            if geo else 'Orientation decoding from eye data: every decoder vs its own block-preserving shuffle null\n')
    fig.suptitle(head + '(rows: continuous | distance tuning | categorical; * p < .05, · p < .1 across subjects, uncorrected)', y=0.995)
    plt.tight_layout()
    P.save(fig, OUT, 'GZ6_DecoderZoo_GeoControl' if geo else 'GZ5_DecoderZoo')


if __name__ == '__main__':
    geo = '--geocontrol' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--figures' in sys.argv:
        st = group_stats(geo)
        st.to_csv(os.path.join(OUT, f'Stats_DecoderZoo{"_GeoControl" if geo else ""}.csv'), index=False)
        figure(st, geo)
        decoder_averaged().to_csv(os.path.join(OUT, 'Stats_DecoderAveraged.csv'), index=False)
    else:
        for study in (args or ['orig', 'rep']):
            main(study, geo)
