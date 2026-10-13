"""
Continuous radial geometry for the bias-replication analyses.

Every trial gets its angle from the true radial axis at the stimulus location
(each subject's phosphene field sits at a different polar angle), instead of
the binary Radial/Tangential label.

  add_geometry(df)        per-trial geometry columns
  fit_anisotropy(sub)     von Mises + uniform mixture whose precision and bias
                          vary continuously with angle from the radial axis
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import i0e

def wrap90(x):
    return (x + 90) % 180 - 90


def add_geometry(df):
    """Adds:
    RadialAxis   orientation (deg, CCW from horizontal) of the line from fixation
                 through the stimulus. polang is screen-based (clockwise, y down).
    PhiRadial    signed angle of the target from the radial axis, [-90, 90)
    DistRadial   |PhiRadial|, 0 = radial, 90 = tangential
    Elevation    stimulus angle below the horizontal meridian
    Mirror       +1 in the left visual field, -1 in the right, so that after
                 multiplying, positive always means 'rotated toward vertical'
    PhiAligned / ReportAligned / ErrAligned   target, report and error in that frame
    GeoType      Radial if the target is closer to the radial than the tangential axis
    """
    df = df.copy()
    df['RadialAxis'] = (180 - df['polang']) % 180
    df['PhiRadial'] = wrap90(df['targOriActual'] - df['RadialAxis'])
    df['DistRadial'] = np.abs(df['PhiRadial'])
    c = df['targOriActual'] % 90
    df['DistCardinal'] = np.minimum(c, 90 - c)
    df['VF'] = np.where(df['targX'] > df['xCenter'], 'RVF', 'LVF')
    df['Mirror'] = np.where(df['VF'] == 'LVF', 1, -1)
    df['Elevation'] = np.abs(wrap90(df['polang']))
    df['PhiAligned'] = df['Mirror'] * df['PhiRadial']
    df['ReportAligned'] = df['Mirror'] * wrap90(df['finalOri'] - df['RadialAxis'])
    df['ErrAligned'] = df['Mirror'] * df['errOri_scaled']
    # Radial/tangential from the actual geometry (closer to the radial or the
    # tangential axis), independent of the category labels.
    df['GeoType'] = np.where(df['DistRadial'] < 45, 'Radial', 'Tangential')
    return df


# ---------------------------------------------------------------------------
# Continuous anisotropy model
# ---------------------------------------------------------------------------
PARAMS = ['b0', 'a', 'c', 'm', 'mc', 'g']
BOUNDS = [(-2, 7), (-3, 3), (-3, 3), (-15, 15), (-15, 15), (0, 0.5)]


def _nll(p, x, phi2, th4):
    b0, a, c, m, mc, g = p
    kappa = np.exp(b0 + a * np.cos(phi2) + c * np.cos(th4))
    mu = np.radians(2 * (m * np.sin(phi2) + mc * np.sin(th4)))
    vm = np.exp(kappa * (np.cos(x - mu) - 1)) / (2 * np.pi * i0e(kappa))
    return -np.sum(np.log((1 - g) * vm + g / (2 * np.pi) + 1e-300))


def fit_anisotropy(sub, n_starts=12, seed=0):
    """Errors are modelled on the doubled-angle circle (as modeller.fit_model):
      kappa_i = exp(b0 + a*cos(2*phi_i) + c*cos(4*theta_i))
      mu_i    = m*sin(2*phi_i) + mc*sin(4*theta_i)        (deg)
    phi = angle from the radial axis, theta = target orientation.
      a  > 0 : more precise near the radial axis (exp(2a) = kappa_radial / kappa_tangential)
      m  > 0 : reports pushed away from the radial axis (efficient-coding repulsion)
      c, mc  : the same for the cardinal axes (oblique effect / cardinal repulsion)
    """
    x = np.radians(2 * sub['errOri_scaled'].values)
    phi2 = np.radians(2 * sub['PhiRadial'].values)
    th4 = np.radians(4 * sub['targOriActual'].values)
    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_starts):
        p0 = [np.log(rng.uniform(5, 40)), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
              rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0, 0.1)]
        r = minimize(_nll, p0, args=(x, phi2, th4), method='L-BFGS-B', bounds=BOUNDS)
        if r.success and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        return None
    out = dict(zip(PARAMS, best.x))
    out['nll'] = best.fun
    out['n'] = len(sub)
    return out


def anisotropy_table(data, group_cols, min_trials=40):
    rows = []
    for keys, sub in data.groupby(group_cols, observed=True):
        if len(sub) < min_trials: continue
        fit = fit_anisotropy(sub)
        if fit is not None:
            rows.append({**dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))), **fit})
    return pd.DataFrame(rows)
