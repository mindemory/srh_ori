"""
gaze_epochs.py

Builds trial epochs of iEye-preprocessed eye data for Study 1 and the bias
replication, aligned to grating onset and matched to the behavioural trials.

Input per block: analysis/subXX/dayYY/EyeData/blockZZ/<edf>.mat (ii_data,
ii_cfg from run_iipreproc / run_iipreproc_noplots).

Per trial (window -500..+4250 ms around grating onset = first XDAT 10/11
sample; the response starts at +4250 ms):
  X, Y      gaze (deg; +x right, +y up), 10-ms bin means, NaN where >50% of
            the bin is missing (blinks / off-screen censored by iEye). iEye's
            per-trial drift correction is a constant offset and drops out of
            the per-trial baseline used downstream.
  Pupil     iEye PupilZ (z-scored within block, NaN in blinks), 10-ms bins
  valid     fraction of valid gaze samples per bin
  pulse_ms  TMS pulse time (onset of XDAT 3 = end of delay 1), NaN if none
  saccades  two detections within the window, flagged by 'method':
            'iEye' - iEye's saccades (velocity 30 deg/s, >= 7.5 ms, >= 0.25 deg)
            'EK'   - velocity-threshold detection on the 1 kHz gaze as in
                     Liu, Nobre & van Ede (2022): 7-ms-smoothed speed,
                     onset > 5 x trial median speed, 100-ms refractory,
                     size/direction from positions 50 ms before vs
                     50-100 ms after onset (microsaccades = < 1 deg)
            columns: onset/offset (ms), amplitude (deg), direction (deg,
            0 = rightward, counter-clockwise), start/end position

Usage: python gaze_epochs.py [orig|rep]   -> <root>/analysis/gaze_epochs_<study>.npz
                                             + gaze_saccades_<study>.csv
"""
import os
import re
import sys
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat

from biasrep_common import STUDIES, ORIG_ROOT, DATA_ROOT

warnings.filterwarnings('ignore')
PRE, POST, BIN = 500, 4250, 10                   # ms
N_BINS = (PRE + POST) // BIN
TIMES = -PRE + BIN / 2 + BIN * np.arange(N_BINS)  # bin centres, ms re grating onset
ROOTS = {'orig': ORIG_ROOT, 'rep': DATA_ROOT}


def _bin(x, valid=None):
    x = x[:N_BINS * BIN].reshape(N_BINS, BIN)
    v = ~np.isnan(x)
    with np.errstate(invalid='ignore'):
        m = np.nanmean(x, axis=1)
    m[v.mean(axis=1) < 0.5] = np.nan
    return m, v.mean(axis=1)


VEL_MULT, REFRACTORY, SMOOTH_MS = 5, 100, 7     # x median speed, ms, ms


def ek_saccades(x, y):
    """Saccade/microsaccade onsets as in Liu, Nobre & van Ede (2022, Nat Commun)
    and de Vries & van Ede (2024): gaze speed smoothed with a 7-ms Gaussian
    window, onset = first sample above VEL_MULT x the trial's median speed,
    100-ms refractory period; size and direction from the mean position
    50-0 ms before onset vs 50-100 ms after. Returns (onset, landing) indices."""
    n = len(x)
    if n < 200:
        return []
    speed = np.full(n, np.nan)
    speed[1:] = np.hypot(np.diff(x), np.diff(y)) * 1000
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.4) ** 2); k /= k.sum()       # 7-sample Gaussian (MATLAB smoothdata 'gaussian', 7)
    sm = np.convolve(np.nan_to_num(speed, nan=0.0), k, mode='same')
    valid = np.convolve((~np.isnan(speed)).astype(float), k, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        sm = np.where(valid > 0.99, sm / valid, np.nan)
    thr = VEL_MULT * np.nanmedian(sm)
    above = np.nan_to_num(sm > thr)
    onsets, last = [], -np.inf
    for i in np.where(above[1:] & ~above[:-1])[0] + 1:
        if i - last >= REFRACTORY and i >= 50 and i + 100 <= n:
            onsets.append(i); last = i
    return [(i, i + 75) for i in onsets]


def block_epochs(fn):
    m = loadmat(fn, squeeze_me=True, struct_as_record=False)
    d, c = m['ii_data'], m['ii_cfg']
    X, Y = np.asarray(d.X, float), np.asarray(d.Y, float)
    Xs, Ys = np.asarray(d.X_smooth, float), np.asarray(d.Y_smooth, float)
    P = np.asarray(d.PupilZ, float)
    XD, tv = np.asarray(d.XDAT).astype(int), np.asarray(c.trialvec).astype(int)
    sacc = np.atleast_2d(np.asarray(c.saccades)).astype(int) - 1 if np.size(c.saccades) else np.zeros((0, 2), int)
    out, sac_rows = [], []
    for t in np.unique(tv[tv > 0]):
        idx = np.where(tv == t)[0]
        on = idx[np.isin(XD[idx], [10, 11])]
        if len(on) == 0:
            continue
        on = on[0]
        a, b = on - PRE, on + POST
        if a < 0 or b > len(X):
            continue
        x, vx = _bin(X[a:b]); y, _ = _bin(Y[a:b]); p, _ = _bin(P[a:b])
        d3 = idx[XD[idx] == 3]
        resp = idx[XD[idx] == 4]
        out.append({'trial': int(t), 'X': x, 'Y': y, 'Pupil': p, 'valid': vx,
                    'pulse_ms': float(d3[0] - on) if len(d3) else np.nan,
                    'resp_ms': float(resp[0] - on) if len(resp) else np.nan})
        for s0, s1 in sacc[(sacc[:, 0] >= a) & (sacc[:, 0] < b)]:
            s1 = min(s1, len(Xs) - 1)
            dx, dy = Xs[s1] - Xs[s0], Ys[s1] - Ys[s0]
            sac_rows.append({'trial': int(t), 'method': 'iEye', 'onset_ms': s0 - on, 'offset_ms': s1 - on,
                             'amp': float(np.hypot(dx, dy)), 'dir': float(np.degrees(np.arctan2(dy, dx)) % 360),
                             'x0': Xs[s0], 'y0': Ys[s0], 'x1': Xs[s1], 'y1': Ys[s1]})
        xw, yw = X[a:b], Y[a:b]
        for s0, _ in ek_saccades(xw, yw):
            with np.errstate(invalid='ignore'):
                x0, y0 = np.nanmean(xw[s0 - 50:s0]), np.nanmean(yw[s0 - 50:s0])
                x1, y1 = np.nanmean(xw[s0 + 50:s0 + 100]), np.nanmean(yw[s0 + 50:s0 + 100])
            dx, dy = x1 - x0, y1 - y0
            if np.isnan(dx) or np.isnan(dy):
                continue
            sac_rows.append({'trial': int(t), 'method': 'EK', 'onset_ms': s0 - PRE, 'offset_ms': s0 - PRE + 75,
                             'amp': float(np.hypot(dx, dy)), 'dir': float(np.degrees(np.arctan2(dy, dx)) % 360),
                             'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
    return out, sac_rows


def build(study):
    root = ROOTS[study]
    beh = pd.read_pickle(STUDIES[study]['pkl'])
    beh['trial'] = beh.groupby(['subjID', 'day', 'block']).cumcount() + 1
    rows, arrays, sacs = [], {k: [] for k in ['X', 'Y', 'Pupil', 'valid']}, []
    for sub_dir in sorted(glob.glob(os.path.join(root, 'analysis', 'sub*'))):
        s = int(re.findall(r'sub(\d+)$', sub_dir)[0]) if re.search(r'sub(\d+)$', sub_dir) else None
        if s is None or s not in set(beh['subjID']):
            continue
        for blk_dir in sorted(glob.glob(os.path.join(sub_dir, 'day*', 'EyeData', 'block*'))):
            day = int(re.findall(r'day(\d+)', blk_dir)[0]); block = int(re.findall(r'block(\d+)$', blk_dir)[0])
            fns = [f for f in glob.glob(os.path.join(blk_dir, '*.mat')) if re.fullmatch(r'\d+\.mat', os.path.basename(f))]
            if not fns:
                continue
            ep, sac = block_epochs(fns[0])
            for e in ep:
                rows.append({'subjID': s, 'day': day, 'block': block, 'trial': e['trial'],
                             'pulse_ms': e['pulse_ms'], 'resp_ms': e['resp_ms']})
                for k in arrays:
                    arrays[k].append(e[k])
            for r in sac:
                sacs.append({'subjID': s, 'day': day, 'block': block, **r})
        print(f'  [{study}] sub{s:02d}: {sum(r["subjID"] == s for r in rows)} trials')
    meta = pd.DataFrame(rows)
    keep = beh[['subjID', 'day', 'block', 'trial']].assign(beh_row=np.arange(len(beh)))
    meta = meta.reset_index().merge(keep, on=['subjID', 'day', 'block', 'trial'], how='inner').sort_values('index')
    sel = meta['index'].values
    out = os.path.join(root, 'analysis', f'gaze_epochs_{study}.npz')
    np.savez_compressed(out, times=TIMES, **{k: np.stack(v)[sel] for k, v in arrays.items()},
                        **{f'meta_{c}': meta[c].values for c in ['subjID', 'day', 'block', 'trial', 'pulse_ms', 'resp_ms', 'beh_row']})
    sac = pd.DataFrame(sacs)
    sac.to_csv(os.path.join(root, 'analysis', f'gaze_saccades_{study}.csv'), index=False)
    print(f'[{study}] {len(meta)} trials matched to behaviour (of {len(rows)} eye trials, {len(beh)} behavioural); '
          f'{len(sac)} saccades -> {out}')


if __name__ == '__main__':
    for st in (sys.argv[1:] or ['orig', 'rep']):
        build(st)
