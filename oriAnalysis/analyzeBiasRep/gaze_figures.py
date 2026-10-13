"""
gaze_figures.py   (srh conda env)

Statistics and figures for the gaze analyses of Study 1 and the replication,
from gaze_epochs.py (epochs, saccades) and gaze_decoding.py (decoders).

All time courses: per-subject means, then mean +- SEM across subjects; tests
are sign-flip cluster-based permutation tests across subjects (cluster
threshold p < .05 two-sided t, cluster mass, 5000 permutations).

Figures (in <COMPARE_DIR>/gaze):
  GZ1_DataQuality_TMS    missing data (blinks), saccade rate, microsaccade rate,
                         pupil and gaze speed, by TMS condition; pulse times marked
  GZ2_Location           gaze 'towardness' to the remembered stimulus, location
                         decoding (AUC), microsaccades toward vs away
  GZ3_Orientation        orientation decoding (gaze, pupil) minus shuffle null;
                         microsaccades aligned vs orthogonal to the grating axis
  GZ4_TMS                decoding and gaze measures by TMS condition and by
                         inPF / outPF, averaged over early / late delay windows
                         (pulse windows excluded)
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
from sklearn.metrics import roc_auc_score

from biasrep_common import COMPARE_DIR, ORIG_ROOT, DATA_ROOT, STUDIES
import panels as P

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
OUT = os.path.join(COMPARE_DIR, 'gaze')
ROOTS = {'orig': ORIG_ROOT, 'rep': DATA_ROOT}
PULSE = {'Early TMS': 250, 'Middle TMS': 2250}       # ms after grating onset
PULSE_MASK = (-50, 400)                               # ms around the pulse train, excluded from window averages
WINDOWS = {'Encoding\n(0.1–0.6 s)': (100, 600), 'Early delay\n(0.6–2.2 s)': (600, 2200), 'Late delay\n(2.4–4.2 s)': (2400, 4200)}
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def cluster_test(data, n_perm=5000, alpha=0.05):
    """data: subjects x time (effect, null = 0). Returns list of (start, stop, p) clusters."""
    data = data[~np.all(np.isnan(data), axis=1)]
    n = len(data)
    if n < 4:
        return []
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)

    def tvals(x):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.nanmean(x, 0) / (np.nanstd(x, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(x), 0)))

    def clusters(tv):
        out, cur = [], None
        for i, v in enumerate(np.nan_to_num(tv)):
            s = np.sign(v) if abs(v) > tcrit else 0
            if s != 0 and cur is not None and cur[2] == s:
                cur[1] = i; cur[3] += v
            else:
                if cur is not None: out.append(cur)
                cur = [i, i, s, v] if s != 0 else None
        if cur is not None: out.append(cur)
        return out

    obs = clusters(tvals(data))
    if not obs:
        return []
    null = np.zeros(n_perm)
    for k in range(n_perm):
        cl = clusters(tvals(data * rng.choice([-1, 1], size=(n, 1))))
        null[k] = max([abs(c[3]) for c in cl], default=0)
    return [(c[0], c[1], np.mean(null >= abs(c[3]))) for c in obs]


def mark_clusters(ax, x, data, y, color, alpha=0.05, slot=0):
    """Significant clusters drawn as bars near the top of the panel (axes
    coordinates, so they stay visible whatever the y-range); slot stacks bars."""
    for a, b, p in cluster_test(data):
        if p < alpha:
            ax.plot([x[a], x[b]], [0.97 - 0.035 * slot] * 2, color=color, lw=3, solid_capstyle='butt',
                    transform=ax.get_xaxis_transform(), clip_on=False)


def tc_plot(ax, x, per_sub, color, label, ref=0.0):
    m, se = np.nanmean(per_sub, 0), stats.sem(per_sub, 0, nan_policy='omit')
    ax.plot(x, m, color=color, lw=1.6, label=label)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.2)


def decorate(ax, groups, ylabel, ref=None):
    for t in groups:
        if t in PULSE:
            ax.axvspan(PULSE[t], PULSE[t] + 100, color=P.TMS_LINE_PAL[t], alpha=0.15, lw=0)
    ax.axvline(0, color='k', lw=0.6); ax.axvline(250, color='k', lw=0.6, ls=':')
    if ref is not None: ax.axhline(ref, color='k', lw=0.6, ls='--')
    ax.set_xlim(-400, 4250); ax.set_xlabel('Time from grating onset (ms)'); ax.set_ylabel(ylabel)
    sns.despine(ax=ax)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(study):
    with open(os.path.join(OUT, f'decoding_{study}.pkl'), 'rb') as fh:
        R = pickle.load(fh)
    tr = R['trials'].copy()
    tr['trial'] = tr['trialInBlock'] + 1
    ux = (tr['targX'] - tr['xCenter']).values; uy = -(tr['targY'] - tr['yCenter']).values
    nrm = np.hypot(ux, uy); tr['ux'], tr['uy'] = ux / nrm, uy / nrm
    tr['stim_dir'] = np.degrees(np.arctan2(tr['uy'], tr['ux'])) % 360
    R['trials'] = tr
    sac = pd.read_csv(os.path.join(ROOTS[study], 'analysis', f'gaze_saccades_{study}.csv'))
    sac = sac.merge(tr[['subjID', 'day', 'block', 'trial', 'TMS', 'Hemifield', 'targOriActual', 'stim_dir']].reset_index()
                    .rename(columns={'index': 'row'}), on=['subjID', 'day', 'block', 'trial'], how='inner')
    R['sac'] = sac
    R['groups'] = STUDIES[study]['groups']
    R['label'] = STUDIES[study]['label']
    return R


def per_subject(tr, values, by=None):
    """values: trials x time -> dict {group: subjects x time}"""
    out = {}
    keys = ['subjID'] + ([by] if by else [])
    df = pd.DataFrame(values); df[keys] = tr[keys].values
    for k, g in df.groupby(by) if by else [(None, df)]:
        out[k] = g.groupby('subjID').mean(numeric_only=True).values
    return out


def rate_tc(sac, tr, mask, edges, by='TMS'):
    """Event rate (Hz) per subject x time bin for saccades selected by mask."""
    n_trials = tr.groupby(['subjID', by]).size()
    s = sac[mask]
    out = {}
    for g in tr[by].unique():
        rows = []
        for subj in sorted(tr['subjID'].unique()):
            nt = n_trials.get((subj, g), 0)
            if nt == 0: continue
            on = s[(s['subjID'] == subj) & (s[by] == g)]['onset_ms'].values
            rows.append(np.histogram(on, edges)[0] / nt / (np.diff(edges) / 1000))
        out[g] = np.array(rows)
    return out


def subset_mask(sub, name):
    if name == 'All':
        return np.ones(len(sub), bool)
    if name.startswith('Geo '):
        return sub['GeoType'].values == name[4:]
    if ' · ' in name:
        g, h = name.split(' · ')
        return (sub['TMS'].values == g) & (sub['Hemifield'].values == h)
    return sub['TMS'].values == name


def wrap180(a):
    return (a + 180) % 360 - 180


def wrap90(a):
    return (a + 90) % 180 - 90


# ---------------------------------------------------------------------------
# Per-study measures
# ---------------------------------------------------------------------------
def measures(R):
    tr, t, cen = R['trials'], R['t'], R['centres']
    M = {'t': t, 'cen': cen}
    # gaze / pupil time courses (10-ms bins)
    M['missing'] = per_subject(tr, 1 - R['V'], 'TMS')
    M['pupil'] = per_subject(tr, R['P'], 'TMS')
    speed = np.hypot(np.diff(R['X'], axis=1), np.diff(R['Y'], axis=1)) / 0.01
    M['speed'] = per_subject(tr, np.column_stack([speed, speed[:, -1]]), 'TMS')
    tow = R['X'] * tr['ux'].values[:, None] + R['Y'] * tr['uy'].values[:, None]
    M['toward'] = per_subject(tr, tow, 'TMS')
    M['toward_hemi'] = per_subject(tr.assign(TH=tr['TMS'] + ' · ' + tr['Hemifield']), tow, 'TH')
    # saccades
    edges = np.arange(-400, 4251, 50); M['sac_t'] = edges[:-1] + 25
    sac = R['sac']
    method = sac['method'] if 'method' in sac else pd.Series('iEye', index=sac.index)
    micro = (method == ('EK' if (method == 'EK').any() else 'iEye')) & (sac['amp'] < 1.0)   # Engbert-Kliegl microsaccades
    M['sac_rate'] = rate_tc(sac, tr, (method == 'iEye').values, edges)
    M['micro_rate'] = rate_tc(sac, tr, micro, edges)
    toward = np.cos(np.radians(wrap180(sac['dir'] - sac['stim_dir']))) > 0
    M['micro_toward'] = rate_tc(sac, tr, micro & toward, edges)
    M['micro_away'] = rate_tc(sac, tr, micro & ~toward, edges)
    aligned = np.abs(wrap90(sac['dir'] - sac['targOriActual'])) < 45
    M['micro_aligned'] = rate_tc(sac, tr, micro & aligned, edges)
    M['micro_orth'] = rate_tc(sac, tr, micro & ~aligned, edges)
    # decoders: each trial subset is scored against the null computed on the same subset
    res = R['results']
    loc_auc, fid = {}, {k: {} for k in ['gaze', 'pupil', 'mirror']}
    for s, r in res.items():
        rows = r['rows']; sub = tr.iloc[rows]
        loc = (sub['VF'].values == 'RVF').astype(int)
        targets = {'gaze': sub['targOriActual'].values, 'pupil': sub['targOriActual'].values, 'mirror': r['theta_m']}
        for cond_name in r['null_gaze'].keys():
            m = subset_mask(sub, cond_name)
            a = np.full(len(cen), np.nan)
            for w in range(len(cen)):
                dv = r['loc_dv'][m, w]; ok = ~np.isnan(dv)
                if ok.sum() > 10 and len(np.unique(loc[m][ok])) == 2:
                    a[w] = roc_auc_score(loc[m][ok], dv[ok])
            loc_auc.setdefault(cond_name, []).append(a)
            for k in fid:
                f = np.cos(np.radians(2 * (r[f'ori_{k}'][m] - targets[k][m][:, None])))
                fid[k].setdefault(cond_name, []).append(np.nanmean(f, axis=0) - np.nanmean(r[f'null_{k}'][cond_name], axis=0))
    M['loc_auc'] = {k: np.array(v) for k, v in loc_auc.items()}
    M['fid'] = {k: {c: np.array(v) for c, v in d.items()} for k, d in fid.items()}
    return M


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_quality(Ms, Rs):
    rows = [('missing', 't', 'Missing data (blinks)\nproportion', None),
            ('sac_rate', 'sac_t', 'Saccade rate (Hz)', None),
            ('micro_rate', 'sac_t', 'Microsaccade rate (Hz)\n(< 1°)', None),
            ('pupil', 't', 'Pupil (z, baseline-corrected)', 0),
            ('speed', 't', 'Gaze speed (°/s)', None)]
    fig, axs = plt.subplots(len(rows), 2, figsize=(14, 3.2 * len(rows)), sharex=True)
    for c, (M, R) in enumerate(zip(Ms, Rs)):
        for r, (key, tkey, ylab, ref) in enumerate(rows):
            ax = axs[r, c]
            for g in R['groups']:
                tc_plot(ax, M[tkey], M[key][g], P.TMS_LINE_PAL[g], P.TMS_SHORT[g])
            for g in R['groups'][1:]:
                d = M[key]['No TMS'] - M[key][g]
                mark_clusters(ax, M[tkey], d, None, P.TMS_LINE_PAL[g], slot=R['groups'].index(g))
            decorate(ax, R['groups'], ylab, ref)
            if r == 0:
                ax.set_title(f"{R['label']} (N={len(M[key]['No TMS'])})", fontsize=11, fontweight='bold')
                ax.legend(fontsize=7, frameon=False, loc='upper right')
    fig.suptitle('Eye-data quality and TMS-locked responses (shaded = pulse train; bars on top = None vs TMS, cluster p < .05)', y=0.995)
    plt.tight_layout()
    P.save(fig, OUT, 'GZ1_DataQuality_TMS')


def fig_location(Ms, Rs):
    fig, axs = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for c, (M, R) in enumerate(zip(Ms, Rs)):
        ax = axs[0, c]
        for g in R['groups']:
            tc_plot(ax, M['t'], M['toward'][g], P.TMS_LINE_PAL[g], P.TMS_SHORT[g])
            mark_clusters(ax, M['t'], M['toward'][g], None, P.TMS_LINE_PAL[g], slot=R['groups'].index(g))
        decorate(ax, R['groups'], 'Gaze toward stimulus (°)', 0)
        ax.set_title(f"{R['label']}: gaze bias toward the remembered location", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc='upper left')
        ax = axs[1, c]
        for g in R['groups']:
            tc_plot(ax, M['cen'], M['loc_auc'][g], P.TMS_LINE_PAL[g], P.TMS_SHORT[g])
            mark_clusters(ax, M['cen'], M['loc_auc'][g] - 0.5, None, P.TMS_LINE_PAL[g], slot=R['groups'].index(g))
        decorate(ax, R['groups'], 'Location decoding (AUC)', 0.5)
        ax.set_title('Stimulus side decoded from gaze X, Y (shrinkage LDA, CV over blocks)', fontsize=10)
        ax = axs[2, c]
        for g in R['groups']:
            d = M['micro_toward'][g] - M['micro_away'][g]
            tc_plot(ax, M['sac_t'], d, P.TMS_LINE_PAL[g], P.TMS_SHORT[g])
            mark_clusters(ax, M['sac_t'], d, None, P.TMS_LINE_PAL[g], slot=R['groups'].index(g))
        decorate(ax, R['groups'], 'Microsaccades toward − away (Hz)', 0)
    fig.suptitle('Location: gaze tracks the remembered grating\'s location (bars = cluster p < .05 vs 0 / chance)', y=0.995)
    plt.tight_layout()
    P.save(fig, OUT, 'GZ2_Location')


def fig_orientation(Ms, Rs):
    sets = [('gaze', 'gaze position + axial displacement'), ('mirror', 'gaze, mirrored to a common radial frame'),
            ('pupil', 'pupil size (control)')]
    fig, axs = plt.subplots(len(sets) + 1, 2, figsize=(14, 3.6 * (len(sets) + 1)), sharex=True)
    for c, (M, R) in enumerate(zip(Ms, Rs)):
        for r, (k, desc) in enumerate(sets):
            ax = axs[r, c]
            tc_plot(ax, M['cen'], M['fid'][k]['All'], 'k', 'All trials')
            mark_clusters(ax, M['cen'], M['fid'][k]['All'], None, 'k', slot=0)
            for g in R['groups']:
                tc_plot(ax, M['cen'], M['fid'][k][g], P.TMS_LINE_PAL[g], P.TMS_SHORT[g])
            decorate(ax, R['groups'], 'Orientation decoding\n− shuffle null', 0)
            ax.set_ylim(-0.06, 0.09)
            ax.set_title(f"{R['label']}: {desc}", fontsize=10)
            if r == 0: ax.legend(fontsize=7, frameon=False, loc='upper left', ncol=2)
        ax = axs[-1, c]
        for g in R['groups']:
            d = M['micro_aligned'][g] - M['micro_orth'][g]
            tc_plot(ax, M['sac_t'], d, P.TMS_LINE_PAL[g], P.TMS_SHORT[g])
            mark_clusters(ax, M['sac_t'], d, None, P.TMS_LINE_PAL[g], slot=R['groups'].index(g))
        decorate(ax, R['groups'], 'Microsaccades aligned −\northogonal to grating (Hz)', 0)
    fig.suptitle('Orientation: decoding fidelity cos2(decoded − true) minus block-preserving shuffle null (black bar = cluster p < .05, all trials)', y=0.995)
    plt.tight_layout()
    P.save(fig, OUT, 'GZ3_Orientation')


def window_mean(x_axis, per_sub, lo, hi, pulse_ms=None):
    w = (x_axis >= lo) & (x_axis < hi)
    if pulse_ms is not None:
        w &= ~((x_axis >= pulse_ms + PULSE_MASK[0]) & (x_axis < pulse_ms + PULSE_MASK[1]))
    return np.nanmean(per_sub[:, w], axis=1)


def fig_tms(Ms, Rs):
    metrics = [('toward', 't', 'Gaze toward\nstimulus (°)'), ('loc_auc', 'cen', 'Location AUC − 0.5'),
               (('fid', 'gaze'), 'cen', 'Orientation (gaze)\n− null'), (('fid', 'mirror'), 'cen', 'Orientation (mirrored)\n− null'),
               (('fid', 'pupil'), 'cen', 'Orientation (pupil)\n− null')]
    rows = []
    fig, axs = plt.subplots(len(metrics), 2, figsize=(14, 3.3 * len(metrics)))
    for c, (M, R) in enumerate(zip(Ms, Rs)):
        for r, (key, tkey, ylab) in enumerate(metrics):
            ax = axs[r, c]
            src = M[key[0]][key[1]] if isinstance(key, tuple) else M[key]
            vals = []
            hemis = ['both'] if key == 'loc_auc' else ['inPF', 'outPF']   # location is constant within a hemifield
            for wi, (wname, (lo, hi)) in enumerate(WINDOWS.items()):
                for g in R['groups']:
                    for h in hemis:
                        name = g if h == 'both' else f'{g} · {h}'
                        if key == 'toward':
                            data = M['toward_hemi'][name]
                        else:
                            data = src[name]
                        # exclude every TMS condition's pulse window from all conditions alike
                        x = M[tkey]
                        w = (x >= lo) & (x < hi)
                        for pm in PULSE.values():
                            w &= ~((x >= pm + PULSE_MASK[0]) & (x < pm + PULSE_MASK[1]))
                        v = np.nanmean(data[:, w], axis=1) - (0.5 if key == 'loc_auc' else 0)
                        vals.append({'Window': wname, 'TMS': g, 'Hemifield': h, 'value': v})
                        rows += [{'Study': R['label'], 'Metric': ylab.replace('\n', ' '), 'Window': wname.replace('\n', ' '),
                                  'TMS': g, 'Hemifield': h, 'subject': i, 'value': x_} for i, x_ in enumerate(v)]
            vdf = pd.DataFrame([{**{k: d[k] for k in ['Window', 'TMS', 'Hemifield']}, 'value': x_} for d in vals for x_ in d['value']])
            vdf['cond'] = vdf['TMS'].map(P.TMS_SHORT) + ' · ' + vdf['Hemifield']
            order = [f'{P.TMS_SHORT[g]} · {h}' for g in R['groups'] for h in hemis]
            pal = {f'{P.TMS_SHORT[g]} · {h}': P.TMS_LINE_PAL[g] for g in R['groups'] for h in hemis}
            sns.barplot(data=vdf, x='Window', y='value', hue='cond', hue_order=order, palette=pal, errorbar='se', ax=ax)
            for bar, lab in zip(ax.patches, [o for o in order for _ in WINDOWS]):
                if 'outPF' in lab: bar.set_alpha(0.45)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_ylabel(ylab); ax.set_xlabel('')
            ax.legend(fontsize=6, frameon=False, ncol=2) if r == 0 else ax.get_legend().remove()
            if r == 0: ax.set_title(f"{R['label']} (solid = inPF, faded = outPF; pulse windows excluded)", fontsize=10)
            sns.despine(ax=ax)
    plt.tight_layout()
    P.save(fig, OUT, 'GZ4_TMS')
    return pd.DataFrame(rows)


def tms_stats(long):
    out = []
    for (st, met, win), g in long.groupby(['Study', 'Metric', 'Window']):
        w = g.pivot_table(index='subject', columns=['TMS', 'Hemifield'], values='value')
        groups = [t for t in ['No TMS', 'Early TMS', 'Middle TMS'] if t in w.columns.get_level_values(0)]
        if (groups[0], 'both') in w.columns:
            for t in groups:
                v = w[(t, 'both')]
                out.append({'Study': st, 'Metric': met, 'Window': win, 'Test': f'{t} vs 0', 'Mean': v.mean(),
                            'SEM': v.sem(), 'p': stats.ttest_1samp(v, 0, nan_policy='omit').pvalue})
            for t in groups[1:]:
                d = w[(t, 'both')] - w[(groups[0], 'both')]
                out.append({'Study': st, 'Metric': met, 'Window': win, 'Test': f'{t} − No TMS', 'Mean': d.mean(),
                            'SEM': d.sem(), 'p': stats.ttest_1samp(d, 0, nan_policy='omit').pvalue})
            continue
        for t in groups:
            both = (w[(t, 'inPF')] + w[(t, 'outPF')]) / 2
            out.append({'Study': st, 'Metric': met, 'Window': win, 'Test': f'{t} vs 0', 'Mean': both.mean(),
                        'SEM': both.sem(), 'p': stats.ttest_1samp(both, 0, nan_policy='omit').pvalue})
        for t in groups[1:]:
            d = (w[(t, 'inPF')] + w[(t, 'outPF')]) / 2 - (w[('No TMS', 'inPF')] + w[('No TMS', 'outPF')]) / 2
            out.append({'Study': st, 'Metric': met, 'Window': win, 'Test': f'{t} − No TMS', 'Mean': d.mean(),
                        'SEM': d.sem(), 'p': stats.ttest_1samp(d, 0, nan_policy='omit').pvalue})
            ds = (w[(t, 'inPF')] - w[(t, 'outPF')]) - (w[('No TMS', 'inPF')] - w[('No TMS', 'outPF')])
            out.append({'Study': st, 'Metric': met, 'Window': win, 'Test': f'{t}: (in−out) − None (in−out)',
                        'Mean': ds.mean(), 'SEM': ds.sem(), 'p': stats.ttest_1samp(ds, 0, nan_policy='omit').pvalue})
    return pd.DataFrame(out)


def tms_cluster_table(Ms, Rs):
    """Cluster tests on TMS minus No TMS time courses for every gaze measure."""
    rows = []
    for M, R in zip(Ms, Rs):
        for metric, key, tkey in [('Location AUC', 'loc_auc', 'cen'), ('Gaze toward stimulus', 'toward', 't'),
                                  ('Orientation (gaze)', ('fid', 'gaze'), 'cen'), ('Orientation (mirrored)', ('fid', 'mirror'), 'cen'),
                                  ('Orientation (pupil)', ('fid', 'pupil'), 'cen'), ('Pupil', 'pupil', 't'),
                                  ('Missing data', 'missing', 't'), ('Microsaccade rate', 'micro_rate', 'sac_t')]:
            src = M[key[0]][key[1]] if isinstance(key, tuple) else M[key]
            for g in R['groups'][1:]:
                found = [(M[tkey][a], M[tkey][b], p) for a, b, p in cluster_test(src[g] - src['No TMS']) if p < 0.05]
                rows.append({'Study': R['label'], 'Measure': metric, 'Contrast': f'{g} − No TMS',
                             'Significant clusters (ms, p)': '; '.join(f'{a:.0f}–{b:.0f} ms, p={p:.4f}' for a, b, p in found) or 'none'})
    return pd.DataFrame(rows)


def location_separation(R, window=(2900, 4150)):
    """Late-delay horizontal gaze separation between left- and right-stimulus
    trials, within-location scatter and their ratio (d'), per subject x TMS."""
    tr, t = R['trials'], R['t']
    w = (t >= window[0]) & (t < window[1])
    x = np.nanmean(R['X'][:, w], axis=1)
    rows = []
    for (s, tms), idx in tr.groupby(['subjID', 'TMS']).groups.items():
        sub = tr.loc[idx]; xs = x[tr.index.get_indexer(idx)]
        l, r = xs[sub['VF'].values == 'LVF'], xs[sub['VF'].values == 'RVF']
        sep = np.nanmean(r) - np.nanmean(l); sd = np.sqrt((np.nanvar(l, ddof=1) + np.nanvar(r, ddof=1)) / 2)
        rows.append({'Study': R['label'], 'subjID': s, 'TMS': tms, 'separation_deg': sep, 'scatter_sd_deg': sd, 'dprime': sep / sd})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    Rs = [load('orig'), load('rep')]
    Ms = [measures(R) for R in Rs]
    with open(os.path.join(OUT, 'gaze_measures.pkl'), 'wb') as fh:
        pickle.dump(Ms, fh)
    fig_quality(Ms, Rs)
    fig_location(Ms, Rs)
    fig_orientation(Ms, Rs)
    long = fig_tms(Ms, Rs)
    st = tms_stats(long)
    st.to_csv(os.path.join(OUT, 'Stats_Gaze_TMS.csv'), index=False)
    tms_cluster_table(Ms, Rs).to_csv(os.path.join(OUT, 'Stats_Gaze_TMS_clusters.csv'), index=False)
    sep = pd.concat([location_separation(R) for R in Rs])
    sep.to_csv(os.path.join(OUT, 'SubjectValues_Gaze_LocationSeparation.csv'), index=False)
    print(tms_cluster_table(Ms, Rs).to_string(index=False))
    pd.set_option('display.width', 220)
    print(st.round(4).to_string(index=False))
