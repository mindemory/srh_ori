"""
Computes (and caches) every quantity the comparison figures need for one
study. Paper analyses follow analyzeBehav/V10_paperFigures.py and V05; the
geometry analyses use biasrep_geometry.py.

compute(study)   study = 'orig' (Study 1) or 'rep' (bias replication)

Radial/Tangential category labels follow the paper: Study 1 takes the
stimulated hemisphere from V1ori_TMS_meta.csv (as V10 did); the replication,
which has no sheet, takes it from where the in-PF stimulus was drawn. The
geometry analyses place every trial by its recorded screen position.
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats

from biasrep_common import (STUDIES, COMPARE_DIR, load_df, run_permutation_test, fit_kappa,
                            subject_values, normalized, contrast_stats)
import biasrep_geometry as bg

CACHE_DIR = os.path.join(COMPARE_DIR, 'cache')


def _per_subject_bins(df, value, bins, by=('TMS',), col='DistRadial'):
    d = df.copy()
    d['bin'] = pd.cut(d[col], bins=bins, include_lowest=True)
    out = d.groupby(['subjID', *by, 'bin'], observed=True)[value].mean().reset_index()
    out['mid'] = out['bin'].apply(lambda x: x.mid).astype(float)
    return out.drop(columns='bin')


def _cw_tendency(ori_array):  # V05 get_cw_tendency_continuous
    if len(ori_array) < 2: return 0.5
    diffs = np.diff(ori_array)
    diffs[diffs > 90] -= 180
    diffs[diffs < -90] += 180
    cw, ccw = np.sum(np.abs(diffs[diffs < 0])), np.sum(diffs[diffs > 0])
    return 0.5 if cw + ccw == 0 else cw / (cw + ccw)


def _label_dependent(df, G, R):
    """Everything that uses the Radial/Tangential labels."""
    R['vals_hemi'] = subject_values(df, 'Hemifield', ['inPF', 'outPF'])
    R['vals_rad'] = subject_values(df, 'DiagType', ['Radial', 'Tangential'])
    R['vals_rad_hemi'] = subject_values(df, 'DiagType', ['Radial', 'Tangential'], extra_cols=('Hemifield',))
    rows, piv = [], {}
    for met, lab in [('AbsError', 'Abs Error'), ('RT', 'Reaction Time'), ('Precision', 'Precision (K)')]:
        piv[('hemi', met)] = normalized(R['vals_hemi'], 'Hemifield', 'inPF', 'outPF', met)
        rows += contrast_stats(piv[('hemi', met)], 'inPF', 'outPF', lab, {'Subset': 'all'}, groups=G)
    for met, lab in [('AbsError', 'Abs Error'), ('Precision', 'Precision (K)')]:
        piv[('rad', met)] = normalized(R['vals_rad'], 'DiagType', 'Radial', 'Tangential', met)
        rows += contrast_stats(piv[('rad', met)], 'Radial', 'Tangential', lab, {'Subset': 'all'}, groups=G)
        for hemi in ['inPF', 'outPF']:
            p = normalized(R['vals_rad_hemi'][R['vals_rad_hemi']['Hemifield'] == hemi], 'DiagType', 'Radial', 'Tangential', met)
            piv[('rad_hemi', met, hemi)] = p
            rows += contrast_stats(p, 'Radial', 'Tangential', lab, {'Subset': f'all, {hemi}'}, groups=G)
    # Offset subsets: < 45 deg as in the paper; < 30 deg = the replication's whole range
    for lim in (45, 30):
        sub = df[df['AbsInitOffset'] < lim]
        same = len(sub) == len(df)
        vh = R['vals_hemi'] if same else subject_values(sub, 'Hemifield', ['inPF', 'outPF'])
        vr = R['vals_rad'] if same else subject_values(sub, 'DiagType', ['Radial', 'Tangential'])
        for met, lab in [('AbsError', 'Abs Error'), ('Precision', 'Precision (K)')]:
            piv[(f'hemi{lim}', met)] = normalized(vh, 'Hemifield', 'inPF', 'outPF', met)
            piv[(f'rad{lim}', met)] = normalized(vr, 'DiagType', 'Radial', 'Tangential', met)
            rows += contrast_stats(piv[(f'hemi{lim}', met)], 'inPF', 'outPF', lab, {'Subset': f'<{lim}'}, groups=G)
            rows += contrast_stats(piv[(f'rad{lim}', met)], 'Radial', 'Tangential', lab, {'Subset': f'<{lim}'}, groups=G)
    R['piv'], R['stats'] = piv, pd.DataFrame(rows)

    # Radial index by day (pooled over TMS conditions)
    d = df.assign(Day='Day ' + df['day'].astype(str))
    vd = subject_values(d, 'DiagType', ['Radial', 'Tangential'], group_col='Day')
    R['day_index'] = normalized(vd, 'DiagType', 'Radial', 'Tangential', 'AbsError', index=('subjID', 'Day'))

    # Pull toward the start, per orientation type (|offset| < 30 in both studies)
    anc = []
    for (s, t, dt), sub in df[df['AbsInitOffset'] < 30].groupby(['subjID', 'TMS', 'DiagType']):
        anc.append({'subjID': s, 'TMS': t, 'DiagType': dt, 'slope': np.polyfit(sub['initOffset'], sub['errOri_scaled'], 1)[0]})
    R['anchor'] = pd.DataFrame(anc)


ERR_GRID = np.linspace(-30, 30, 121)
KDE_BW = 2.5   # deg, fixed bandwidth for per-subject error densities


def _label_independent(df, G, R):
    """Geometry and response dynamics (recorded stimulus positions, no labels)."""
    R['an'] = bg.anisotropy_table(df, ['subjID', 'TMS'])
    R['an_hemi'] = bg.anisotropy_table(df, ['subjID', 'TMS', 'Hemifield'])

    # Error distributions and accuracy per subject x TMS x hemifield x geometric type
    dens, cells = [], []
    for (s, t, h, gt), sub in df.groupby(['subjID', 'TMS', 'Hemifield', 'GeoType']):
        e = sub['errOri_scaled'].values
        if len(e) < 8: continue
        dens.append({'subjID': s, 'TMS': t, 'Hemifield': h, 'GeoType': gt,
                     'density': np.exp(-0.5 * ((ERR_GRID[:, None] - e[None, :]) / KDE_BW) ** 2).mean(axis=1)
                                / (KDE_BW * np.sqrt(2 * np.pi))})
        kappa = np.nan
        if len(e) >= 10:
            try:
                kappa = fit_kappa(e)
            except ValueError:
                pass
        cells.append({'subjID': s, 'TMS': t, 'Hemifield': h, 'GeoType': gt, 'n': len(e),
                      'AbsError': np.abs(e).mean(), 'Precision': kappa})
    R['err_density'], R['geo_cells'] = pd.DataFrame(dens), pd.DataFrame(cells)

    # Where each orientation category falls relative to the radial axis, per subject
    R['geo_overview'] = (df.groupby(['subjID', 'Hemifield', 'OriCat', 'DiagType'])
                         .agg(Elevation=('Elevation', 'first'), VF=('VF', 'first'),
                              phi_min=('PhiAligned', 'min'), phi_max=('PhiAligned', 'max'),
                              phi_mean=('PhiAligned', 'mean'))
                         .reset_index())

    bins15 = np.arange(0, 91, 15)
    R['curve_err'] = _per_subject_bins(df, 'AbsError', bins15)
    R['curve_err_hemi'] = _per_subject_bins(df, 'AbsError', bins15, by=('TMS', 'Hemifield'))
    R['curve_rt'] = _per_subject_bins(df, 'RT', bins15)
    anc = []
    d = df[df['AbsInitOffset'] < 30].copy()
    d['bin'] = pd.cut(d['DistRadial'], bins=[0, 30, 60, 90], include_lowest=True)
    for (s, t, b), sub in d.groupby(['subjID', 'TMS', 'bin'], observed=True):
        if len(sub) < 15: continue
        anc.append({'subjID': s, 'TMS': t, 'mid': b.mid, 'slope': np.polyfit(sub['initOffset'], sub['errOri_scaled'], 1)[0]})
    R['curve_pull'] = pd.DataFrame(anc)

    # Fig 4 (V10 plot_sat_grid / compute_sat_stats and V05 P(CW))
    n_bins = 25
    rt_parts, off_parts, sat = [], [], []
    for (subj, tms), sdf in df.groupby(['subjID', 'TMS']):
        if len(sdf) < n_bins: continue
        sdf = sdf.copy()
        sdf['RT_Decile'] = pd.qcut(sdf['RT'], q=n_bins, labels=False, duplicates='drop')
        a = sdf.groupby('RT_Decile').agg({'RT': 'mean', 'AbsError': 'mean'}).reset_index()
        a.update(a[['RT', 'AbsError']].rolling(window=3, center=True).mean())
        sdf['Offset_Decile'] = pd.qcut(sdf['AbsInitOffset'], q=n_bins, labels=False, duplicates='drop')
        b = sdf.groupby('Offset_Decile').agg({'AbsInitOffset': 'mean', 'RT': 'mean', 'AbsError': 'mean'}).reset_index()
        b.update(b[['AbsInitOffset', 'RT', 'AbsError']].rolling(window=3, center=True).mean())
        a['subjID'], a['TMS'], b['subjID'], b['TMS'] = subj, tms, subj, tms
        rt_parts.append(a); off_parts.append(b)
        sat.append({'subjID': subj, 'TMS': tms,
                    'Offset_RT': stats.spearmanr(sdf['AbsInitOffset'], sdf['RT'])[0],
                    'RT_Error': stats.spearmanr(sdf['RT'], sdf['AbsError'])[0],
                    'Offset_Error': stats.spearmanr(sdf['AbsInitOffset'], sdf['AbsError'])[0]})
    df_rt, df_off = pd.concat(rt_parts), pd.concat(off_parts)
    df_rt['Group_RT'] = df_rt.groupby(['RT_Decile', 'TMS'])['RT'].transform('mean')
    df_off['Group_Offset'] = df_off.groupby(['Offset_Decile', 'TMS'])['AbsInitOffset'].transform('mean')
    R['df_rt'], R['df_off'], R['sat'] = df_rt, df_off, pd.DataFrame(sat)

    pcw = df[(df['RT'] < 3.0) & (df['oriSteps'] > 0)].copy()  # V05 filters
    pcw['CW_Tendency'] = pcw['oriArray'].apply(_cw_tendency)
    pcw['Offset_Bin'] = pd.cut(pcw['initOffset'], bins=np.arange(-90, 92, 2))
    ss = pcw.groupby(['subjID', 'TMS', 'Offset_Bin'], observed=True).agg(
        Mean_Tendency=('CW_Tendency', 'mean'), Bin_Mid=('initOffset', 'mean')).reset_index()
    full_range = pcw['initOffset'].abs().max() > 80
    sm = []
    for _, sdf in ss.groupby(['subjID', 'TMS']):
        sdf = sdf.sort_values('Bin_Mid').reset_index(drop=True)
        if full_range:   # V05: circular 5-bin smoothing over -90..90
            if len(sdf) < 10: continue
            padded = pd.concat([sdf.tail(5), sdf, sdf.head(5)]).reset_index(drop=True)
            padded['Mean_Tendency'] = padded['Mean_Tendency'].rolling(window=5, center=True).mean()
            sdf = padded.iloc[5:-5].copy()
        else:            # offsets span only -30..30: no wrap-around
            sdf['Mean_Tendency'] = sdf['Mean_Tendency'].rolling(window=5, center=True, min_periods=3).mean()
        sm.append(sdf)
    R['pcw'] = pd.concat(sm).groupby(['TMS', 'Offset_Bin'], observed=True).agg(
        Pop_Mean=('Mean_Tendency', 'mean'), Pop_SEM=('Mean_Tendency', lambda x: np.std(x) / np.sqrt(len(x))),
        X=('Bin_Mid', 'mean')).reset_index()

    # Orientation x visual field (raw abs error) and where targets sit relative to the radial axis
    R['ori_vf'] = df.groupby(['subjID', 'TMS', 'VF', 'OriCat'])['AbsError'].mean().reset_index()
    R['elevation'] = df[df['Hemifield'] == 'inPF'].groupby('subjID')['Elevation'].first()
    R['dist_by_type'] = df[['DiagType', 'DistRadial']].copy()


PAPER_HEMI = {'orig': 'sheet', 'rep': 'position'}


def load_study(study, drop_outliers=True):
    """Trial table with the paper's labels and geometry columns."""
    return bg.add_geometry(load_df(study, PAPER_HEMI[study], verbose=False, drop_outliers=drop_outliers))


def compute(study, force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{study}.pkl')
    if os.path.exists(path) and not force:
        with open(path, 'rb') as f:
            return pickle.load(f)
    np.random.seed(0)
    G = STUDIES[study]['groups']
    df = load_study(study)
    R = {'study': study, 'groups': G, 'label': STUDIES[study]['label'], 'N': df['subjID'].nunique()}
    print(f'[{study}] paper analyses...')
    _label_dependent(df, G, R)
    print(f'[{study}] geometry and response dynamics...')
    _label_independent(df, G, R)
    with open(path, 'wb') as f:
        pickle.dump(R, f)
    return R
