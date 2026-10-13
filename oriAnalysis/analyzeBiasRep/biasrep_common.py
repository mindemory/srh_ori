"""
Shared loading, labelling and statistics for the Study 1 / bias-replication
analyses. Everything here mirrors analyzeBehav/V10_paperFigures.py.
"""
import os
import sys
import types

import numpy as np
import pandas as pd
from scipy import stats

# modeller.py imports pycircstat at module level but fit_model never uses it.
sys.modules.setdefault('pycircstat', types.ModuleType('pycircstat'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analyzeBehav'))
import modeller as md  # noqa: E402

DATA_ROOT = os.environ.get('BIASREP_ROOT', '/Users/mrugank/Documents/TMS ori/bias_rep')
ORIG_ROOT = os.environ.get('ORIG_ROOT', '/Users/mrugank/Documents/TMS ori/orig_data')
COMPARE_DIR = os.path.join(DATA_ROOT, 'Figures', 'comparison_orig_vs_rep')

STUDIES = {
    'orig': dict(pkl=os.path.join(ORIG_ROOT, 'analysis', 'master_df_orig.pkl'),
                 groups=['No TMS', 'Early TMS', 'Middle TMS'], label='Study 1'),
    'rep': dict(pkl=os.path.join(DATA_ROOT, 'analysis', 'master_df_biasrep.pkl'),
                groups=['No TMS', 'Middle TMS'], label='Replication'),
}
TMS_GROUPS = STUDIES['rep']['groups']   # default for replication-only scripts
RADIAL_RVF = [120, 150, 135]   # obtuse orientations are radial in the right visual field
RADIAL_LVF = [30, 60, 45]      # acute orientations are radial in the left visual field


def get_ori_type(row):
    ori, hemi, pf = row['OriCat'], row['HemiStim'], row['Hemifield']
    if hemi == 'Left':  # Stimulating Left Hemi -> Phosphene in Right VF
        if pf == 'inPF':
            return 'Radial' if ori in RADIAL_RVF else 'Tangential'
        return 'Radial' if ori in RADIAL_LVF else 'Tangential'
    else:  # Stimulating Right Hemi -> Phosphene in Left VF
        if pf == 'inPF':
            return 'Radial' if ori in RADIAL_LVF else 'Tangential'
        return 'Radial' if ori in RADIAL_RVF else 'Tangential'


def load_df(study='rep', hemi_source='position', verbose=True, drop_outliers=True):
    """master_df with LargeError trials removed and V10's derived columns.
    hemi_source: 'position' = hemisphere implied by where the in-PF stimulus
    was drawn (always correct); 'sheet' = V1ori_TMS_meta.csv, as the paper used
    (Study 1 only). drop_outliers=False keeps the LargeError trials."""
    df = pd.read_pickle(STUDIES[study]['pkl'])
    df.reset_index(drop=True, inplace=True)
    if study == 'orig' and hemi_source == 'position':
        df['hemisphereStimulated'] = df['hemisphereStimulated_pos']
    # Previous trial in the same block (rows are in presentation order), taken
    # before outlier removal so every trial keeps its true predecessor.
    blk = df.groupby(['subjID', 'day', 'block'], sort=False)
    df['trialInBlock'] = blk.cumcount()
    df['prevTarg'] = blk['targOriActual'].shift(1)
    df['prevResp'] = blk['finalOri'].shift(1)
    df['prevStimPF'] = blk['stimPF'].shift(1)
    df['prevLargeError'] = blk['LargeError'].shift(1)
    n_all = len(df)
    if drop_outliers:
        df = df[df['LargeError'] == 0].copy()
        if verbose:
            print(f"[{study}] Removed {n_all - len(df)} LargeError trials; {len(df)} remain.")
    df.rename(columns={'TMScond_label': 'TMS', 'stimPF_label': 'Hemifield',
                       'targOriCategory': 'OriCat', 'hemisphereStimulated': 'HemiStim'}, inplace=True)
    df['AbsError'] = np.abs(df['errOri_scaled'])
    df['AbsInitOffset'] = np.abs(df['initOffset'])
    df['OriCat'] = df['OriCat'].astype(int)
    df['DiagType'] = df.apply(get_ori_type, axis=1)
    return df


def run_permutation_test(vals, n_perms=10000):
    vals = np.array(vals); vals = vals[~np.isnan(vals)]
    if len(vals) == 0: return 1.0
    obs = np.mean(vals)
    null = [np.mean(vals * np.random.choice([-1, 1], size=len(vals))) for _ in range(n_perms)]
    p = np.mean(np.abs(null) >= np.abs(obs))
    return max(p, 1 / n_perms)


def fit_kappa(errs):
    return md.fit_model(errs, 'mixtureModel', max_trials=100)[1]


def subject_values(data, p_col, levels, extra_cols=(), group_col='TMS'):
    """Per subject x group x p_col (x extra_cols): mean AbsError, RT and mixture kappa
    (V10 plot_contrast_grid aggregation; cells with < 10 trials skipped)."""
    g_cols = ['subjID', group_col, p_col, *extra_cols]
    df_s = data.groupby(g_cols, observed=False)[['AbsError', 'RT']].mean().reset_index()
    mix = []
    for keys, sub in data.groupby(g_cols, observed=True):
        if keys[2] not in levels or len(sub) < 10: continue
        try:
            mix.append(dict(zip(g_cols, keys), Precision=fit_kappa(sub['errOri_scaled'].dropna().values)))
        except ValueError:
            pass
    return pd.merge(df_s, pd.DataFrame(mix), on=g_cols, how='outer')


def normalized(vals, p_col, A, B, met, index=('subjID', 'TMS')):
    piv = vals.pivot_table(index=list(index), columns=p_col, values=met, observed=False).reset_index()
    total = piv[A] + piv[B] + 1e-12
    piv[f'{A}_norm'] = piv[A] / total
    piv[f'{B}_norm'] = piv[B] / total
    piv['CI'] = piv[f'{A}_norm'] - piv[f'{B}_norm']
    return piv


def contrast_stats(piv, A, B, label, extra=None, groups=None, group_col='TMS'):
    """V10 stats per group (A_norm vs 0.5, reported as CI = A_norm - B_norm),
    plus paired tests of CI between the first group and each other group
    (not in the paper)."""
    groups = groups or TMS_GROUPS
    rows = []
    for t in groups:
        sub = piv[piv[group_col] == t].dropna(subset=[f'{A}_norm', f'{B}_norm'])
        v = sub[f'{A}_norm'].values
        if len(v) <= 5: continue
        diffs = v - 0.5
        ci = sub['CI'].values
        rows.append({**(extra or {}), 'Metric': label, 'Contrast': f'{A}-{B}', 'TMS': t, 'N': len(v),
                     f'{A}_norm_mean': np.mean(v), f'{A}_norm_sem': stats.sem(v),
                     'CI_mean': np.mean(ci), 'CI_sem': stats.sem(ci),
                     'p_perm': run_permutation_test(diffs),
                     'Cohen_d': np.mean(diffs) / np.std(diffs),
                     't_stat': stats.ttest_1samp(v, 0.5)[0]})
    w = piv.pivot_table(index='subjID', columns=group_col, values='CI')
    for other in groups[1:]:
        if groups[0] not in w.columns or other not in w.columns: continue
        dd = (w[groups[0]] - w[other]).dropna().values
        if len(dd) <= 5: continue
        rows.append({**(extra or {}), 'Metric': label, 'Contrast': f'{A}-{B}', 'TMS': f'{groups[0]} minus {other}',
                     'N': len(dd), 'CI_mean': np.mean(dd), 'CI_sem': stats.sem(dd),
                     'p_perm': run_permutation_test(dd), 'Cohen_d': np.mean(dd) / np.std(dd),
                     't_stat': stats.ttest_1samp(dd, 0)[0]})
    return rows
