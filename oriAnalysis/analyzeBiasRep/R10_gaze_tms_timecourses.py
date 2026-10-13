"""
R10_gaze_tms_timecourses.py   (srh conda env: ~/.conda/envs/srh/bin/python)

TMS-locked eye responses (blinks, saccade and microsaccade rate, pupil, gaze
speed, gaze toward the stimulus, location AUC) from gaze_figures.py's cached
time courses (gaze_measures.pkl). Adds what Stats_Gaze_TMS_clusters.csv lacks:
cluster tests for saccade rate and gaze speed, middle-delay TMS pooled across
both cohorts (N = 35), and post-pulse window means.

Output (<COMPARE_DIR>/gaze): Stats_Gaze_TMS_TimeCourse.csv
"""
import os
import pickle

import numpy as np
import pandas as pd

import gaze_figures as G
from biasrep_common import run_permutation_test

KEYS = [('Missing data', 'missing', 't'), ('Saccade rate', 'sac_rate', 'sac_t'),
        ('Microsaccade rate', 'micro_rate', 'sac_t'), ('Pupil', 'pupil', 't'), ('Gaze speed', 'speed', 't'),
        ('Gaze toward stimulus', 'toward', 't'), ('Location AUC', 'loc_auc', 'cen')]
WINDOWS = [('Pupil', 'pupil', 't', 700, 2000), ('Pupil', 'pupil', 't', 1000, 2200), ('Pupil', 'pupil', 't', 2400, 4200),
           ('Missing data', 'missing', 't', 2400, 3400), ('Microsaccade rate', 'micro_rate', 'sac_t', 2400, 3400),
           ('Saccade rate', 'sac_rate', 'sac_t', 2400, 3400), ('Gaze speed', 'speed', 't', 2400, 3400)]

with open(os.path.join(G.OUT, 'gaze_measures.pkl'), 'rb') as fh:
    Ms = pickle.load(fh)
labels = ['Study 1', 'Replication']
rows = []


def add_clusters(study, measure, contrast, d, x):
    for a, b, p in G.cluster_test(d):
        rows.append({'Test': 'cluster', 'Study': study, 'Measure': measure, 'Contrast': contrast, 'N': len(d),
                     'from_ms': x[a], 'to_ms': x[b], 'Mean': np.nanmean(d[:, a:b + 1]), 'p': p})


for M, lab in zip(Ms, labels):
    for name, key, tk in KEYS:
        for g in [k for k in M[key] if k in ('Early TMS', 'Middle TMS')]:
            add_clusters(lab, name, f'{g} − No TMS', M[key][g] - M[key]['No TMS'], M[tk])
for name, key, tk in KEYS:
    if key == 'loc_auc':
        continue
    d = np.vstack([M[key]['Middle TMS'] - M[key]['No TMS'] for M in Ms])
    add_clusters('Both datasets', name, 'Middle TMS − No TMS', d, Ms[0][tk])

for name, key, tk, lo, hi in WINDOWS:
    for g in ['Early TMS', 'Middle TMS']:
        vals = []
        for M, lab in zip(Ms, labels):
            if g not in M[key]:
                continue
            w = (M[tk] >= lo) & (M[tk] < hi)
            v = np.nanmean((M[key][g] - M[key]['No TMS'])[:, w], 1)
            vals.append((lab, v[~np.isnan(v)]))
        if g == 'Middle TMS':
            vals.append(('Both datasets', np.concatenate([v for _, v in vals])))
        for lab, v in vals:
            rows.append({'Test': 'window', 'Study': lab, 'Measure': name, 'Contrast': f'{g} − No TMS', 'N': len(v),
                         'from_ms': lo, 'to_ms': hi, 'Mean': v.mean(), 'SEM': v.std(ddof=1) / np.sqrt(len(v)),
                         'd': v.mean() / v.std(), 'p': run_permutation_test(v)})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(G.OUT, 'Stats_Gaze_TMS_TimeCourse.csv'), index=False)
pd.set_option('display.width', 220)
print(out[(out['Test'] == 'window') | (out['p'] < 0.1)].round(4).to_string(index=False))
