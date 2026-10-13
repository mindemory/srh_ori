"""
R03_diagnostics.py

Why does (or doesn't) the bias replication reproduce the manuscript's
radial-sensitivity result? Checks, on the same trials R02 uses:

  1. Study comparison: replication contrast indices next to the values the
     manuscript reports (Study 1), with a z-test of the difference and power.
  2. Day / practice: TMS condition is almost entirely confounded with day in
     this cohort, so radial sensitivity and accuracy are compared by day.
  3. Geometry: each trial's angular distance from the true radial axis at the
     stimulus location, compared with what Study 1's orientation set
     (30/60/120/150 +/- 10) would give at the same locations; error as a
     function of that distance.
  4. Orientation x visual field breakdown of absolute error.
  5. Pull toward the probe's starting orientation (anchoring slope).
  6. Split-half reliability of the per-subject radial index.

Usage: python R03_diagnostics.py [data_root]
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from biasrep_common import (DATA_ROOT, TMS_GROUPS, load_df, run_permutation_test,
                            subject_values, normalized, contrast_stats)

warnings.filterwarnings('ignore')
sns.set_theme(style='white', context='notebook')
matplotlib.rcParams['pdf.fonttype'] = 42
np.random.seed(0)
out_dir = os.path.join(DATA_ROOT, 'Figures', 'paperFigures_biasrep', 'diagnostics')
os.makedirs(out_dir, exist_ok=True)
COL = {'No TMS': 'k', 'Middle TMS': '#29A3E0', 'Early TMS': '#C62828'}

df = load_df()
N = df['subjID'].nunique()
lines = []


def say(s=''):
    print(s); lines.append(s)


def wrap90(x):
    return (x + 90) % 180 - 90


# ---------------------------------------------------------------------------
# 1. Replication vs Study 1 (values as reported in the manuscript, N = 23)
# ---------------------------------------------------------------------------
STUDY1 = pd.DataFrame([
    # subset, metric, TMS, CI_mean, CI_sem, p, d
    ('all', 'Abs Error', 'No TMS', -0.045, 0.016, 0.011, -0.605),
    ('all', 'Abs Error', 'Early TMS', -0.027, 0.013, 0.059, -0.430),
    ('all', 'Abs Error', 'Middle TMS', -0.015, 0.010, 0.134, -0.328),
    ('all', 'Precision (K)', 'No TMS', 0.077, 0.028, 0.015, 0.586),
    ('all', 'Precision (K)', 'Early TMS', 0.045, 0.022, 0.055, 0.433),
    ('all', 'Precision (K)', 'Middle TMS', 0.037, 0.021, 0.086, 0.384),
    ('<45', 'Abs Error', 'No TMS', -0.074, 0.021, 0.002, -0.754),
    ('<45', 'Abs Error', 'Early TMS', -0.021, 0.012, 0.096, -0.369),
    ('<45', 'Abs Error', 'Middle TMS', -0.023, 0.014, 0.115, -0.355),  # text says -0.0023; d and SEM imply -0.023
    ('<45', 'Precision (K)', 'No TMS', 0.139, 0.037, 0.001, 0.799),
    ('<45', 'Precision (K)', 'Early TMS', 0.038, 0.024, 0.132, 0.340),
    ('<45', 'Precision (K)', 'Middle TMS', 0.048, 0.029, 0.112, 0.356),
], columns=['Subset', 'Metric', 'TMS', 'CI_mean', 'CI_sem', 'p_perm', 'Cohen_d'])

vals_rad = subject_values(df, 'DiagType', ['Radial', 'Tangential'])
rep_rows = []
for met, lab in [('AbsError', 'Abs Error'), ('Precision', 'Precision (K)')]:
    rep_rows += contrast_stats(normalized(vals_rad, 'DiagType', 'Radial', 'Tangential', met), 'Radial', 'Tangential', lab)
rep = pd.DataFrame(rep_rows)
rep_cond = rep[rep['TMS'].isin(TMS_GROUPS)]


def power_one_sample(d, n, alpha=0.05):
    tc = stats.t.ppf(1 - alpha / 2, n - 1)
    nc = abs(d) * np.sqrt(n)
    return 1 - stats.nct.cdf(tc, n - 1, nc) + stats.nct.cdf(-tc, n - 1, nc)


def n_for_power(d, target=0.8):
    for n in range(4, 1000):
        if power_one_sample(d, n) >= target: return n


say(f'=== 1. Radial index, replication (N={N}) vs Study 1 (N=23, as reported) ===')
for met in ['Abs Error', 'Precision (K)']:
    for t in TMS_GROUPS:
        r = rep_cond[(rep_cond['Metric'] == met) & (rep_cond['TMS'] == t)].iloc[0]
        for subset in ['all', '<45']:
            s = STUDY1[(STUDY1['Metric'] == met) & (STUDY1['TMS'] == t) & (STUDY1['Subset'] == subset)].iloc[0]
            z = (r['CI_mean'] - s['CI_mean']) / np.hypot(r['CI_sem'], s['CI_sem'])
            say(f'{met:13s} {t:10s} rep {r["CI_mean"]:+.3f}±{r["CI_sem"]:.3f} (p={r["p_perm"]:.3f}, d={r["Cohen_d"]:+.2f}) | '
                f'Study1[{subset:>3s}] {s["CI_mean"]:+.3f}±{s["CI_sem"]:.3f} (d={s["Cohen_d"]:+.2f}) | '
                f'difference z={z:+.2f}, p={2 * stats.norm.sf(abs(z)):.3f}')
for met in ['Abs Error', 'Precision (K)']:
    r = rep[(rep['Metric'] == met) & (rep['TMS'].str.contains('minus'))].iloc[0]
    say(f'{met:13s} No TMS - Middle TMS (replication): {r["CI_mean"]:+.3f}±{r["CI_sem"]:.3f}, p={r["p_perm"]:.3f}, '
        f'95% CI [{r["CI_mean"] - stats.t.ppf(.975, N - 1) * r["CI_sem"]:+.3f}, {r["CI_mean"] + stats.t.ppf(.975, N - 1) * r["CI_sem"]:+.3f}]')
say('Power of this replication (N=%d, two-sided .05) to detect the Study 1 baseline effect:' % N)
for subset, d in [('all', 0.605), ('<45', 0.754)]:
    say(f'  d={d:.3f} ({subset}): power={power_one_sample(d, N):.2f}; N for 80% power = {n_for_power(d)}')
d_rep = abs(rep_cond[(rep_cond['Metric'] == 'Abs Error') & (rep_cond['TMS'] == 'No TMS')]['Cohen_d'].iloc[0])
say(f'  replication baseline d={d_rep:.2f}: N for 80% power = {n_for_power(d_rep)}')

# ---------------------------------------------------------------------------
# 2. Day / practice confound
# ---------------------------------------------------------------------------
say('\n=== 2. Day confound ===')
order = df.groupby('subjID').apply(lambda d: 'mixed' if d.groupby('day')['TMS'].nunique().max() > 1
                                   else ('NoTMS first' if (d[d['day'] == 1]['TMS'] == 'No TMS').all() else 'TMS first'))
say('Order groups: ' + ', '.join(f'{k}: {list(v.index)}' for k, v in order.groupby(order)))
mae_day = df.groupby(['subjID', 'day'])['AbsError'].mean().unstack()
dd = (mae_day[2] - mae_day[1]).values
say(f'Abs error day 1 = {mae_day[1].mean():.2f}°, day 2 = {mae_day[2].mean():.2f}°; '
    f'day2 - day1 = {dd.mean():+.2f} ± {stats.sem(dd):.2f}, p_perm = {run_permutation_test(dd):.3f}')
df['Day'] = 'Day ' + df['day'].astype(str)
vals_day = subject_values(df, 'DiagType', ['Radial', 'Tangential'], group_col='Day')
piv_day = normalized(vals_day, 'DiagType', 'Radial', 'Tangential', 'AbsError', index=('subjID', 'Day'))
for r in contrast_stats(piv_day, 'Radial', 'Tangential', 'Abs Error', groups=['Day 1', 'Day 2'], group_col='Day'):
    say(f'  Radial index by day, {r["TMS"]:18s}: {r["CI_mean"]:+.3f} ± {r["CI_sem"]:.3f}, p = {r["p_perm"]:.3f}, d = {r["Cohen_d"]:+.2f}')
ci_tms = normalized(vals_rad, 'DiagType', 'Radial', 'Tangential', 'AbsError').pivot_table(index='subjID', columns='TMS', values='CI')
ci_tms['order'] = order
say('  Per-subject radial index (abs error) by TMS, with order:')
say(ci_tms.round(3).to_string())
for g, sub in ci_tms.groupby('order'):
    say(f'    {g:12s} No TMS {sub["No TMS"].mean():+.3f}, Middle {sub["Middle TMS"].mean():+.3f} (n={len(sub)})')

# ---------------------------------------------------------------------------
# 3. Geometry: distance from the true radial axis
# ---------------------------------------------------------------------------
say('\n=== 3. Geometry ===')
# polang is in screen coordinates (clockwise from +x, y down); orientations are
# counter-clockwise from horizontal, so the radial axis is (180 - polang) mod 180.
df['RadialAxis'] = (180 - df['polang']) % 180
df['DistRadial'] = np.abs(wrap90(df['targOriActual'] - df['RadialAxis']))
c = df['targOriActual'] % 90
df['DistCardinal'] = np.minimum(c, 90 - c)
elev = df[df['Hemifield'] == 'inPF'].groupby('subjID')['polang'].first()
elev = np.abs(wrap90(elev))  # angle below the horizontal meridian
say(f'Stimulus angle below horizontal: mean {elev.mean():.1f}°, range {elev.min():.1f}–{elev.max():.1f}° (45° would put the radial axis on the diagonal)')

rng = np.random.default_rng(0)
sim = []
for (s, h), loc in df.groupby(['subjID', 'Hemifield']):
    ax = loc['RadialAxis'].iloc[0]
    hemi, vf_right = loc['HemiStim'].iloc[0], loc['targX'].iloc[0] > loc['xCenter'].iloc[0]
    for ori in [30, 60, 120, 150]:
        th = ori + rng.uniform(-10, 10, 200)
        radial = (ori > 90) if vf_right else (ori < 90)
        sim.append(pd.DataFrame({'DistRadial': np.abs(wrap90(th - ax)), 'DiagType': 'Radial' if radial else 'Tangential'}))
sim = pd.concat(sim)
for name, d in [('Replication (45/135 ±24)', df), ('Study-1 design at these locations (30/60/120/150 ±10)', sim)]:
    r = d[d['DiagType'] == 'Radial']['DistRadial']; t = d[d['DiagType'] == 'Tangential']['DistRadial']
    say(f'  {name}: radial trials mean {r.mean():.1f}° from radial axis ({(r < 15).mean() * 100:.0f}% within 15°); '
        f'tangential trials mean {t.mean():.1f}° ({(t > 75).mean() * 100:.0f}% beyond 75°)')

slopes = []
for (s, t), sub in df.groupby(['subjID', 'TMS']):
    X = np.column_stack([np.ones(len(sub)), sub['DistRadial'], sub['DistCardinal']])
    b = np.linalg.lstsq(X, sub['AbsError'].values, rcond=None)[0]
    slopes.append({'subjID': s, 'TMS': t, 'b_radial': b[1] * 10, 'b_cardinal': b[2] * 10})
slopes = pd.DataFrame(slopes)
say('  Abs error per 10° further from the radial axis (controlling distance from cardinal):')
for t in TMS_GROUPS:
    v = slopes[slopes['TMS'] == t]['b_radial'].values
    w = slopes[slopes['TMS'] == t]['b_cardinal'].values
    say(f'    {t:10s} radial slope {v.mean():+.3f}°/10° ± {stats.sem(v):.3f}, p = {run_permutation_test(v):.3f} | '
        f'cardinal slope {w.mean():+.3f} ± {stats.sem(w):.3f}, p = {run_permutation_test(w):.3f}')
sw = slopes.pivot(index='subjID', columns='TMS', values='b_radial')
v = (sw['No TMS'] - sw['Middle TMS']).values
say(f'    No TMS - Middle TMS radial slope: {v.mean():+.3f} ± {stats.sem(v):.3f}, p = {run_permutation_test(v):.3f}')

# ---------------------------------------------------------------------------
# 4. Orientation x visual field
# ---------------------------------------------------------------------------
say('\n=== 4. Orientation x visual field (abs error, mean ± SEM over subjects) ===')
df['VF'] = np.where(df['targX'] > df['xCenter'], 'RVF', 'LVF')
grid = df.groupby(['subjID', 'TMS', 'VF', 'OriCat'])['AbsError'].mean().reset_index()
gsum = grid.groupby(['TMS', 'VF', 'OriCat'])['AbsError'].agg(['mean', 'sem']).round(2)
say(gsum.to_string())

# ---------------------------------------------------------------------------
# 5. Anchoring to the probe's starting orientation
# ---------------------------------------------------------------------------
say('\n=== 5. Pull toward the starting orientation (slope of signed error on signed offset) ===')
anc = []
for (s, t, d), sub in df.groupby(['subjID', 'TMS', 'DiagType']):
    anc.append({'subjID': s, 'TMS': t, 'DiagType': d, 'slope': np.polyfit(sub['initOffset'], sub['errOri_scaled'], 1)[0]})
anc = pd.DataFrame(anc)
for (t, d), sub in anc.groupby(['TMS', 'DiagType']):
    say(f'  {t:10s} {d:10s} slope {sub["slope"].mean():+.3f} ± {stats.sem(sub["slope"]):.3f}, p = {run_permutation_test(sub["slope"].values):.4f}')
aw = anc.pivot_table(index='subjID', columns=['TMS', 'DiagType'], values='slope')
for t in TMS_GROUPS:
    v = (aw[(t, 'Tangential')] - aw[(t, 'Radial')]).values
    say(f'  {t:10s} tangential - radial: {v.mean():+.3f} ± {stats.sem(v):.3f}, p = {run_permutation_test(v):.3f}')
v = (aw['Middle TMS'].mean(axis=1) - aw['No TMS'].mean(axis=1)).values
say(f'  Middle - No TMS (pooled types): {v.mean():+.3f} ± {stats.sem(v):.3f}, p = {run_permutation_test(v):.3f}')

# ---------------------------------------------------------------------------
# 6. Split-half reliability of the radial index (abs error)
# ---------------------------------------------------------------------------
say('\n=== 6. Reliability of the per-subject radial index (abs error) ===')
df['Half'] = np.where(df['block'] % 2 == 1, 'odd', 'even')
for t in TMS_GROUPS:
    sub = df[df['TMS'] == t]
    m = sub.groupby(['subjID', 'Half', 'DiagType'])['AbsError'].mean().unstack()
    ci = ((m['Radial'] - m['Tangential']) / (m['Radial'] + m['Tangential'])).unstack()
    r = stats.pearsonr(ci['odd'], ci['even'])[0]
    say(f'  {t:10s} odd vs even blocks r = {r:+.2f} (Spearman-Brown full-length = {2 * r / (1 + r):+.2f})')
r = stats.pearsonr(ci_tms['No TMS'], ci_tms['Middle TMS'])
say(f'  No TMS vs Middle TMS index across subjects: r = {r[0]:+.2f}, p = {r[1]:.3f}')

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(18, 11))

# A: forest plot of radial index (abs error)
ax = axs[0, 0]
rows = []
for subset, lab in [('all', 'Study 1 (all trials)'), ('<45', 'Study 1 (|Δprobe|<45°)')]:
    for t in ['No TMS', 'Early TMS', 'Middle TMS']:
        s = STUDY1[(STUDY1['Metric'] == 'Abs Error') & (STUDY1['Subset'] == subset) & (STUDY1['TMS'] == t)].iloc[0]
        rows.append((lab, t, s['CI_mean'], s['CI_sem'], 23))
for t in TMS_GROUPS:
    r = rep_cond[(rep_cond['Metric'] == 'Abs Error') & (rep_cond['TMS'] == t)].iloc[0]
    rows.append((f'Replication (N={N})', t, r['CI_mean'], r['CI_sem'], N))
for i, (study, t, m, se, n) in enumerate(rows):
    y = len(rows) - i
    ax.errorbar(m, y, xerr=stats.t.ppf(.975, n - 1) * se, fmt='o', color=COL[t], ms=8, capsize=3)
    ax.text(0.075, y, f'{study} · {t}', va='center', fontsize=8)
ax.axvline(0, color='k', ls='--', lw=1)
ax.set_xlim(-0.15, 0.25); ax.set_yticks([])
ax.set_xlabel('Radial index, abs error: (R−T)/(R+T)\n(negative = radial advantage; 95% CI)')
ax.set_title('A. Radial index across studies', loc='left', fontweight='bold')

# B: radial index by day
ax = axs[0, 1]
pd_ = piv_day.pivot_table(index='subjID', columns='Day', values='CI')
tms_day1 = df[df['day'] == 1].groupby('subjID')['TMS'].agg(lambda s: s.mode()[0] if s.nunique() == 1 else 'mixed')
for s, row in pd_.iterrows():
    ax.plot([0, 1], [row['Day 1'], row['Day 2']], color=COL.get(tms_day1[s], 'grey'), alpha=0.6, marker='o')
ax.errorbar([0, 1], pd_.mean(), yerr=pd_.sem(), color='m', lw=3, capsize=4, marker='s', ms=9, label='mean ± SEM')
ax.axhline(0, color='k', ls='--', lw=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Day 1', 'Day 2']); ax.set_xlim(-0.3, 1.3)
ax.set_ylabel('Radial index (abs error)')
ax.set_title('B. Radial index by day\n(line colour = condition on day 1: black No TMS, blue Middle, grey mixed)', loc='left', fontweight='bold', fontsize=10)
ax.legend(frameon=False, fontsize=8)

# C: distance-from-radial distributions
ax = axs[0, 2]
bins = np.arange(0, 91, 5)
for d, ls in [('Radial', '-'), ('Tangential', '--')]:
    ax.hist(df[df['DiagType'] == d]['DistRadial'], bins=bins, density=True, histtype='step', lw=2, ls=ls, color='tab:purple', label=f'Replication · {d}')
    ax.hist(sim[sim['DiagType'] == d]['DistRadial'], bins=bins, density=True, histtype='step', lw=2, ls=ls, color='tab:orange', label=f'Study-1 design · {d}')
ax.set_xlabel('Distance of target from the true radial axis (°)'); ax.set_ylabel('Density')
ax.set_title('C. Where the orientation sets sit\nrelative to the radial axis (these 12 stimulus locations)', loc='left', fontweight='bold', fontsize=10)
ax.legend(frameon=False, fontsize=8)

# D: error vs distance from the radial axis
ax = axs[1, 0]
df['DistBin'] = pd.cut(df['DistRadial'], bins=np.arange(0, 91, 15))
b = df.groupby(['subjID', 'TMS', 'DistBin'], observed=True)['AbsError'].mean().reset_index()
b['mid'] = b['DistBin'].apply(lambda x: x.mid).astype(float)
for t in TMS_GROUPS:
    g = b[b['TMS'] == t].groupby('mid')['AbsError'].agg(['mean', 'sem'])
    ax.errorbar(g.index + (1 if t == 'Middle TMS' else -1), g['mean'], yerr=g['sem'], color=COL[t], marker='o', capsize=3, label=t)
ax.set_xlabel('Distance from radial axis (°, 15° bins)'); ax.set_ylabel('Abs error (°)')
ax.set_title('D. Error vs distance from the radial axis', loc='left', fontweight='bold')
ax.legend(frameon=False, fontsize=8)

# E: orientation x visual field
ax = axs[1, 1]
for i, t in enumerate(TMS_GROUPS):
    for ori, mk in [(45, 'o'), (135, 's')]:
        g = grid[(grid['TMS'] == t) & (grid['OriCat'] == ori)].groupby('VF')['AbsError'].agg(['mean', 'sem']).reindex(['LVF', 'RVF'])
        ax.errorbar(np.array([0, 1]) + i * 2.5 + (0.08 if ori == 135 else -0.08), g['mean'], yerr=g['sem'], color=COL[t],
                    marker=mk, ls='-' if ori == 45 else '--', capsize=3, label=f'{t} · {ori}°')
ax.set_xticks([0, 1, 2.5, 3.5]); ax.set_xticklabels(['LVF', 'RVF', 'LVF', 'RVF'])
ax.set_ylabel('Abs error (°)')
ax.set_title('E. Orientation × visual field\n(radial = 45° in LVF, 135° in RVF)', loc='left', fontweight='bold', fontsize=10)
ax.legend(frameon=False, fontsize=7, ncol=2)

# F: anchoring
ax = axs[1, 2]
sns.barplot(data=anc, x='TMS', y='slope', hue='DiagType', order=TMS_GROUPS, ax=ax, errorbar='se', palette=['#4472C4', '#A9C4E8'])
sns.stripplot(data=anc, x='TMS', y='slope', hue='DiagType', order=TMS_GROUPS, ax=ax, dodge=True, color='k', alpha=0.4, legend=False)
ax.axhline(0, color='k', lw=1)
ax.set_ylabel('Signed error per ° of starting offset')
ax.set_title('F. Pull toward the starting orientation', loc='left', fontweight='bold')

for a in axs.flat: sns.despine(ax=a, offset=5)
plt.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(os.path.join(out_dir, f'Diagnostics.{ext}'), dpi=200, bbox_inches='tight')
with open(os.path.join(out_dir, 'Diagnostics.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nSaved to {out_dir}')
