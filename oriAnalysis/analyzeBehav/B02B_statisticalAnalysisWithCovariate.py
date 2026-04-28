import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy import stats
import generate_behavdf as gm
import os
import sys
import warnings
import datetime
import modeller as md

warnings.filterwarnings('ignore')
pd.set_option("display.max.columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.precision", 3)

# =============================================================================
# Logger
# =============================================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file   = os.path.join(script_dir, "B02B_statistical_results.txt")
sys.stdout = Logger(log_file)

print("=" * 80)
print(f"Analysis Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Analysis: Covariate-Corrected (initOffset removed) 2-Way RM-ANOVA")
print("Factors : TMS_Condition x DiagType  (OffsetGroup removed — corrected out)")
print("DVs     : AbsError_resid, RT_resid, BiasTowardsDiagonal_resid, Precision")
print("=" * 80 + "\n")

# =============================================================================
# Load and Feature-Engineer (identical to B02A / V03_visualBiasesWithCovariate)
# =============================================================================
print("Loading data...")
df = gm.master_df.copy()

print("Preparing features...")

df.rename(columns={
    'TMScond_label':         'TMS',
    'stimPF_label':          'Hemifield',
    'targOriCategory':       'OriCat',
    'hemisphereStimulated':  'HemiStim'
}, inplace=True)

df = df[df['LargeError'] == 0].copy()

# Raw metrics
df['AbsError']            = np.abs(df['errOri_scaled'])
df['AbsInitOffset']       = np.abs(df['initOffset'])
df['BiasTowardsDiagonal'] = df['errOri_scaled'].copy()
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1

# TMS Group
df['TMS_Group'] = df['TMS'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
df['TMS_Group'] = pd.Categorical(df['TMS_Group'], categories=['No TMS', 'TMS'], ordered=True)

df['Hemifield'] = pd.Categorical(df['Hemifield'], categories=['outPF', 'inPF'], ordered=True)
df['OriCat']    = df['OriCat'].astype(int)

# OriCat45
df['OriCat45'] = df['OriCat'].replace({
    30: 'Acute (30+60)',  60: 'Acute (30+60)',
    120: 'Obtuse (120+150)', 150: 'Obtuse (120+150)'
})

# DiagType: nearDiag vs offDiag
def classify_diag(row):
    hemi_stim = row['HemiStim']
    in_pf     = row['Hemifield'] == 'inPF'
    is_acute  = row['OriCat45'] == 'Acute (30+60)'
    if hemi_stim == 'Left':
        return 'nearDiag' if (in_pf and not is_acute) or (not in_pf and is_acute) else 'offDiag'
    else:
        return 'nearDiag' if (in_pf and is_acute) or (not in_pf and not is_acute) else 'offDiag'

df['DiagType'] = df.apply(classify_diag, axis=1)

# TMS Condition
def make_tms_inpf_cat(row):
    if row['TMS_Group'] == 'No TMS':
        return 'NoTMS'
    return f"TMS_{row['Hemifield']}"

df['TMS_Condition'] = df.apply(make_tms_inpf_cat, axis=1)
x_order = ['NoTMS', 'TMS_inPF', 'TMS_outPF']
df['TMS_Condition'] = pd.Categorical(df['TMS_Condition'], categories=x_order, ordered=True)

df['subjID']   = df['subjID'].astype('category')
df['DiagType'] = df['DiagType'].astype('category')

# =============================================================================
# COVARIATE REGRESSION
# Strategy: Pool all NoTMS trials (inPF + outPF) across all subjects.
#           Fit a quadratic polynomial: initOffset -> metric.
#           Residual = raw - poly(offset) + poly(0)
#           poly(0) is the model's prediction at 0° offset — a meaningful,
#           interpretable anchor so residuals stay in natural units.
# =============================================================================
METRICS_TO_RESIDUALIZE = [
    'AbsError',
    'errOri_scaled',
    'BiasTowardsDiagonal',
    'RT'
]

print("\n--- Covariate Regression: initOffset trend removed (NoTMS pooled) ---")

notms_pool    = df[df['TMS_Group'] == 'No TMS'].copy()
x_all         = df['initOffset'].values
fit_polys     = {}   # store for reference

for met in METRICS_TO_RESIDUALIZE:
    resid_col = f"{met}_resid"
    clean     = notms_pool[['initOffset', met]].dropna()
    coeffs    = np.polyfit(clean['initOffset'].values, clean[met].values, 2)
    poly      = np.poly1d(coeffs)
    fit_polys[met] = poly

    anchor = float(poly(0))  # model prediction at exactly 0° offset

    df[resid_col] = df[met].values - poly(x_all)

    print(f"  {met:30s}: poly(0) anchor = {anchor:.3f}, "
          f"coeffs = {coeffs.round(5)}, "
          f"fit on {len(clean)} NoTMS trials")

print("Covariate regression complete.\n")

# Quick diagnostic: correlation of raw vs resid with initOffset
print("--- Residualization check (r with initOffset, should be near 0 after) ---")
for met in METRICS_TO_RESIDUALIZE:
    resid_col = f"{met}_resid"
    r_raw = df[[met, 'initOffset']].dropna().corr().iloc[0, 1]
    r_res = df[[resid_col, 'initOffset']].dropna().corr().iloc[0, 1]
    print(f"  {met:30s}  r_raw={r_raw:+.4f}  r_resid={r_res:+.4f}")
print()

# =============================================================================
# AGGREGATION
# Group by subjID x TMS_Condition x DiagType only (no OffsetGroup).
# Average the residualized metrics per cell.
# =============================================================================
DVS_CONTINUOUS = ['AbsError_resid', 'RT_resid', 'BiasTowardsDiagonal_resid']

print("Aggregating residualized DVs per Subject x TMS_Condition x DiagType...")
agg_means = (df
    .groupby(['subjID', 'TMS_Condition', 'DiagType'], observed=True)[DVS_CONTINUOUS]
    .mean()
    .reset_index())

# =============================================================================
# PRECISION: Fit von Mises mixture model on errOri_scaled_resid
# (consistent with V03_visualBiasesWithCovariate.py)
# =============================================================================
tms_conds  = df['TMS_Condition'].cat.categories
diag_types = df['DiagType'].cat.categories
subjects   = df['subjID'].unique()

mixture_stats = []
print("Fitting Mixture Models for Precision (on errOri_scaled_resid)...")

for subj in subjects:
    for tms in tms_conds:
        for diag in diag_types:
            subset = df[
                (df['subjID']       == subj) &
                (df['TMS_Condition']== tms)  &
                (df['DiagType']     == diag)
            ]
            if len(subset) < 5:
                continue
            angles = subset['errOri_scaled_resid'].dropna().values
            try:
                res = md.fit_model(angles, 'mixtureModel')
                g, K = res
                if g < 0 or g > 1 or K < 0 or K > 50:
                    K = np.nan
            except Exception:
                K = np.nan

            if not np.isnan(K):
                mixture_stats.append({
                    'subjID':        subj,
                    'TMS_Condition': tms,
                    'DiagType':      diag,
                    'Precision':     K
                })

df_prec = pd.DataFrame(mixture_stats)

if not df_prec.empty:
    agg = pd.merge(agg_means, df_prec,
                   on=['subjID', 'TMS_Condition', 'DiagType'], how='left')
else:
    agg = agg_means.copy()
    agg['Precision'] = np.nan

# =============================================================================
# 2-WAY REPEATED-MEASURES ANOVA
# Factors: TMS_Condition (3 levels) x DiagType (2 levels)
# Expected cells per subject: 3 x 2 = 6
# =============================================================================
ALL_DVS = ['AbsError_resid', 'RT_resid', 'BiasTowardsDiagonal_resid', 'Precision']

for dv in ALL_DVS:
    print("\n\n" + "=" * 80)
    print(f"2-WAY RM-ANOVA: {dv} ~ TMS_Condition * DiagType")
    print("=" * 80)

    valid_data = agg.dropna(subset=[dv])
    if valid_data.empty:
        print(f"No valid data for {dv}.")
        continue

    # Balance check: each subject must have all 6 cells
    counts = valid_data.groupby('subjID', observed=True)[dv].count()
    complete_subjs = counts[counts == 6].index
    data_aov = valid_data[valid_data['subjID'].isin(complete_subjs)]
    n = len(complete_subjs)
    print(f"N = {n} subjects with complete data in all 6 cells for {dv}.")

    if n < 5:
        print(f"Skipping {dv} (N too low).")
        continue

    try:
        aov = AnovaRM(data_aov, depvar=dv, subject='subjID',
                      within=['TMS_Condition', 'DiagType'])
        res = aov.fit()
        print(res)
    except Exception as e:
        print(f"ANOVA failed for {dv}: {e}")

    print(f"\nMeans for {dv}:")

    print("\n--- Main Effect: TMS_Condition ---")
    print(data_aov.groupby('TMS_Condition', observed=True)[dv].agg(['mean', 'sem']))

    print("\n--- Main Effect: DiagType ---")
    print(data_aov.groupby('DiagType', observed=True)[dv].agg(['mean', 'sem']))

    print("\n--- Interaction: TMS_Condition x DiagType ---")
    interaction = (data_aov
        .groupby(['TMS_Condition', 'DiagType'], observed=True)[dv]
        .agg(['mean', 'sem'])
        .unstack())
    print(interaction)

    print("\n--- Full cell means (TMS_Condition x DiagType) ---")
    full = (data_aov
        .groupby(['TMS_Condition', 'DiagType'], observed=True)[dv]
        .mean()
        .reset_index())
    print(full.to_string(index=False))

    # --- Planned paired t-tests: TMS_inPF vs TMS_outPF, within each DiagType ---
    print(f"\n--- Paired t-tests: TMS_inPF vs NoTMS and TMS_outPF vs NoTMS, per DiagType ---")
    for diag in diag_types:
        for tms_comp, base in [('TMS_inPF', 'NoTMS'), ('TMS_outPF', 'NoTMS'), ('TMS_inPF', 'TMS_outPF')]:
            try:
                a = data_aov[(data_aov['DiagType'] == diag) & (data_aov['TMS_Condition'] == tms_comp)].set_index('subjID')[dv]
                b = data_aov[(data_aov['DiagType'] == diag) & (data_aov['TMS_Condition'] == base)].set_index('subjID')[dv]
                common = a.index.intersection(b.index)
                t, p = stats.ttest_rel(a.loc[common], b.loc[common])
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                d = (a.loc[common] - b.loc[common]).mean() / (a.loc[common] - b.loc[common]).std()
                print(f"  {diag:10s} | {tms_comp} vs {base}: t({len(common)-1})={t:.3f}, p={p:.4f} {sig}, d={d:.3f}")
            except Exception as e:
                print(f"  {diag:10s} | {tms_comp} vs {base}: failed ({e})")
