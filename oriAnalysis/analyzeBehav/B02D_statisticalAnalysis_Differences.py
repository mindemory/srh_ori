import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy import stats
import os
import sys
import warnings
import datetime

# Use absolute path handling
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import generate_behavdf as gm
import modeller as md

warnings.filterwarnings('ignore')
pd.set_option("display.max.columns", None)
pd.set_option("display.width", 1000)

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
log_file   = os.path.join(script_dir, "B02D_statistical_results.txt")
sys.stdout = Logger(log_file)

print("=" * 80)
print(f"Analysis Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Analysis: Fig 5 Differences (inPF - outPF)")
print("Factors : TMS_Group (TMS vs NoTMS) x OriMapped (30,60,120,150)")
print("=" * 80 + "\n")

# --- 1. Load Data & Feature Engineering ---
df = gm.master_df.copy()

df.rename(columns={
    'TMScond_label':         'TMS',
    'stimPF_label':          'Hemifield',
    'targOriCategory':       'OriCat',
    'hemisphereStimulated':  'HemiStim'
}, inplace=True)

df = df[df['LargeError'] == 0].copy()

df['AbsError']            = np.abs(df['errOri_scaled'])
df['BiasTowardsDiagonal'] = df['errOri_scaled'].copy()
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1

df['TMS_Group'] = df['TMS'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
df['TMS_Group'] = pd.Categorical(df['TMS_Group'], categories=['No TMS', 'TMS'], ordered=True)

df['Hemifield'] = pd.Categorical(df['Hemifield'], categories=['outPF', 'inPF'], ordered=True)
df['OriCat']    = df['OriCat'].astype(int)

def map_ori(row):
    return 180 - row['OriCat'] if row['HemiStim'] == 'Right' else row['OriCat']

df['OriMapped'] = df.apply(map_ori, axis=1)

# --- 2. Covariate Residualization (using purely 0-anchored fit like V03 Fig 5) ---
METRICS_TO_RESIDUALIZE = ['AbsError', 'errOri_scaled', 'BiasTowardsDiagonal', 'RT']

notms_pool = df[df['TMS_Group'] == 'No TMS'].copy()
x_all      = df['initOffset'].values

for met in METRICS_TO_RESIDUALIZE:
    resid_col = f"{met}_resid"
    clean     = notms_pool[['initOffset', met]].dropna()
    coeffs    = np.polyfit(clean['initOffset'].values, clean[met].values, 2)
    poly      = np.poly1d(coeffs)
    
    # Residuals explicitly 0-centered so differences are unconfounded
    df[resid_col] = df[met].values - poly(x_all)

# --- 3. Aggregate Means per Hemisphere & Orientation ---
DVS_CONTINUOUS = ['AbsError_resid', 'RT_resid', 'BiasTowardsDiagonal_resid']

agg_means = df.groupby(['subjID', 'TMS_Group', 'OriMapped', 'Hemifield'], observed=True)[DVS_CONTINUOUS].mean().reset_index()

# --- 4. Mixture Models for Precision ---
subjs = df['subjID'].unique()
ori_cats = [30, 60, 120, 150]
hemis = ['inPF', 'outPF']
tms_groups = ['TMS', 'No TMS']

prec_records = []
print("Fitting Mixture Models for Precision...")
for subj in subjs:
    for tms_grp in tms_groups:
        for ori in ori_cats:
            for hemi in hemis:
                sub_df = df[(df['subjID'] == subj) & (df['TMS_Group'] == tms_grp) & 
                            (df['OriMapped'] == ori) & (df['Hemifield'] == hemi)]
                if len(sub_df) < 5: continue
                angles = sub_df['errOri_scaled_resid'].dropna().values
                try:
                    res = md.fit_model(angles, 'mixtureModel')
                    g, K = res
                    if g < 0 or g > 1 or K < 0 or K > 50: K = np.nan
                except:
                    K = np.nan
                if not np.isnan(K):
                    prec_records.append({'subjID': subj, 'TMS_Group': tms_grp, 'OriMapped': ori, 'Hemifield': hemi, 'Precision': K})

df_prec = pd.DataFrame(prec_records)
if not df_prec.empty:
    agg = pd.merge(agg_means, df_prec, on=['subjID', 'TMS_Group', 'OriMapped', 'Hemifield'], how='left')
else:
    agg = agg_means
    agg['Precision'] = np.nan

# --- 5. Calculate Differences (inPF - outPF) ---
print("Computing differences (inPF - outPF)...")
diff_records = []
for subj in subjs:
    for tms_grp in tms_groups:
        for ori in ori_cats:
            sub = agg[(agg['subjID'] == subj) & (agg['TMS_Group'] == tms_grp) & (agg['OriMapped'] == ori)]
            if len(sub) == 2:
                in_val = sub[sub['Hemifield'] == 'inPF'].iloc[0]
                out_val = sub[sub['Hemifield'] == 'outPF'].iloc[0]
                rec = {'subjID': subj, 'TMS_Group': tms_grp, 'OriMapped': ori}
                for m in DVS_CONTINUOUS + ['Precision']:
                    if m in in_val and not pd.isna(in_val[m]) and not pd.isna(out_val[m]):
                        rec[m] = in_val[m] - out_val[m]
                diff_records.append(rec)

df_diff = pd.DataFrame(diff_records)
df_diff['OriMapped'] = df_diff['OriMapped'].astype(str)

# --- 6. STATISTICAL ANALYSIS ---
print("\n" + "="*80)
print("STATISTICAL RESULTS")
print("="*80)

for dv in DVS_CONTINUOUS + ['Precision']:
    print("\n\n" + "-"*60)
    print(f"METRIC: {dv}")
    print("-"*60)
    
    valid_data = df_diff.dropna(subset=[dv])
    
    # 2-WAY RM-ANOVA
    # Filter subjects that have complete data for all 8 cells (2 TMS x 4 Ori)
    counts = valid_data.groupby('subjID', observed=True)[dv].count()
    complete_subjs = counts[counts == 8].index
    data_aov = valid_data[valid_data['subjID'].isin(complete_subjs)]
    n = len(complete_subjs)
    
    print(f"\n[ 2-WAY RM-ANOVA on inPF-outPF differences | N = {n} ]")
    if n >= 5:
        try:
            aov = AnovaRM(data_aov, depvar=dv, subject='subjID', within=['TMS_Group', 'OriMapped'])
            res = aov.fit()
            print(res)
        except Exception as e:
            print(f"ANOVA failed: {e}")
    else:
        print("Not enough subjects with complete data for ANOVA.")
        
    # PAIRED T-TESTS (TMS vs No TMS) for EACH ANGLE
    print("\n[ POST-HOC PAIRED t-TESTS: TMS vs No TMS ]")
    for ori in ['30', '60', '120', '150']:
        sub_valid = valid_data[valid_data['OriMapped'] == ori]
        a = sub_valid[sub_valid['TMS_Group'] == 'TMS'].set_index('subjID')[dv]
        b = sub_valid[sub_valid['TMS_Group'] == 'No TMS'].set_index('subjID')[dv]
        common = a.index.intersection(b.index)
        
        if len(common) > 2:
            t, p = stats.ttest_rel(a.loc[common], b.loc[common])
            diff_mean = (a.loc[common] - b.loc[common]).mean()
            d = diff_mean / (a.loc[common] - b.loc[common]).std()
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f"  Ori {ori:>3s}° : t({len(common)-1:2d}) = {t:6.3f}, p = {p:.4f} {sig:3s} | d = {d:.3f} | MeanDiff = {diff_mean:+.3f}")
        else:
            print(f"  Ori {ori:>3s}° : Insufficient pairs ({len(common)})")
            
print("\nDone! Results saved to B02D_statistical_results.txt")
