import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy import stats
import generate_behavdf as gm
import os
import sys
import warnings
import datetime
import socket
import itertools
import modeller as md

warnings.filterwarnings('ignore')
pd.set_option("display.max.columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.precision", 3)

# Simple Logger
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
log_file = os.path.join(script_dir, "B02A_statistical_results.txt")
sys.stdout = Logger(log_file)

print("="*80)
print(f"Analysis Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Analysis Focus: Multiple DVs | Factors: OffsetGroup x TMS_Condition x DiagType")
print("="*80 + "\n")

print("Loading data...")
df = gm.master_df.copy()

# --- Feature Engineering Match V03 ---
print("Preparing features...")

# Factor Renaming
df.rename(columns={
    'TMScond_label': 'TMS',        
    'stimPF_label': 'Hemifield',   
    'targOriCategory': 'OriCat',
    'hemisphereStimulated': 'HemiStim'
}, inplace=True)

df = df[df['LargeError'] == 0].copy()

# Metrics
df['AbsError'] = np.abs(df['errOri_scaled'])
df['AbsInitOffset'] = np.abs(df['initOffset'])
df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1

# --- GROUPING LOGIC ---
# TMS Group: No TMS vs TMS (Early/Middle combined)
df['TMS_Group'] = df['TMS'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
df['TMS_Group'] = pd.Categorical(df['TMS_Group'], categories=['No TMS', 'TMS'], ordered=True)

# Offset Group: Small (0-30) vs Medium (30-60) vs Large (60-90)
df['OffsetGroup'] = pd.cut(df['AbsInitOffset'], bins=[0, 30, 60, 90], labels=['Small Offset', 'Medium Offset', 'Large Offset'], include_lowest=True)

df['Hemifield'] = pd.Categorical(df['Hemifield'], categories=['outPF', 'inPF'], ordered=True)
df['OriCat'] = df['OriCat'].astype(int)

# OriCat45: 30+60 (diagonal ~45°) vs 120+150 (diagonal ~135°)
df['OriCat45'] = df['OriCat'].replace({
    30: 'Acute (30+60)', 60: 'Acute (30+60)',
    120: 'Obtuse (120+150)', 150: 'Obtuse (120+150)'
})

# DiagType: nearDiag vs offDiag
def classify_diag(row):
    hemi_stim = row['HemiStim']
    in_pf = row['Hemifield'] == 'inPF'
    is_acute = row['OriCat45'] == 'Acute (30+60)'
    
    if hemi_stim == 'Left':
        return 'nearDiag' if (in_pf and not is_acute) or (not in_pf and is_acute) else 'offDiag'
    else: # Right TMS
        return 'nearDiag' if (in_pf and is_acute) or (not in_pf and not is_acute) else 'offDiag'

df['DiagType'] = df.apply(classify_diag, axis=1)

# TMS Condition: NoTMS vs TMS_inPF vs TMS_outPF
def make_tms_inpf_cat(row):
    if row['TMS_Group'] == 'No TMS':
        return 'NoTMS'
    else:
        return f"TMS_{row['Hemifield']}"
        
df['TMS_Condition'] = df.apply(make_tms_inpf_cat, axis=1)
x_order = ['NoTMS', 'TMS_inPF', 'TMS_outPF']
df['TMS_Condition'] = pd.Categorical(df['TMS_Condition'], categories=x_order, ordered=True)

# Reorder Factors
df['subjID'] = df['subjID'].astype('category')
df['OffsetGroup'] = df['OffsetGroup'].astype('category')
df['DiagType'] = df['DiagType'].astype('category')


# --- Aggregation ---
print("\nAggregating Mean DVs per Subject x OffsetGroup x TMS_Condition x DiagType...")
agg_means = df.groupby(['subjID', 'OffsetGroup', 'TMS_Condition', 'DiagType'], observed=True)[['AbsError', 'RT', 'BiasTowardsDiagonal']].mean().reset_index()

# Compute Precision
mixture_stats = []
subjects = df['subjID'].unique()
offsets = df['OffsetGroup'].cat.categories
tms_conds = df['TMS_Condition'].cat.categories
diag_types = df['DiagType'].cat.categories

print("Fitting Mixture Models for Precision...")
for subj in subjects:
    for off in offsets:
        for tms in tms_conds:
            for diag in diag_types:
                subset = df[
                    (df['subjID'] == subj) & 
                    (df['OffsetGroup'] == off) & 
                    (df['TMS_Condition'] == tms) & 
                    (df['DiagType'] == diag)
                ]
                
                if len(subset) < 5:
                    continue
                
                angles = subset['errOri_scaled'].dropna().values
                try:
                    res = md.fit_model(angles, 'mixtureModel')
                    g, K = res
                    if g < 0 or g > 1 or K < 0 or K > 50: K = np.nan
                except:
                    K = np.nan
                
                if not np.isnan(K):
                    mixture_stats.append({
                        'subjID': subj,
                        'OffsetGroup': off,
                        'TMS_Condition': tms,
                        'DiagType': diag,
                        'Precision': K
                    })

df_prec = pd.DataFrame(mixture_stats)

if not df_prec.empty:
    agg = pd.merge(agg_means, df_prec, on=['subjID', 'OffsetGroup', 'TMS_Condition', 'DiagType'], how='left')
else:
    agg = agg_means
    agg['Precision'] = np.nan

dvs = ['AbsError', 'RT', 'BiasTowardsDiagonal', 'Precision']

for dv in dvs:
    # --- Statistical Analysis: 3-Way RM-ANOVA ---
    print("\n\n" + "="*80)
    print(f"3-WAY RM-ANOVA: {dv} ~ OffsetGroup * TMS_Condition * DiagType")
    print("="*80)

    # Check balance for this DV
    valid_data = agg.dropna(subset=[dv])
    if valid_data.empty:
        print(f"No valid data for {dv}.")
        continue

    counts = valid_data.groupby('subjID')[dv].count()
    # Expected groups per subject: 3 (Offset) * 3 (TMS_Condition) * 2 (DiagType) = 18
    valid = counts[counts == 18].index 

    data_aov = valid_data[valid_data['subjID'].isin(valid)]
    n = len(valid)
    print(f"N = {n} subjects with complete data in all 18 cells for {dv}.")

    if n < 5:
        print(f"Skipping {dv} (N too low).")
        continue

    try:
        aov = AnovaRM(data_aov, depvar=dv, subject='subjID', within=['OffsetGroup', 'TMS_Condition', 'DiagType'])
        res = aov.fit()
        print(res)
    except Exception as e:
        print(f"ANOVA Failed for {dv}: {e}")

    print(f"\nMean {dv} per Main Effect and Interaction:")

    print("\n--- Main Effect: OffsetGroup ---")
    print(data_aov.groupby('OffsetGroup', observed=True)[dv].mean())

    print("\n--- Main Effect: TMS_Condition ---")
    print(data_aov.groupby('TMS_Condition', observed=True)[dv].mean())

    print("\n--- Main Effect: DiagType ---")
    print(data_aov.groupby('DiagType', observed=True)[dv].mean())

    print("\n--- Interaction: OffsetGroup x TMS_Condition ---")
    print(data_aov.groupby(['OffsetGroup', 'TMS_Condition'], observed=True)[dv].mean().unstack())

    print("\n--- Interaction: OffsetGroup x DiagType ---")
    print(data_aov.groupby(['OffsetGroup', 'DiagType'], observed=True)[dv].mean().unstack())

    print("\n--- Interaction: TMS_Condition x DiagType ---")
    print(data_aov.groupby(['TMS_Condition', 'DiagType'], observed=True)[dv].mean().unstack())

    print("\n--- 3-Way Interaction: OffsetGroup x TMS_Condition x DiagType ---")
    # Reset index for better printing
    summary_3way = data_aov.groupby(['OffsetGroup', 'TMS_Condition', 'DiagType'], observed=True)[dv].mean().reset_index()
    print(summary_3way.to_string(index=False))
