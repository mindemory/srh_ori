import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import generate_behavdf as gm
import modeller as md
import os
import sys
import warnings
import datetime
import socket

warnings.filterwarnings('ignore')
pd.set_option("display.max.columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.precision", 3)

import sys
import os
import io

# ... Imports ...

# Simple Logger to write to both stdout and file
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

# Define output path relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "B02_statistical_results.txt")
sys.stdout = Logger(log_file)

# Context manager to suppress stdout during noisy fitting
# (Modified to work with Logger)
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        # Store the actual logger object if needed, or just replace sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

print("="*80)
print(f"Analysis Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {socket.gethostname()}")
print(f"Output saved to: {log_file}")
print("="*80 + "\n")

print("Loading data...")
df = gm.master_df.copy()

# --- Feature Engineering ---
print("Preparing features...")

# Factor Renaming
df.rename(columns={
    'TMScond_label': 'TMS',        
    'stimPF_label': 'Hemifield',   
    'targOriCategory': 'OriCat',
}, inplace=True)

# 1. Absolute Init Offset and Bins
# Bins: [0, 30, 60, 90]
df['AbsInitOffset'] = np.abs(df['initOffset'])
bins = [0, 30, 60, 90]
labels = ["0:30", "30:60", "60:90"]
df['AbsOffsetBin'] = pd.cut(df['AbsInitOffset'], bins=bins, labels=labels, include_lowest=True)

# 2. Bias Towards Diagonal (Signed)
# Flip sign for 60 and 150
df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1

# 3. AbsError and RT
df['AbsError'] = np.abs(df['errOri_scaled'])
if 'respDuration' in df.columns:        
     df.rename(columns={'respDuration': 'RT'}, inplace=True)

# Handle potential DataFrame wrapper for RT or duplicate columns
# FAIL-SAFE RT EXTRACTION
try:
    rt_col = df['RT']
    if isinstance(rt_col, pd.DataFrame):
        rt_col = rt_col.iloc[:, 0]
    
    if hasattr(rt_col, "values"):
        vals = rt_col.values
    else:
        vals = np.array(rt_col)
        
    vals = np.array(vals)
    if vals.ndim > 1:
        vals = vals.ravel()
        if len(vals) != len(df): 
             vals = np.array(df['RT'])
             if vals.ndim > 1: vals = vals[:, 0]
             
    # Strict list conversion
    df['RT'] = pd.to_numeric(vals.tolist(), errors='coerce')

except Exception as e:
    print(f"WARNING: Could not process RT column. Dropping RT. Error: {e}")
    if 'RT' in df.columns: df.drop(columns=['RT'], inplace=True)
    if 'respDuration' in df.columns: df.drop(columns=['respDuration'], inplace=True)

# Type Conversions
for col in ['TMS', 'Hemifield', 'AbsOffsetBin', 'subjID']:
    df[col] = df[col].astype('category')

# Reorder
df['TMS'] = df['TMS'].cat.reorder_categories(['No TMS', 'Early TMS', 'Middle TMS'], ordered=True)
df['Hemifield'] = df['Hemifield'].cat.reorder_categories(['outPF', 'inPF'], ordered=True)

# --- Data Aggregation & Mixture Modelling ---
print("Aggregating data and fitting Mixture Models (this may take a while)...")

group_cols = ['subjID', 'TMS', 'Hemifield', 'AbsOffsetBin']
agg_rows = []

# Iterating manually to fit models
tms_conds = df['TMS'].unique()
hues = df['Hemifield'].unique() # inPF, outPF
bins = df['AbsOffsetBin'].unique()
subjects = df['subjID'].unique()

for tms in tms_conds:
    for h in hues:
        for b in bins:
            for subj in subjects:
                subset = df[
                    (df['TMS'] == tms) & 
                    (df['Hemifield'] == h) & 
                    (df['AbsOffsetBin'] == b) &
                    (df['subjID'] == subj)
                ]
                
                if len(subset) < 5: continue
                
                # Standard Means
                mean_abs_err = subset['AbsError'].mean()
                mean_bias_diag = subset['BiasTowardsDiagonal'].mean()
                mean_rt = subset['RT'].mean()
                
                # Mixture Fit
                angles = subset['errOri_scaled'].dropna().values
                g, K = np.nan, np.nan
                try:
                    # Suppress fitting output
                    with HiddenPrints():
                        res = md.fit_model(angles, 'mixtureModel')
                    g, K = res
                    # Sanity bounds
                    if g < 0 or g > 1 or K < 0 or K > 100:
                        g, K = np.nan, np.nan
                        
                except Exception:
                    g, K = np.nan, np.nan
                
                agg_rows.append({
                    'subjID': subj,
                    'TMS': tms,
                    'Hemifield': h,
                    'AbsOffsetBin': b,
                    'AbsError': mean_abs_err,
                    'BiasTowardsDiagonal': mean_bias_diag,
                    'RT': mean_rt,
                    'Precision': K,
                    'GuessRate': g
                })

df_agg = pd.DataFrame(agg_rows)
print(f"Aggregation Complete. Shape: {df_agg.shape}")

# --- Statistical Analysis ---
# Metrics and Transforms
# 1. AbsError (Continuous)
# 2. Precision (Log Normalized)
# 3. DiagAttraction (Continuous)
# 4. RT (Log Normalized)
# 5. GuessRate (Arcsin Normalized)

# Apply Transforms
labels = {
    'AbsError': 'Absolute Error (deg)',
    'log_Precision': 'Log Precision (log K)',
    'BiasTowardsDiagonal': 'Diagonal Attraction (deg)',
    'log_RT': 'Log Reaction Time (log s)',
    'asin_GuessRate': 'Transformed Guess Rate (arcsin sqrt g)'
}

df_agg['log_Precision'] = np.log(df_agg['Precision'] + 1e-6) # Avoid log(0)
# Ensure RT is numeric float before log
df_agg['RT'] = pd.to_numeric(df_agg['RT'], errors='coerce')
df_agg['log_RT'] = np.log(df_agg['RT'])
df_agg['asin_GuessRate'] = np.arcsin(np.sqrt(df_agg['GuessRate']))

metrics = list(labels.keys())

# --- LMM Framework ---

for met in metrics:
    print(f"\n\n{'='*100}")
    print(f"ANALYSIS: {labels[met]} ({met})")
    print(f"{'='*100}")
    
    # Drop NaNs for this metric
    data_clean = df_agg.dropna(subset=[met, 'TMS', 'Hemifield', 'AbsOffsetBin', 'subjID']).copy()
    
    # 1. Omnibus LMM
    # Formula: Metric ~ TMS * Hemifield * AbsOffsetBin
    # Random: 1 | subjID
    
    formula = f"{met} ~ TMS * Hemifield * AbsOffsetBin"
    print(f"\n[Omnibus Model] {formula} + (1|Subject)")
    
    try:
        model = smf.mixedlm(formula, data_clean, groups=data_clean["subjID"])
        result = model.fit()
        print(result.summary())
    except Exception as e:
        print(f"LMM Convergence Failed: {e}")
        # Fallback?
        continue

    # 2. Post-hoc Pairwise Comparisons (inPF vs outPF)
    # Requested: "Find specific bins where TMS caused a significant shift"
    # We compare In vs Out for each Cell (TMS x Bin).
    
    print(f"\n[Post-Hoc: Hemifield Effect (In vs Out) per Condition]")
    print(f"{'TMS':<12} | {'Bin':<10} | {'t-stat':<8} | {'p-val':<8} | {'Mean_In':<8} | {'Mean_Out':<8} | {'Sig'}")
    print("-" * 80)
    
    for tms in tms_conds:
        for b in bins:
            subset = data_clean[(data_clean['TMS'] == tms) & (data_clean['AbsOffsetBin'] == b)]
            
            # Pivot to get paired samples
            try:
                pivoted = subset.pivot(index='subjID', columns='Hemifield', values=met)
                in_vals = pivoted['inPF'].dropna()
                out_vals = pivoted['outPF'].dropna()
                
                # Ensure intersection of indices (paired)
                common = in_vals.index.intersection(out_vals.index)
                if len(common) < 3:
                    continue
                    
                in_vec = in_vals.loc[common]
                out_vec = out_vals.loc[common]
                
                t_stat, p_val = stats.ttest_rel(in_vec, out_vec)
                
                sig_symbol = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "." if p_val < 0.1 else ""
                
                print(f"{tms:<12} | {b:<10} | {t_stat:8.3f} | {p_val:8.3f} | {in_vec.mean():8.3f} | {out_vec.mean():8.3f} | {sig_symbol}")
                
            except Exception as e:
                # print(e)
                pass

print("\n" + "="*80)
print("Analysis Complete.")
print("="*80)
