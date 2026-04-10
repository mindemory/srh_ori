 import numpy as np
import pandas as pd
import generate_behavdf as gm
import os
import sys
import datetime

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
log_file = os.path.join(script_dir, "B01_dataquality_counts.txt")
sys.stdout = Logger(log_file)

print("="*80)
print(f"Data Quality Report: Trial Counts per Condition")
print(f"Run Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

print("Loading data...")
df = gm.master_df.copy()

# Feature Eng
df.rename(columns={
    'TMScond_label': 'TMS', 
    'stimPF_label': 'Hemifield', 
    'targOriCategory': 'OriCat'
}, inplace=True)

# Ensure categorical types for pivot
tms_order = ['No TMS', 'Early TMS', 'Middle TMS']
hemi_order = ['inPF', 'outPF']
# OriCat is numeric

df['TMS'] = pd.Categorical(df['TMS'], categories=tms_order, ordered=True)
df['Hemifield'] = pd.Categorical(df['Hemifield'], categories=hemi_order, ordered=True)

# Binning
df['AbsInitOffset'] = np.abs(df['initOffset'])
df['AbsOffsetBin'] = pd.cut(df['AbsInitOffset'], bins=[0, 30, 60, 90], labels=["0:30", "30:60", "60:90"], include_lowest=True)

# Display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)

# --- TABLE 1: BROAD SUMMARY (TMS x Hemifield) ---
print("\n" + "="*80)
print("TABLE 1: Broad Summary (TMS x Hemifield)")
print("Aggregated across Orientation and Offset")
print("="*80)

pivot_broad = df.pivot_table(
    index='subjID', 
    columns=['TMS', 'Hemifield'], 
    values='errOri_scaled', 
    aggfunc='count',
    fill_value=0,
    observed=False
)
print(pivot_broad)

# --- TABLE 2: ORIENTATION SUMMARY (TMS x Hemifield x OriCat) ---
print("\n" + "="*80)
print("TABLE 2: Orientation Summary (TMS x Hemifield x OriCat)")
print("Aggregated across Offset Bins")
print("="*80)

pivot_ori = df.pivot_table(
    index='subjID', 
    columns=['TMS', 'Hemifield', 'OriCat'], 
    values='errOri_scaled', 
    aggfunc='count',
    fill_value=0,
    observed=False
)
print(pivot_ori)


# --- TABLE 2b: OFFSET SUMMARY (TMS x Hemifield x AbsOffsetBin) ---
print("\n" + "="*80)
print("TABLE 2b: Offset Summary (TMS x Hemifield x AbsOffsetBin)")
print("Aggregated across Orientation Categories")
print("="*80)

pivot_offset = df.pivot_table(
    index='subjID', 
    columns=['TMS', 'Hemifield', 'AbsOffsetBin'], 
    values='errOri_scaled', 
    aggfunc='count',
    fill_value=0,
    observed=False
)
print(pivot_offset)


# --- TABLE 3: DETAILED SUMMARY (All Factors) ---
print("\n" + "="*80)
print("TABLE 3: Detailed Summary (TMS x Hemifield x OriCat x Bin)")
print("Full Granularity")
print("="*80)

pivot_full = df.pivot_table(
    index='subjID', 
    columns=['TMS', 'Hemifield', 'OriCat', 'AbsOffsetBin'], 
    values='errOri_scaled', 
    aggfunc='count',
    fill_value=0,
    observed=False
)
print(pivot_full)


# --- MISSING DATA CHECK ---
print("\n" + "="*80)
print("MISSING DATA FLAGS (Count == 0)")
print("Checking Table 3 (Full Granularity)")
print("="*80)

missing_mask = pivot_full == 0
if missing_mask.sum().sum() == 0:
    print("No missing cells found! dataset is fully balanced.")
else:
    # Get indices of missing cells
    stacked = pivot_full.stack(['TMS', 'Hemifield', 'OriCat', 'AbsOffsetBin'])
    missing = stacked[stacked == 0]
    
    print(f"Found {len(missing)} missing cells:\n")
    for idx, val in missing.items():
        # idx is (subjID, TMS, Hemifield, OriCat, Bin)
        print(f"Subj {idx[0]}: {idx[1]} | {idx[2]} | Ori {idx[3]} | Bin {idx[4]} -> {val}")

# --- SUMMARY STATS ---
print("\n" + "="*80)
print("Total Trials per Subject")
print("="*80)
total_trials = pivot_full.sum(axis=1)
print(total_trials)

print("\n" + "="*80)
print("Report Complete.")
print("="*80)
