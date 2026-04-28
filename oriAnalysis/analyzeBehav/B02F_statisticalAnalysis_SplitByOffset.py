import numpy as np
import pandas as pd
from scipy import stats
import generate_behavdf as gm
import os
import sys
import warnings
import datetime

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
log_file   = os.path.join(script_dir, "B02F_statistical_results.txt")
sys.stdout = Logger(log_file)

print("=" * 80)
print(f"Analysis Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Analysis: One-Way Within-Subject Permutation Test (TMS_Condition) Split by Offset")
print("Offset Threshold: 50 degrees")
print("Metrics: RT, MAE, BiasTowardsDiagonal, Precision")
print("Permutations: 10,000")
print("=" * 80 + "\n")

# =============================================================================
# Load and Feature-Engineer
# =============================================================================
print("Loading data...")
df = gm.master_df.copy()

# Pre-filtering
df = df[df['LargeError'] == 0].copy()
df = df[df['RT'] < 3.0].copy() 

# Rename columns early
df.rename(columns={
    'TMScond_label': 'TMS',        
    'stimPF_label': 'Hemifield',   
    'targOriCategory': 'OriCat',
    'hemisphereStimulated': 'HemiStim'
}, inplace=True)

# Define OffsetGroup (50 degree threshold)
df['AbsInitOffset'] = df['initOffset'].abs()
df['OffsetGroup'] = pd.cut(df['AbsInitOffset'], bins=[0, 50, 90], labels=['Small Offset', 'Large Offset'], include_lowest=True)

# TMS Condition
df['TMS_Group'] = df['TMS'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
def make_tms_cond(row):
    if row['TMS_Group'] in ['No TMS', 'NoTMS']:
        return 'NoTMS'
    return f"TMS_{row['Hemifield']}"

df['TMS_Condition'] = df.apply(make_tms_cond, axis=1)
x_order = ['NoTMS', 'TMS_inPF', 'TMS_outPF']
df['TMS_Condition'] = pd.Categorical(df['TMS_Condition'], categories=x_order, ordered=True)

# Metrics
df['MAE'] = df['abserrorOri']
df['Bias'] = df['errorOri'].copy()
df.loc[df['OriCat'].isin([60, 150]), 'Bias'] *= -1

# --- Compute binary motor direction ---
def get_binary_cw(ori_array):
    if len(ori_array) < 2: return 0
    diffs = np.diff(ori_array)
    diffs[diffs > 90] -= 180
    diffs[diffs < -90] += 180
    return 1 if np.sum(diffs) < 0 else 0

print("Computing motor direction from oriArray...")
df['Binary_CW'] = df['oriArray'].apply(get_binary_cw)

# --- SPLIT DATA AND PROTECT LARGE OFFSET DIRECTION ---
print("Filtering for motor certainty in large offsets...")
small_df = df[df['OffsetGroup'] == 'Small Offset'].copy()
large_correct = df[
    (df['OffsetGroup'] == 'Large Offset') &
    ((
        (df['initOffset'] > 0) & (df['Binary_CW'] == 1)
    ) | (
        (df['initOffset'] < 0) & (df['Binary_CW'] == 0)
    ) | (
        df['initOffset'] == 0
    ))
].copy()

# =============================================================================
# Permutation Test
# =============================================================================

N_PERM = 10000
RNG = np.random.default_rng(42)

def compute_f_stat(data_matrix):
    """Compute one-way repeated-measures F-statistic.
    data_matrix: shape (n_subjects, n_conditions)
    Uses the standard between-conditions MS / error MS formula.
    """
    n_subj, n_cond = data_matrix.shape
    grand_mean = data_matrix.mean()
    cond_means = data_matrix.mean(axis=0)       # (n_cond,)
    subj_means = data_matrix.mean(axis=1)       # (n_subj,)

    ss_cond = n_subj * np.sum((cond_means - grand_mean) ** 2)
    ss_subj = n_cond * np.sum((subj_means - grand_mean) ** 2)
    ss_total = np.sum((data_matrix - grand_mean) ** 2)
    ss_error = ss_total - ss_cond - ss_subj

    df_cond  = n_cond - 1
    df_error = (n_subj - 1) * (n_cond - 1)

    ms_cond  = ss_cond / df_cond
    ms_error = ss_error / df_error

    return ms_cond / ms_error if ms_error > 0 else np.nan


def perm_f_test(data_matrix, n_perm=N_PERM):
    """Within-subject permutation F-test.
    Shuffles condition labels within each subject.
    Returns observed F and permutation p-value.
    """
    obs_f = compute_f_stat(data_matrix)
    null_f = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = data_matrix.copy()
        for s in range(data_matrix.shape[0]):
            shuffled[s] = RNG.permutation(data_matrix[s])
        null_f[i] = compute_f_stat(shuffled)
    p_val = np.mean(null_f >= obs_f)
    return obs_f, p_val


def perm_t_test(v1, v2, n_perm=N_PERM):
    """Paired permutation t-test (two-tailed).
    Shuffles condition assignment within each subject.
    """
    diff = v1 - v2
    obs_t = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
    null_t = np.zeros(n_perm)
    for i in range(n_perm):
        signs = RNG.choice([-1, 1], size=len(diff))
        shuffled_diff = diff * signs
        null_t[i] = np.mean(shuffled_diff) / (np.std(shuffled_diff, ddof=1) / np.sqrt(len(shuffled_diff)))
    p_val = np.mean(np.abs(null_t) >= np.abs(obs_t))
    return obs_t, p_val


def run_stats(subset_df, group_name):
    print("\n" + "#" * 40)
    print(f"### Results for: {group_name}")
    print("#" * 40)

    conditions = ['NoTMS', 'TMS_inPF', 'TMS_outPF']
    metrics = ['RT', 'MAE', 'Bias', 'Precision']

    for dv in metrics:
        print(f"\n--- Permutation Test: {dv} ---")

        # Aggregate by subject x condition
        if dv == 'Precision':
            subj_data = subset_df.groupby(['subjID', 'TMS_Condition'])['errorOri'].std().reset_index()
            subj_data.rename(columns={'errorOri': 'Precision'}, inplace=True)
        else:
            subj_data = subset_df.groupby(['subjID', 'TMS_Condition'])[dv].mean().reset_index()

        # Keep only subjects who have all 3 conditions
        subj_counts = subj_data.groupby('subjID').size()
        full_subjs = sorted(subj_counts[subj_counts == 3].index)

        if len(full_subjs) < len(subj_counts):
            print(f"Warning: Dropped {len(subj_counts) - len(full_subjs)} subjects due to missing conditions.")

        analysis_df = subj_data[subj_data['subjID'].isin(full_subjs)]

        # Build (n_subjects, n_conditions) matrix
        pivot = analysis_df.pivot(index='subjID', columns='TMS_Condition', values=dv)
        pivot = pivot[conditions]  # enforce order
        data_matrix = pivot.values  # shape: (N, 3)

        obs_f, p_val = perm_f_test(data_matrix)
        print(f"  Observed F = {obs_f:.3f},  permutation p = {p_val:.4f}  (N={data_matrix.shape[0]} subjects)")

        if p_val < 0.05:
            print("  *** Significant! Running post-hoc pairwise permutation t-tests (Bonferroni x3) ***")
            pairs = [('NoTMS', 'TMS_inPF'), ('NoTMS', 'TMS_outPF'), ('TMS_inPF', 'TMS_outPF')]
            for c1, c2 in pairs:
                v1 = pivot[c1].values
                v2 = pivot[c2].values
                obs_t, p_raw = perm_t_test(v1, v2)
                p_corr = min(1.0, p_raw * 3)
                sig = "*" if p_corr < 0.05 else ""
                print(f"    {c1} vs {c2}: t={obs_t:.3f}, p_raw={p_raw:.4f}, p_Bonf={p_corr:.4f} {sig}")
        elif p_val < 0.10:
            print("  ~ Trending (p < 0.10). Post-hoc exploratory pairwise tests:")
            pairs = [('NoTMS', 'TMS_inPF'), ('NoTMS', 'TMS_outPF'), ('TMS_inPF', 'TMS_outPF')]
            for c1, c2 in pairs:
                v1 = pivot[c1].values
                v2 = pivot[c2].values
                obs_t, p_raw = perm_t_test(v1, v2)
                p_corr = min(1.0, p_raw * 3)
                sig = "*" if p_corr < 0.05 else ""
                print(f"    {c1} vs {c2}: t={obs_t:.3f}, p_raw={p_raw:.4f}, p_Bonf={p_corr:.4f} {sig}")
        else:
            print("  Not significant.")


print("\nRunning permutation tests (10,000 iterations each — may take a moment)...")
run_stats(small_df, "SMALL OFFSET (0-50°)")
run_stats(large_correct, "LARGE OFFSET (50-90°, correct direction only)")

print("\nDone.")
