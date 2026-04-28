import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import warnings
import generate_behavdf as gm

warnings.filterwarnings('ignore')

# ============================================================
# Core RSA Utilities (Optimized for Permutations)
# ============================================================

def get_upper_tri(mat):
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


CENTERS_4CAT = np.array([30, 60, 120, 150])


def build_model_rdms(x_sorted):
    """
    Given sorted target orientations for a window, return three model RDM
    upper-triangle vectors: H1 (continuous), H2 (4-cat), H3 (2-cat).
    """
    x_cat   = CENTERS_4CAT[np.argmin(np.abs(CENTERS_4CAT - x_sorted[:, None]), axis=1)]
    x_broad = np.where(x_sorted < 90, 45, 135)
    H1 = get_upper_tri(np.abs(x_sorted[:, None] - x_sorted[None, :]))
    H2 = get_upper_tri(np.abs(x_cat[:, None]    - x_cat[None, :]))
    H3 = get_upper_tri(np.abs(x_broad[:, None]  - x_broad[None, :]))
    return H1, H2, H3


def compute_partial_rsa_with_perms(x_ori, y_resp, n_perms=1000, min_pairs=50):
    """
    Compute r(H3), r(H2|H3), r(H1|H2,H3) for a set of trials, and run `n_perms` 
    shuffles of y_resp to build the empirical chance distributions.
    Returns: dict of real r's and their empirical 95% CIs.
    """
    if len(x_ori) < 10:
        return None

    sort_idx = np.argsort(x_ori)
    x_s = x_ori[sort_idx]
    y_s = y_resp[sort_idx]

    H1, H2, H3 = build_model_rdms(x_s)

    # Variance check — skip degenerate windows where a model is constant
    if np.std(H3) < 1e-8:
        return None

    # ---- 1. Precompute Ranks and Regression Matrices for X-side ----
    rh1 = stats.rankdata(H1)
    rh2 = stats.rankdata(H2)
    rh3 = stats.rankdata(H3)
    
    N = len(rh1)
    if N < min_pairs: 
        return None
        
    ones = np.ones(N)
    
    # Precompute X2 (for H2 | H3)
    X2 = np.column_stack([ones, rh3])
    X2_pinv = np.linalg.pinv(X2)  # shape (2, N)
    beta_x_2 = X2_pinv @ rh2
    res_x2 = rh2 - X2 @ beta_x_2
    res_x2_m = res_x2 - res_x2.mean()
    res_x2_norm = np.linalg.norm(res_x2_m) + 1e-12
    
    # Precompute X3 (for H1 | H2, H3)
    X3 = np.column_stack([ones, rh2, rh3])
    X3_pinv = np.linalg.pinv(X3)     # shape (3, N)
    beta_x_1 = X3_pinv @ rh1
    res_x1 = rh1 - X3 @ beta_x_1
    res_x1_m = res_x1 - res_x1.mean()
    res_x1_norm = np.linalg.norm(res_x1_m) + 1e-12
    
    rh3_mean = rh3.mean()
    rh3_norm = np.linalg.norm(rh3 - rh3_mean) + 1e-12

    rh2_mean = rh2.mean()
    rh2_norm = np.linalg.norm(rh2 - rh2_mean) + 1e-12

    rh1_mean = rh1.mean()
    rh1_norm = np.linalg.norm(rh1 - rh1_mean) + 1e-12

    # ---- 2. Define the exact fast evaluator ----
    def evaluate_rdm(y_vec):
        data_vec = get_upper_tri(np.abs(y_vec[:, None] - y_vec[None, :]))
        ry = stats.rankdata(data_vec)
        ry_mean = ry.mean()
        ry_cent = ry - ry_mean
        ry_norm = np.linalg.norm(ry_cent) + 1e-12
        
        # r(H3)
        r3 = np.dot(ry_cent, rh3 - rh3_mean) / (ry_norm * rh3_norm)
        r3_dir = r3
        
        # r(H2|H3)
        beta_y_2 = X2_pinv @ ry
        res_y2 = ry - X2 @ beta_y_2
        res_y2_m = res_y2 - res_y2.mean()
        r2 = np.dot(res_y2_m, res_x2_m) / (np.linalg.norm(res_y2_m) * res_x2_norm)
        r2_dir = np.dot(ry_cent, rh2 - rh2_mean) / (ry_norm * rh2_norm)
        
        # r(H1|H2, H3)
        beta_y_3 = X3_pinv @ ry
        res_y3 = ry - X3 @ beta_y_3
        res_y3_m = res_y3 - res_y3.mean()
        r1 = np.dot(res_y3_m, res_x1_m) / (np.linalg.norm(res_y3_m) * res_x1_norm)
        r1_dir = np.dot(ry_cent, rh1 - rh1_mean) / (ry_norm * rh1_norm)
        
        return r3, r2, r1, r3_dir, r2_dir, r1_dir

    # ---- 3. Compute Real Values ----
    real_r3, real_r2, real_r1, dir_r3, dir_r2, dir_r1 = evaluate_rdm(y_s)

    return dict(r_H3=real_r3, r_H2_H3=real_r2, r_H1_H2_H3=real_r1,
                r_dirH3=dir_r3, r_dirH2=dir_r2, r_dirH1=dir_r1)

def compute_empirical_rdm(df_in, n_bins=36):
    if len(df_in) < 10:
        return np.full((n_bins, n_bins), np.nan)
    bins = np.linspace(0, 180, n_bins+1)
    df_in['targ_bin'] = pd.cut(df_in['targOriActual'], bins=bins, labels=False, include_lowest=True)
    binned = df_in.groupby('targ_bin')['finalOri'].mean()
    full_resp = np.full(n_bins, np.nan)
    for b, val in binned.items():
        if pd.notnull(b):
            full_resp[int(b)] = val
    return np.abs(full_resp[:, None] - full_resp[None, :])



# ============================================================
# Sliding-window RSA across initOffset
# ============================================================

def sliding_window_rsa(df_sub, window_centers, half_win=7.5, min_pairs=50):
    """
    For one subject's dataframe, compute partial RSA at each window.
    """
    r_arr = {'r3': np.full(len(window_centers), np.nan),
             'r2': np.full(len(window_centers), np.nan),
             'r1': np.full(len(window_centers), np.nan),
             'r3_dir': np.full(len(window_centers), np.nan),
             'r2_dir': np.full(len(window_centers), np.nan),
             'r1_dir': np.full(len(window_centers), np.nan)}

    for j, ctr in enumerate(window_centers):
        mask = (df_sub['initOffset'] >= ctr - half_win) & \
               (df_sub['initOffset'] <  ctr + half_win)
        df_w = df_sub[mask]
        res = compute_partial_rsa_with_perms(
            df_w['targOriActual'].values,
            df_w['finalOri'].values,
            n_perms=0, # no permutations
            min_pairs=min_pairs
        )
        if res is not None:
            r_arr['r3'][j] = res['r_H3']
            r_arr['r2'][j] = res['r_H2_H3']
            r_arr['r1'][j] = res['r_H1_H2_H3']
            r_arr['r3_dir'][j] = res['r_dirH3']
            r_arr['r2_dir'][j] = res['r_dirH2']
            r_arr['r1_dir'][j] = res['r_dirH1']

    return r_arr
def plot_sliding_rsa():
    # ----- Load data -----
    df = gm.master_df.copy()
    df = df[df['LargeError'] == 0]
    df.reset_index(drop=True, inplace=True)

    subjects = sorted(df['subjID'].unique())

    # Window parameters
    # Window parameters
    window_centers = np.arange(-87.5, 90, 5)   # centers every 5 deg
    half_win = 12.5                            # 25 deg bin size

    print(f"Running sliding-window RSA for {len(subjects)} subjects (all trials, separated by TMS cond)...")
    
    conds = ['no', 'in', 'out']
    keys = ['r3', 'r2', 'r1', 'r3_dir', 'r2_dir', 'r1_dir']
    res = {c: {k: [] for k in keys} for c in conds}

    for sid in subjects:
        df_sub = df[df['subjID'] == sid]
        
        # 1) No TMS
        df_no = df_sub[df_sub['TMScond'] == 1]
        r_no = sliding_window_rsa(df_no, window_centers, half_win=half_win, min_pairs=30)
        
        # 2) TMS inPF
        df_in = df_sub[(df_sub['TMScond'] > 1) & (df_sub['stimPF'] == 1)]
        r_in = sliding_window_rsa(df_in, window_centers, half_win=half_win, min_pairs=30)
        
        # 3) TMS outPF
        df_out = df_sub[(df_sub['TMScond'] > 1) & (df_sub['stimPF'] == 0)]
        r_out = sliding_window_rsa(df_out, window_centers, half_win=half_win, min_pairs=30)
        
        for k in keys:
            res['no'][k].append(r_no[k])
            res['in'][k].append(r_in[k])
            res['out'][k].append(r_out[k])
        
        print(f"  sub{sid:02d} done.")

    # Convert to arrays
    for c in conds:
        for k in keys:
            res[c][k] = np.array(res[c][k])

    # ----------------------------------------------------------------
    # Generate Model RDMs for visualization (Row 1)
    # ----------------------------------------------------------------
    x_plot = np.linspace(0, 179, 180)
    x_cat = CENTERS_4CAT[np.argmin(np.abs(CENTERS_4CAT - x_plot[:, None]), axis=1)]
    x_broad = np.where(x_plot < 90, 45, 135)
    
    H1_full = np.abs(x_plot[:, None] - x_plot[None, :])
    H2_full = np.abs(x_cat[:, None] - x_cat[None, :])
    H3_full = np.abs(x_broad[:, None] - x_broad[None, :])

    # ----------------------------------------------------------------
    # Plot — 3 rows x 3 cols
    # ----------------------------------------------------------------
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Representational Structure as a Function of Motor Offset', fontsize=16, y=1.02)

    # --- Row 0: Model RDMs ---
    models = [
        (H3_full, 'H3: 2-Category (Hemifield) Model'),
        (H2_full, 'H2: 4-Category Model'),
        (H1_full, 'H1: Continuous Spatial Model')
    ]
    for col_idx, (mat, title) in enumerate(models):
        ax = axs[0, col_idx]
        im = ax.imshow(mat, cmap='RdBu_r', origin='lower', extent=[0, 180, 0, 180])
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Target Orientation (deg)', fontsize=11)
        ax.set_ylabel('Target Orientation (deg)', fontsize=11)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_yticks([0, 45, 90, 135, 180])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Absolute Angular Difference')

    # --- Row 1: Sliding Window Traces (Partial Rank) ---
    colors = {
        'no': ('#7f8c8d', 'No TMS'),      # Grey
        'in': ('#c0392b', 'TMS inPF'),    # Red
        'out': ('#2980b9', 'TMS outPF')   # Blue
    }
    
    for col_idx, (k, title) in enumerate([
        ('r3', 'Partial r(H3)'),
        ('r2', 'Partial r(H2|H3)'),
        ('r1', 'Partial r(H1|H2,H3)'),
    ]):
        ax = axs[1, col_idx]
        
        for c in conds:
            r_mat = res[c][k]
            color, label = colors[c]
            
            n_valid = np.sum(~np.isnan(r_mat), axis=0)
            mean_r  = np.nanmean(r_mat, axis=0)
            sem_r   = np.nanstd(r_mat, axis=0) / np.sqrt(np.maximum(n_valid, 1))
            
            ok = n_valid >= 3
            x, m, s = window_centers[ok], mean_r[ok], sem_r[ok]
            
            ax.plot(x, m, color=color, linewidth=2, label=label)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('initOffset (deg)', fontsize=12)
        ax.set_ylabel('Partial Spearman ρ', fontsize=12)
        # ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.axvspan(-25, 25, color='gray', alpha=0.08)
        if col_idx == 0:
            ax.legend(fontsize=10, loc='best', frameon=False)

    # --- Row 2: Sliding Window Traces (Direct Corrs) ---
    for col_idx, (k, title) in enumerate([
        ('r3_dir', 'Direct r(Data, H3)'),
        ('r2_dir', 'Direct r(Data, H2)'),
        ('r1_dir', 'Direct r(Data, H1)'),
    ]):
        ax = axs[2, col_idx]
        
        for c in conds:
            r_mat = res[c][k]
            color, label = colors[c]
            
            n_valid = np.sum(~np.isnan(r_mat), axis=0)
            mean_r  = np.nanmean(r_mat, axis=0)
            sem_r   = np.nanstd(r_mat, axis=0) / np.sqrt(np.maximum(n_valid, 1))
            
            ok = n_valid >= 3
            x, m, s = window_centers[ok], mean_r[ok], sem_r[ok]
            
            ax.plot(x, m, color=color, linewidth=2, label=label)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('initOffset (deg)', fontsize=12)
        ax.set_ylabel('Direct Spearman ρ', fontsize=12)
        # ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.axvspan(-25, 25, color='gray', alpha=0.08)

    plt.tight_layout()
    plt.savefig('/d/DATD/datd/TMS_ori/Figures/V09_RSA_OffsetSplit.png', dpi=150, bbox_inches='tight')
    print("Saved: V09_RSA_OffsetSplit.png")
    plt.show()


if __name__ == "__main__":
    plot_sliding_rsa()
