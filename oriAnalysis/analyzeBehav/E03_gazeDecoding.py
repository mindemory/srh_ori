"""
E03_gazeDecoding.py - Time-Resolved Gaze Classification (Visual Field: Left vs Right).

Classification: LeftVF (0) vs RightVF (1)
Preprocessing:
1. Interpolation to common 100Hz grid
2. Drift Removal: Subtract Mean Trace (Global Mean across trials)
3. Strict Rejection: Exclude trials > 2.0 deg (after drift removal) during Sample/Delay

- Sliding Window: 200ms
- Model: SVC (RBF kernel)
- CV: 10-fold ShuffleSplit
- Metrics: Accuracy, F1-Score
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
import generate_behavdf as gm
from GazeAnalyzer import GazeAnalyzer
import time as time_module

# --- Configuration ---
WINDOW_MS = 200        
DOWNSAMPLE_RATE = 10   
N_SPLITS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing Config
INITIAL_EXTRACT_THRESH = 5.0 # Load broad range first
STRICT_REJECT_THRESH = 2.0   # Reject if deviation > 2 deg
EPOCH_END_EVENT = 4          # Response Cue (end of Delay 2)

datd_path = gm.p['datd']
analyzer = GazeAnalyzer(datd_path=datd_path, master_df=gm.master_df)

out_dir = os.path.join(datd_path, 'figures', 'gazeDecoding')
os.makedirs(out_dir, exist_ok=True)

# Subjects
all_subj_ids = sorted(gm.master_df['subjID'].unique())
# all_subj_ids = [1] # Debug

print(f"=== Starting Gaze Classification (Visual Field, De-trended) ===", flush=True)
print(f"Subjects: {len(all_subj_ids)}", flush=True)
print(f"Criteria: Reject > {STRICT_REJECT_THRESH} deg, Subtract Mean Trace", flush=True)

group_results = []

for subj_id in all_subj_ids:
    print(f"\n--- Subject {subj_id} ---", flush=True)
    
    try:
        # 1. Load Data
        gaze_data = analyzer.load_gaze_data(subj_id)
        behav_df = analyzer.get_behavioral_df(subj_id)
        filtered_df = analyzer.filter_trials(behav_df, TMScond=1) # NoTMS
        
        if len(filtered_df) < 40:
            print(f"  Not enough trials ({len(filtered_df)}), skipping.")
            continue
            
        # 2. Extract Traces (Relaxed threshold first)
        traces_df = analyzer.extract_traces(
            df=filtered_df, 
            gaze_data=gaze_data, 
            gaze_threshold=INITIAL_EXTRACT_THRESH, 
            downsample=DOWNSAMPLE_RATE,
            baseline_window=(-1.0, 0.0)
        )
        
        if len(traces_df) == 0:
            print("  No traces extracted.")
            continue
            
        # 3. Drift Removal & Strict Filtering
        
        trials = sorted(traces_df['Trial'].unique())
        
        # Create common time axis (100Hz)
        common_times = np.arange(-0.5, 4.30, 0.01) # 100Hz
        
        X_list = []
        Y_list = []
        valid_trials_extract = []
        
        for trial in trials:
            trial_data = traces_df[traces_df['Trial'] == trial]
            if len(trial_data) < 2: continue
            
            x_interp = np.interp(common_times, trial_data['Time'].values, trial_data['X'].values, left=np.nan, right=np.nan)
            y_interp = np.interp(common_times, trial_data['Time'].values, trial_data['Y'].values, left=np.nan, right=np.nan)
            
            # Simple check for edge NaNs: if mostly valid, keep.
            # But filtering step handles NaNs.
            
            X_list.append(x_interp)
            Y_list.append(y_interp)
            valid_trials_extract.append(trial)
            
        X_full = np.array(X_list) 
        Y_full = np.array(Y_list)
        trials = valid_trials_extract
        times = common_times

        # --- Drift Removal (Step 1) ---
        mean_X = np.nanmean(X_full, axis=0)
        mean_Y = np.nanmean(Y_full, axis=0)
        
        X_detrend = X_full - mean_X
        Y_detrend = Y_full - mean_Y
        
        # --- Strict Filtering (Step 2) ---
        if 4 in analyzer.mean_event_times:
            t_max_epoch = analyzer.mean_event_times[4]
        else:
            t_max_epoch = 4.25
            
        t_indices = [i for i, t in enumerate(times) if 0 <= t <= t_max_epoch]
        
        valid_trials_mask = []
        for i in range(len(trials)):
            x_seg = X_detrend[i, t_indices]
            y_seg = Y_detrend[i, t_indices]
            
            if np.isnan(x_seg).any():
                valid_trials_mask.append(False)
                continue
                
            dist = np.sqrt(x_seg**2 + y_seg**2)
            if np.max(dist) > STRICT_REJECT_THRESH:
                valid_trials_mask.append(False)
            else:
                valid_trials_mask.append(True)
                
        valid_trials_mask = np.array(valid_trials_mask)
        accepted_trials = np.array(trials)[valid_trials_mask]
        
        if len(accepted_trials) < 20:
            print(f"  Too few trials after strict filtering ({len(accepted_trials)}), skipping.")
            continue
            
        print(f"  Trials accepted: {len(accepted_trials)} / {len(trials)}")
        
        # Use De-trended & Filtered matrices
        X_final = X_detrend[valid_trials_mask]
        Y_final = Y_detrend[valid_trials_mask]
        
        # 4. Create Targets (Visual Field: Left vs Right)
        # Need to match Trial ID to behavioral metadata
        # traces_df['Trial'] is the row index in behav_df (reset).
        
        behav_df_reset = behav_df.reset_index(drop=True)
        # Select subset matching accepted trials
        behav_subset = behav_df_reset.iloc[accepted_trials]
        
        target_labels = []
        final_indices = []
        
        # Visual Field Logic:
        # Left Hemi: inPF -> RightVF, outPF -> LeftVF
        # Right Hemi: inPF -> LeftVF, outPF -> RightVF
        
        for idx, (_, row) in enumerate(behav_subset.iterrows()):
            hemi = row['hemisphereStimulated']
            pf = row['stimPF_label']
            
            label = None
            if hemi == 'Left':
                if pf == 'inPF': label = 1 # RightVF
                elif pf == 'outPF': label = 0 # LeftVF
            elif hemi == 'Right':
                if pf == 'inPF': label = 0 # LeftVF
                elif pf == 'outPF': label = 1 # RightVF
                
            if label is not None:
                target_labels.append(label)
                final_indices.append(idx)
                
        if not final_indices:
            print("  No valid class targets.")
            continue
            
        y = np.array(target_labels)
        X_final = X_final[final_indices]
        Y_final = Y_final[final_indices]
        
        print(f"  Final trials for decoding: {len(y)}")
        
        # 5. Decoding Loop
        n_times = len(times)
        win_samples = int(WINDOW_MS / DOWNSAMPLE_RATE)
        half_win = int(win_samples / 2)
        
        cv = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        acc_over_time = np.zeros(n_times)
        f1_over_time = np.zeros(n_times)
        valid_time_mask = np.zeros(n_times, dtype=bool)
        
        t_max_analysis = t_max_epoch + 0.05
        
        start_time = time_module.time()
        
        for t_idx, t_val in enumerate(times):
            if t_val < -0.5 or t_val > t_max_analysis:
                continue
            
            t_start = t_idx - half_win
            t_end = t_idx + half_win
            
            if t_start < 0 or t_end > n_times:
                continue
                
            valid_time_mask[t_idx] = True
            
            feat_X = X_final[:, t_start:t_end]
            feat_Y = Y_final[:, t_start:t_end]
            
            if np.isnan(feat_X).any():
                feat_X = np.nan_to_num(feat_X)
                feat_Y = np.nan_to_num(feat_Y)
            
            features = np.hstack([feat_X, feat_Y])
            
            fold_accs = []
            fold_f1s = []
            
            for train_idx, test_idx in cv.split(features, y):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Check target balance
                if len(np.unique(y_test)) < 2:
                    # Skip fold if test set has only one class (unlikely with ShuffleSplit)
                    continue
                    
                clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                fold_accs.append(accuracy_score(y_test, y_pred))
                fold_f1s.append(f1_score(y_test, y_pred, average='binary'))
                
            if fold_accs:
                acc_over_time[t_idx] = np.mean(fold_accs)
                f1_over_time[t_idx] = np.mean(fold_f1s)
            
        elapsed = time_module.time() - start_time
        print(f"  Classification complete in {elapsed:.1f}s", flush=True)
        
        valid_idx = np.where(valid_time_mask)[0]
        subj_results = pd.DataFrame({
            'Time': [times[i] for i in valid_idx],
            'Accuracy': acc_over_time[valid_idx],
            'F1': f1_over_time[valid_idx],
            'SubjID': subj_id
        })
        group_results.append(subj_results)
        
        # Plot Individual
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(subj_results['Time'], subj_results['Accuracy'], 'b-', lw=2, label='Accuracy')
        ax.plot(subj_results['Time'], subj_results['F1'], 'g-', lw=2, label='F1 (Macro)')
        ax.axhline(0.5, color='gray', ls='--', label='Chance (0.5)')
        
        try:
            ts = analyzer.mean_event_times
            def add_epoch(lbl, s, e, c, a=0.1):
                if s >= t_max_analysis: return
                eff_e = min(e, t_max_analysis)
                m = (s + eff_e) / 2
                ax.axvspan(s, eff_e, color=c, alpha=a, lw=0)
                if s < eff_e: ax.text(m, ax.get_ylim()[1]*0.95, lbl, ha='center', fontsize=9, color='k')
                ax.axvline(s, color='k', ls=':', lw=1)

            if 2 in ts:
                add_epoch("Sample", 0, ts[2], 'yellow')
                if 3 in ts:
                    add_epoch("Delay1", ts[2], ts[3], 'green', 0.05)
                    if 4 in ts:
                        add_epoch("Delay2", ts[3], ts[4], 'blue', 0.05)
                        ax.axvline(ts[4], color='k', ls='-', lw=1.5, label='Response Cue')
            ax.axvline(0, color='k', ls='-', lw=1.5)
        except: pass
        
        ax.set_ylabel('Score')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Subject {subj_id} - VF Classification (Left vs Right)')
        ax.set_ylim(0, 1.0)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, t_max_analysis)
        
        out_path = os.path.join(out_dir, f'E03_Subj{subj_id:02d}_ClassificationVF.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"  Error subj {subj_id}: {e}", flush=True)
        import traceback
        traceback.print_exc()

if group_results:
    print("\n=== Generating Group Plot ===", flush=True)
    all_res = pd.concat(group_results, ignore_index=True)
    
    agg_acc = all_res.groupby('Time')['Accuracy'].agg(['mean', 'sem']).reset_index()
    agg_f1 = all_res.groupby('Time')['F1'].agg(['mean', 'sem']).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(agg_acc['Time'], agg_acc['mean'], 'b-', lw=2, label='Mean Accuracy')
    ax.fill_between(agg_acc['Time'], 
                    agg_acc['mean'] - agg_acc['sem'], 
                    agg_acc['mean'] + agg_acc['sem'], color='b', alpha=0.15)
                    
    ax.plot(agg_f1['Time'], agg_f1['mean'], 'g-', lw=2, label='Mean F1')
    ax.fill_between(agg_f1['Time'], 
                    agg_f1['mean'] - agg_f1['sem'], 
                    agg_f1['mean'] + agg_f1['sem'], color='g', alpha=0.15)
    
    ax.axhline(0.5, color='k', ls='--', label='Chance (0.5)')
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Time (s)')
    ax.set_title('Group Level Gaze Classification (Visual Field: Left vs Right)')
    ax.set_ylim(0.4, 0.9)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', ls='-')
    ax.set_xlim(-0.5, 4.3)
    
    out_path = os.path.join(out_dir, 'E03_Group_ClassificationVF.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}", flush=True)

print("\n=== Done ===", flush=True)
