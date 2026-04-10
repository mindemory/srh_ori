"""
GazeAnalyzer: Class-based gaze data analysis for TMS orientation study.

Supports flexible filtering by TMS condition, hemifield, orientation, etc.
"""

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Dict, Any


class GazeAnalyzer:
    """Load and analyze gaze data for one or more subjects."""
    
    # Event codes from XDAT
    EVENT_CODES = {
        1: ('gray', 'Fixation', 'F'),
        2: ('purple', 'Delay1', 'D1'),
        3: ('cyan', 'Delay2', 'D2'),
        4: ('magenta', 'Response', 'R'),
        5: ('brown', 'Feedback', 'FB'),
        6: ('pink', 'ITI', 'ITI')
    }
    
    # Orientation color palette
    ORI_PALETTE = {30: '#1f77b4', 60: '#2ca02c', 120: '#ff7f0e', 150: '#d62728'}
    
    def __init__(self, datd_path: str, master_df: pd.DataFrame):
        """
        Initialize GazeAnalyzer.
        
        Args:
            datd_path: Root path to data directory (e.g., /d/DATD/datd/TMS_ori)
            master_df: Behavioral dataframe from generate_behavdf
        """
        self.datd_path = datd_path
        self.master_df = master_df
        self._gaze_cache: Dict[int, Dict] = {}  # Cache gaze data by subject
        
    def load_gaze_data(self, subj_id: int) -> Dict[str, List[np.ndarray]]:
        """
        Load all gaze data (both days) for a subject.
        
        Args:
            subj_id: Subject ID (1-based)
            
        Returns:
            Dictionary with keys 'X', 'Y', 'Pupil', 'XDAT', each a list of traces
        """
        if subj_id in self._gaze_cache:
            return self._gaze_cache[subj_id]
        
        subj_str = f'sub{subj_id:02d}'
        days = ['day01', 'day02']
        
        all_X, all_Y, all_Pupil, all_XDAT = [], [], [], []
        
        for day in days:
            mat_path = os.path.join(
                self.datd_path, 'analysis', subj_str, day, 
                f'ii_sess_{subj_str}_{day}.mat'
            )
            
            if not os.path.exists(mat_path):
                print(f"Warning: {mat_path} not found, skipping.")
                continue
                
            mat = sio.loadmat(mat_path)
            
            if 'ii_sess' not in mat:
                print(f"Warning: 'ii_sess' not found in {mat_path}, skipping.")
                continue
                
            struct = mat['ii_sess'][0, 0]
            X = struct['X']
            Y = struct['Y']
            Pupil = struct['Pupil']
            XDAT = struct['XDAT']
            
            n_trials = X.shape[0]
            print(f"Loaded {subj_str}/{day}: {n_trials} trials")
            
            for i in range(n_trials):
                all_X.append(X[i, 0].flatten())
                all_Y.append(Y[i, 0].flatten())
                all_Pupil.append(Pupil[i, 0].flatten())
                all_XDAT.append(XDAT[i, 0].flatten())
        
        gaze_data = {
            'X': all_X,
            'Y': all_Y,
            'Pupil': all_Pupil,
            'XDAT': all_XDAT
        }
        
        self._gaze_cache[subj_id] = gaze_data
        return gaze_data
    
    def get_behavioral_df(self, subj_id: int) -> pd.DataFrame:
        """Get behavioral dataframe subset for a subject."""
        return self.master_df[self.master_df['subjID'] == subj_id].copy()
    
    def filter_trials(self, df: pd.DataFrame, **filters) -> pd.DataFrame:
        """
        Filter trials by specified conditions.
        
        Supported filters:
            TMScond: 1=NoTMS, 2=Early, 3=Middle
            targHemifield: 'Left' or 'Right'
            targOriCategory: 30, 60, 120, 150
            
        Example:
            filter_trials(df, TMScond=1, targHemifield='Left')
        """
        filtered = df.copy()
        
        for key, value in filters.items():
            if key in filtered.columns:
                filtered = filtered[filtered[key] == value]
            else:
                print(f"Warning: Column '{key}' not found in dataframe")
                
        return filtered
    def _lowpass_filter(self, data: np.ndarray, cutoff: float = 30.0, fs: float = 1000.0, order: int = 2) -> np.ndarray:
        """
        Apply Butterworth low-pass filter to data.
        
        Args:
            data: 1D array of signal data
            cutoff: Cutoff frequency in Hz (default 30 Hz)
            fs: Sampling frequency in Hz (default 1000 Hz for 1kHz eye tracker)
            order: Filter order (default 2)
            
        Returns:
            Filtered data array
        """
        # Nyquist frequency
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        
        # Ensure cutoff is valid (must be < 1.0 after normalization)
        if normal_cutoff >= 1.0:
            return data
            
        # Design Butterworth filter
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply filter (use filtfilt for zero-phase filtering)
        try:
            filtered = filtfilt(b, a, data)
            return filtered
        except Exception:
            # If filtering fails (e.g., data too short), return original
            return data
    
    def _detect_blinks(
        self, 
        pupil: np.ndarray, 
        threshold: float = 0.0,
        min_duration: int = 50,
        padding: int = 50
    ) -> np.ndarray:
        """
        Detect blinks based on pupil diameter.
        
        Blinks are detected when pupil diameter is 0 or below threshold.
        We also add padding samples before and after each blink to capture
        the full blink artifact.
        
        Args:
            pupil: Pupil diameter trace
            threshold: Threshold below which samples are considered blinks (default 0)
            min_duration: Minimum blink duration in samples to be considered valid
            padding: Number of samples to add before/after detected blinks
            
        Returns:
            Boolean mask where True = blink sample
        """
        # Detect samples where pupil is 0 or NaN or below threshold
        is_blink = (pupil <= threshold) | np.isnan(pupil)
        
        # Find contiguous blink regions
        blink_mask = np.zeros_like(is_blink)
        
        # Find start and end of each blink
        diff = np.diff(is_blink.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if is_blink[0]:
            starts = np.concatenate([[0], starts])
        if is_blink[-1]:
            ends = np.concatenate([ends, [len(is_blink)]])
        
        # Apply padding and mark blinks
        for start, end in zip(starts, ends):
            duration = end - start
            if duration >= min_duration or duration > 0:  # Include all blinks
                padded_start = max(0, start - padding)
                padded_end = min(len(blink_mask), end + padding)
                blink_mask[padded_start:padded_end] = True
        
        return blink_mask
    
    def _interpolate_blinks(
        self, 
        data: np.ndarray, 
        blink_mask: np.ndarray,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Interpolate over blink periods.
        
        Args:
            data: Data array (X, Y, or Pupil)
            blink_mask: Boolean mask where True = blink sample
            method: Interpolation method ('linear' or 'cubic')
            
        Returns:
            Data array with blinks interpolated
        """
        data_interp = data.copy()
        
        if not np.any(blink_mask):
            return data_interp
        
        # Get valid (non-blink) indices
        valid_idx = np.where(~blink_mask)[0]
        blink_idx = np.where(blink_mask)[0]
        
        if len(valid_idx) < 2:
            # Not enough valid data to interpolate
            return data_interp
        
        # Interpolate
        try:
            data_interp[blink_idx] = np.interp(blink_idx, valid_idx, data[valid_idx])
        except Exception:
            pass  # If interpolation fails, return original
        
        return data_interp
    
    def extract_traces(
        self, 
        df: pd.DataFrame, 
        gaze_data: Dict[str, List[np.ndarray]],
        gaze_threshold: float = 1.5,
        downsample: int = 10,
        baseline_window: tuple = (-1.0, 0.0),
        lowpass_cutoff: float = 30.0
    ) -> pd.DataFrame:
        """
        Extract time-aligned traces for filtered trials.
        
        Args:
            df: Filtered behavioral dataframe (with original index preserved)
            gaze_data: Dictionary of gaze traces from load_gaze_data()
            gaze_threshold: Exclude samples outside ±threshold degrees
            downsample: Downsample factor (10 = sample every 10ms)
            baseline_window: Time window for pupil baseline correction (seconds)
            lowpass_cutoff: Cutoff frequency for Butterworth low-pass filter (Hz). 
                           Set to None or 0 to disable filtering.
            
        Returns:
            DataFrame with columns: Time, X, Y, Pupil_Z, OriCat, Trial, TMScond, Hemifield
        """
        # Compute session-level pupil stats for z-scoring
        all_pupil = np.concatenate(gaze_data['Pupil'])
        p_mean = np.nanmean(all_pupil)
        p_std = np.nanstd(all_pupil)
        
        plot_data = []
        event_times = {code: [] for code in self.EVENT_CODES.keys()}
        
        for idx, (df_idx, row) in enumerate(df.iterrows()):
            # The behavioral dataframe for a subject is ordered by trial.
            # We need to find the position of this trial within the subject's behavioral data.
            # Since df is filtered from behav_df (which is subj-specific), we can use 
            # the position relative to the subject's full behavioral dataframe.
            
            # Get subject's full behavioral df and find position
            subj_df = self.master_df[self.master_df['subjID'] == row['subjID']]
            subj_df_reset = subj_df.reset_index()
            
            # Find the position of this trial in the subject's data
            position_in_subj = subj_df_reset[subj_df_reset['index'] == df_idx].index
            if len(position_in_subj) == 0:
                print(f"Warning: Could not find trial {df_idx} in subject data, skipping.")
                continue
            gaze_trial = position_in_subj[0]
            
            if gaze_trial >= len(gaze_data['X']):
                print(f"Warning: Trial {gaze_trial} exceeds gaze data length, skipping.")
                continue
            
            x_trace = gaze_data['X'][gaze_trial].copy()
            y_trace = gaze_data['Y'][gaze_trial].copy()
            pup_trace = gaze_data['Pupil'][gaze_trial].copy()
            xdat_trace = gaze_data['XDAT'][gaze_trial]
            
            # Detect and interpolate blinks (based on pupil = 0 or NaNs)
            # 1. Detect blinks from pupil
            blink_mask = self._detect_blinks(pup_trace, threshold=0, padding=50)
            
            # 2. Also treat existing NaNs in X/Y as blinks/missing data
            nan_mask = np.isnan(x_trace) | np.isnan(y_trace) | np.isnan(pup_trace)
            
            # Combine masks
            combined_mask = blink_mask | nan_mask
            
            # Interpolate over all missing/blink periods
            x_trace = self._interpolate_blinks(x_trace, combined_mask)
            y_trace = self._interpolate_blinks(y_trace, combined_mask)
            pup_trace = self._interpolate_blinks(pup_trace, combined_mask)
            
            # Apply low-pass filter if enabled (now safe from NaNs)
            if lowpass_cutoff and lowpass_cutoff > 0:
                # One final check: if there are still NaNs (e.g. at edges if interpolate failed), fill them
                if np.isnan(x_trace).any():
                    # Forward fill then backward fill
                    x_trace = pd.Series(x_trace).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
                    y_trace = pd.Series(y_trace).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
                    pup_trace = pd.Series(pup_trace).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
                
                try:
                    x_trace = self._lowpass_filter(x_trace, cutoff=lowpass_cutoff)
                    y_trace = self._lowpass_filter(y_trace, cutoff=lowpass_cutoff)
                    pup_trace = self._lowpass_filter(pup_trace, cutoff=lowpass_cutoff)
                except Exception as e:
                    print(f"Warning: properties filtering failed for trial {gaze_trial}: {e}")
            
            # Find Sample Onset (10 or 11)
            onset_indices = np.where((xdat_trace == 10) | (xdat_trace == 11))[0]
            if len(onset_indices) == 0:
                continue
            t0_idx = onset_indices[0]
            
            # Collect event times
            for code in event_times.keys():
                code_indices = np.where(xdat_trace == code)[0]
                if len(code_indices) > 0:
                    event_times[code].append((code_indices[0] - t0_idx) / 1000.0)
            
            # Z-score pupil
            pup_z = (pup_trace - p_mean) / p_std
            
            # Time axis
            time_axis = (np.arange(len(x_trace)) - t0_idx) / 1000.0
            
            # Baseline correction
            baseline_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
            if np.sum(baseline_mask) > 0:
                baseline_mean = np.nanmean(pup_z[baseline_mask])
                pup_bc = pup_z - baseline_mean
            else:
                pup_bc = pup_z
            
            # Extract samples
            for t_idx, t in enumerate(time_axis):
                if t_idx % downsample != 0:
                    continue
                
                x_val = x_trace[t_idx]
                y_val = y_trace[t_idx]
                
                # Filter by gaze threshold
                if abs(x_val) > gaze_threshold or abs(y_val) > gaze_threshold:
                    continue
                
                plot_data.append({
                    'Time': t,
                    'X': x_val,
                    'Y': y_val,
                    'Pupil_Z': pup_bc[t_idx],
                    'OriCat': int(row['targOriCategory']),
                    'Trial': gaze_trial,
                    'TMScond': row['TMScond'],
                    'Hemifield': row.get('targHemifield', 'Unknown')
                })
        
        # Compute mean event times
        self.mean_event_times = {
            code: np.mean(times) for code, times in event_times.items() if len(times) > 0
        }
        
        return pd.DataFrame(plot_data)
    
    def compute_aggregates(
        self, 
        df: pd.DataFrame, 
        group_by: List[str] = ['Time', 'OriCat'],
        columns: List[str] = ['X', 'Y', 'Pupil_Z']
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute mean and SEM for traces grouped by specified columns.
        
        Args:
            df: DataFrame from extract_traces()
            group_by: Columns to group by (e.g., ['Time', 'OriCat'])
            columns: Data columns to aggregate
            
        Returns:
            Dictionary mapping column names to aggregated DataFrames
        """
        aggregates = {}
        
        for col in columns:
            agg = df.groupby(group_by)[col].agg(['mean', 'sem']).reset_index()
            aggregates[col] = agg
            
        return aggregates
    
    def plot_traces(
        self,
        aggregates: Dict[str, pd.DataFrame],
        title: str = "Gaze Traces",
        save_path: Optional[str] = None,
        group_col: str = 'OriCat',
        time_range: tuple = (-1.0, 4.5),
        ylims: Dict[str, tuple] = None,
        figsize: tuple = (12, 15)
    ):
        """
        Generate X, Y, Pupil subplots.
        
        Args:
            aggregates: Dictionary from compute_aggregates()
            title: Plot title
            save_path: Path to save figure
            group_col: Column to use for color grouping
            time_range: Time range for x-axis
            ylims: Dictionary of y-limits per column
            figsize: Figure size
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        if ylims is None:
            ylims = {'X': (-0.5, 0.5), 'Y': (-0.5, 0.5), 'Pupil_Z': None}
        
        columns = ['X', 'Y', 'Pupil_Z']
        labels = ['X (deg)', 'Y (deg)', 'Pupil (Z)']
        
        for ax, col, label in zip(axes, columns, labels):
            agg = aggregates[col]
            
            # Get unique group values
            groups = agg[group_col].unique()
            
            for group in sorted(groups):
                color = self.ORI_PALETTE.get(group, 'gray')
                group_data = agg[agg[group_col] == group].sort_values('Time')
                
                t = group_data['Time'].values
                m = group_data['mean'].values
                sem = group_data['sem'].values
                
                ax.plot(t, m, color=color, linewidth=1.5, label=f'{group}°')
                ax.fill_between(t, m - sem, m + sem, color=color, alpha=0.15)
            
            # Set y-limits
            if ylims.get(col) is not None:
                ax.set_ylim(ylims[col])
            
            ax.set_ylabel(label)
            ax.set_title(f"Mean {col} Trace - {title}")
            
            # Add event lines
            self._add_event_lines(ax, ylims.get(col))
        
        # Add legend to first axis
        axes[0].legend(loc='upper right', fontsize='small', title=group_col)
        
        # Set x-axis
        x_min = self.mean_event_times.get(1, -1.5) - 0.3
        x_max = self.mean_event_times.get(4, 5.0) + 0.05
        for ax in axes:
            ax.set_xlim(x_min, x_max)
        
        axes[2].set_xlabel("Time from Sample Onset (s)")
        
        plt.tight_layout()
        
        if save_path:
            print(f"Saving to {save_path}")
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _add_event_lines(self, ax, ylim):
        """Add event marker lines to axis."""
        y_top = ylim[1] if ylim else ax.get_ylim()[1]
        
        # Sample onset line
        ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
        ax.text(0.02, y_top, 'S', fontsize=8, color='k', verticalalignment='top')
        
        for code, (color, label, short) in self.EVENT_CODES.items():
            if code in self.mean_event_times:
                t = self.mean_event_times[code]
                ax.axvline(t, color=color, linestyle=':', linewidth=1.2)
                ax.text(t + 0.02, y_top, short, fontsize=7, color=color, verticalalignment='top')
