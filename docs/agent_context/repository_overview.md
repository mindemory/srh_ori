# Repository Overview: srh_ori

## Summary
This repository contains MATLAB scripts for running and analyzing two primary neuroscientific experiments: **Ori** (Orientation) and **Phosphene** (Phosphene Mapping). The codebase is structured into "Task" directories for experimental control (likely using Psychtoolbox) and "Analysis" directories for data processing (Behavioral, Eye-tracking, and EEG).

## Directory Structure

### 1. `oriTask`
Contains scripts to run the Orientation experiment.
*   **Key File**: `ori_task.m` (Likely the main entry point)
*   **Helper Scripts**:
    *   `initScreen.m`, `initEyeTracker.m`, `initFiles.m`: Initialization routines.
    *   `loadParameters.m`: Parameter loading.
    *   `PsychPowerMateMrugank.m`, `testPowerMate.m`: Input device handling (PowerMate).
    *   `drawTextures.m`, `showprompts.m`: Stimulus display functions.

### 2. `oriAnalysis`
Contains scripts for analyzing data from the Orientation experiment. The analysis seems to follow a pipeline structure (A01-A04).
*   **Pipeline Scripts**:
    *   `A01_GetTimeReport.m`: Initial timing report.
    *   `A02_EyeAnalysis.m`: Eye-tracking data analysis.
    *   `A03_BehavAnalysis.m`: Behavioral performance analysis.
    *   `A04_FinalAnalysis.m`: Final aggregated analysis.
*   **Subdirectories**:
    *   `analyzeEEG`: EEG specific analysis.
    *   `analyzeBehav`: Behavioral specific analysis.
    *   `analysisEye`: Eye-tracking specific analysis.
    *   `tms_efield`: Likely related to TMS E-field modeling.

### 3. `phospheneTask`
Contains scripts to run the Phosphene Recording experiment.
*   **Key File**: `recordPhosphenes.m` (Main entry point)
*   **Helper Scripts**:
    *   `initScreen.m`, `initPeripherals.m`: Hardware setup.
    *   `drawTextures.m`: Graphic rendering.

### 4. `phospheneAnalysis`
Contains scripts for analyzing the mapped phosphenes.
*   **Key Files**:
    *   `S01_phosphene_analysis.m`, `S02_saveTaskMap.m`: Sequential analysis steps.
    *   `calcPhospheneArea.m`: Calculating properties of phosphenes.
    *   `plotGrating.m`: Visualization.

## Dependencies
*   **MATLAB**: The primary language.
*   **Psychtoolbox**: Strongly implied by file names like `initScreen`, `drawTextures`, `initEyeTracker` (standard naming in PTB).
*   **Eyelink**: Specific support for Eyelink eye trackers (`initEyeTracker.m`, `A02_EyeAnalysis.m`).
