# Statistical Analysis Summary

The "complex" model successfully identified specific conditions where TMS and experimental factors influence behavior. Here is the plain-English breakdown:

## Key Findings

### 1. Does TMS change perception? (Bias / Signed Error)
**YES**.
- **The "In-Field" Effect**: Normally, stimuli presented in the Phosphene Field (InPF) had a different bias than OutPF.
- **TMS Disruption**: **Early TMS** specifically interacting with the **Phosphene Field** caused a significant shift in bias (+2.01°, *p*=.03). This is the "smoking gun" that TMS to that retinotopic locaiton had an effect.
- **Orientation Bias**: As expected, there are huge biases depending on the angle (e.g., 60° and 150° have strong negative biases vs 30°). This is the standard "Oblique Effect" or cardinal repulsion.

### 2. Does TMS make them worse? (Accuracy / Absolute Error)
**sort of, but it's complicated.**
- **General Improvement?**: Surprisingly, **Early TMS** generally *reduced* absolute error (improved precision) by ~1 degree (*p*=.02).
- **Orientation Dependence**: However, at **120°**, Early TMS made things *worse* (+1.39°, *p*=.03).
- **Takeaway**: TMS doesn't just "add noise"; it seems to sharpen some representations while distorting others.

### 3. Do they repeat their mistakes? (Serial Dependence)
**YES, and TMS changes it.**
- **History Effect**: The error on the previous trial strongly predicts the error on the current trial (*p*<.001).
- **TMS Modulation**: **Middle TMS** significantly changed this relationship (*p*=.009). This suggests TMS might be disrupting the "memory trace" of the previous trial.

### 4. Is it just speed? (RT)
**NO.**
- There were no significant effects of TMS or Hemifield on Reaction Time. This is **good news**: it means the changes in error are genuine perceptual changes, not just subjects rushing or slowing down.

---

## Recommended Next Steps

Since we have confirmed *where* the significant effects are, we should now create **targeted visualizations** to show them clearly (instead of the messy "plot everything" approach).

1.  **Plot the "In vs Out" TMS Effect**:
    - Focus on **Bias (Signed Error)**.
    - Bar plot: `Error` on Y-axis, `TMS Condition` on X-axis, hue = `Hemifield` (In vs Out).
    - *Expectation*: A distinct difference for Early TMS in the InPF bar.

2.  **Plot the Serial Dependence**:
    - Scatter/Line plot: `Current Error` (Y) vs `Previous Error` (X).
    - Hue = `TMS Condition`.
    - *Expectation*: Different slopes for Middle TMS vs No TMS.

3.  **Plot Orientation Bias**:
    - Line plot: `Error` (Y) vs `Target Orientation` (X).
    - *Expectation*: A squiggly line showing the systematic bias (cardinal repulsion).

Shall I generate these specific plots?
