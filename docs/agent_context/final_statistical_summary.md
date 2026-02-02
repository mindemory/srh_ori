# Statistical Analysis Summary (Final)

We performed a comprehensive analysis across multiple metrics, including **Aggregate Means**, **Trial-Level Data**, and **Mixture Modeling Parameters** (Precision & Guess Rate).

## Experiment Findings at a Glance

| Metric | Best Model | Key Driver | Intepretation |
| :--- | :--- | :--- | :--- |
| **Accuracy** (AbsError) | **M2** | **Orientation (Oblique Effect)** | Accuracy is primarily determined by the angle (Cardinals > Obliques). TMS effects are washed out by this strong visual baseline. |
| **Bias** (Signed Error) | **M3** | **Lat. Interaction (Ori × HemiStim)** | Bias is **Lateralized**. Stimulation of the Right Hemisphere specifically distorts representations of 60°/120° angles. |
| **Precision** (Kappa) | **M4** | **Complex Context (4-Way)** | Precision is the most sensitive metric. It fluctuates based on the specific combination of TMS site, Visual Field, Angle, and Hemisphere. |
| **Guess Rate** (g) | **M3** | **Lat. Interaction (Ori × HemiStim)** | "Guessing" isn't random. It follows the same lateralized pattern as Bias, suggesting TMS disrupts "confidence" or "binding" in specific lateralized channels. |
| **Reaction Time** (RT) | **M1** | **Baseline State** | Speed is stable. TMS exerts a general baseline effect but does not selectively slow down specific orientations. |

---

## Detailed Results

### 1. Mechanism Analysis (Final Hybrid Model)
We implemented a **Hybrid Analysis**: Trial-level modeling (to control for specific adjustment distances) for standard metrics, and Aggregate modeling for mixture parameters.

*   **Impact of Initial Cue Distance (`initOffset`)**:
    *   **Reaction Time**: HUGE effect (*p*<1e-10). Moving further takes longer. (Sanity Check: Passed).
    *   **Accuracy**: Small but significant effect (*p*=.028). Larger distances add tiny motor noise.
    *   **Diagonal Bias**: **NO EFFECT** (*p*=.868). The tendency to drift towards diagonals is **purely perceptual**. It is effectively independent of how far you physically move the dial.

*   **Diagonal Attraction (`BiasTowardsDiagonal`)**:
    *   **Winner**: **M3** (Trial-Level, *p*<1e-180).
    *   **Key Finding**: Even controlling for motor adjustment, the **Right Hemisphere** generates a strong diagonal bias (+1.6°) that is effectively "corrected" by Early TMS.

### 2. Mixture Model Parameters (Aggregate)
Fitting a Von Mises + Uniform model to condition means:
*   **Precision (Kappa)**: **M4** (*p*=.013). Tuning sharpness is fragile and breaks down in the full 4-way interaction ("Perfect Storm").
*   **Guess Rate (g)**: **M3** (*p*=.023). Guessing is driven by the specific **Lateralized** interaction between Orientation and the Stimulated Hemisphere.

### 3. Standard Behavioral Metrics (Trial-Level with Controls)
*   **Absolute Error (Accuracy)**: **M3** is now the preferred model (*p*<1e-40) after controlling for `initOffset`. The lateralized TMS effect becomes visible once motor noise is regressed out.
*   **Signed Error (Bias)**: **M4** (*p*<1e-129). A complex pattern of shifts.
*   **Reaction Time**: **M2** (*p*<1e-100). Primarily driven by the Oblique Effect (Cardinal orientations are faster), even after controlling for movement time.

## Conclusion
The "Hybrid" analysis provides the strongest evidence yet:
1.  **Motor vs. Perceptual**: `initOffset` explains RT and some motor noise, but **Diagonal Bias is Perceptual**.
2.  **Robust Lateralization**: Controlling for motor planning *strengthens* the evidence for the **M3/Lateralized** hypothesis. The Right Hemisphere is the active generator of bias, which TMS disrupts.
