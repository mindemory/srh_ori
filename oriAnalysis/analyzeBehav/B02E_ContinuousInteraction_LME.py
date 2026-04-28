import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings

# Use absolute path handling just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import generate_behavdf as gm

# Ignore pandas categorical warnings if any
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

output_file = "B02E_LME_Interaction_results.txt"

with open(output_file, "w") as f:
    f.write("================================================================================\n")
    f.write("LINEAR MIXED-EFFECTS (LME) ANALYSIS OF BEHAVIORAL METRICS\n")
    f.write("B02E: Testing Continuous Interference (Bias ~ TMS * DiagType * initOffset)\n")
    f.write("================================================================================\n\n")

df = gm.master_df.copy()
print(f"Loaded master_df with {len(df)} trials.")

# --- 1. FILTERING ---
# Exclude LargeError (outliers > 3SD) just like in B02B
clean_df = df[df['LargeError'] == False].copy()
print(f"Removed {len(df) - len(clean_df)} LargeError trials.")
df = clean_df.dropna(subset=['targOriActual', 'initOffset']).copy()

# --- 2. CATEGORICAL VARIABLES ---
# DiagType standard classification
df['OriCat'] = df['targOriCategory'].astype(int)
df['OriCat45'] = df['OriCat'].replace({
    30: 'Acute (30+60)', 60: 'Acute (30+60)',
    120: 'Obtuse (120+150)', 150: 'Obtuse (120+150)'
})

def classify_diag(row):
    hemi_stim = row['hemisphereStimulated']
    in_pf = row['stimPF_label'] == 'inPF'
    is_acute = row['OriCat45'] == 'Acute (30+60)'
    
    if hemi_stim == 'Left':
        return 'nearDiag' if (in_pf and not is_acute) or (not in_pf and is_acute) else 'offDiag'
    else: # Right TMS
        return 'nearDiag' if (in_pf and is_acute) or (not in_pf and not is_acute) else 'offDiag'

df['DiagType'] = df.apply(classify_diag, axis=1)

# Diagonal Bias
df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1

# --- 3. CONTINUOUS COVARIATES ---
df['initOffset_sq'] = df['initOffset'] ** 2

df['TMS'] = df['TMScond_label']
df['TMS_Group'] = df['TMS'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
def make_tms_cond(row):
    if row['TMS_Group'] in ['No TMS', 'NoTMS']:
        return 'NoTMS'
    return f"TMS_{row['stimPF_label']}"
df['TMS_Condition'] = df.apply(make_tms_cond, axis=1)

# Set up Categorical levels
df['DiagType'] = pd.Categorical(df['DiagType'], categories=['offDiag', 'nearDiag'], ordered=True)
df['TMS_Condition'] = pd.Categorical(df['TMS_Condition'], categories=['NoTMS', 'TMS_inPF', 'TMS_outPF'], ordered=True)

metrics = ['abserrorOri', 'RT', 'BiasTowardsDiagonal']

for met in metrics:
    print(f"Fitting MixedLM for {met}...")
    model_df = df.dropna(subset=[met, 'initOffset', 'initOffset_sq', 'TMS_Condition', 'DiagType']).copy()
    if met == 'BiasTowardsDiagonal':
        formula = f"{met} ~ C(TMS_Condition, Treatment('NoTMS')) * C(DiagType, Treatment('nearDiag')) * initOffset + initOffset_sq"
    else:
        formula = f"{met} ~ C(TMS_Condition, Treatment('NoTMS')) * C(DiagType, Treatment('nearDiag')) + initOffset + initOffset_sq"
    
    md = smf.mixedlm(formula, model_df, groups=model_df["subjID"])
    
    try:
        mdf = md.fit(method='lbfgs')
    except Exception as e:
        print(f"  LBFGS failed, falling back to default powell for {met}")
        mdf = md.fit()
        
    with open(output_file, "a") as f:
        f.write("="*80 + "\n")
        f.write(f"LME RESULTS FOR: {met}\n")
        f.write("="*80 + "\n")
        f.write(str(mdf.summary()))
        f.write("\n\n")
        
        # Marginal Wald tests for omnibus categorical effects
        try:
            # Tell statsmodels we want scalar return to avoid FutureWarnings and 2D arrays
            wald = mdf.wald_test_terms(scalar=True)
            wald_df = wald.summary_frame()
            for col in wald_df.columns:
                wald_df[col] = wald_df[col].apply(lambda x: x.item() if isinstance(x, np.ndarray) and x.size == 1 else x)
            f.write("--- Omnibus Wald Tests (ANOVA Style) ---\n")
            f.write(wald_df.to_string(float_format=lambda x: f"{x:.4f}"))
        except Exception as e:
            f.write(f"--- Omnibus Wald Tests Failed: {e} ---\n")
            
        f.write("\n\n\n")

print(f"Finished! Results written to {output_file}")
