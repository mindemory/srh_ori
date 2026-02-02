
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import generate_behavdf as gm
import modeller as md
import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max.columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.precision", 2)

print("Loading data...")
df = gm.master_df.copy()

# --- Data Preparation ---
print("Preparing data features...")

# Factor conversions
df.rename(columns={
    'TMScond_label': 'TMS',        
    'stimPF_label': 'Hemifield',   
    'targOriCategory': 'OriCat',
    'hemisphereStimulated': 'HemiStim'
}, inplace=True)

# 1. InitOffset (Existing: signed difference)
# We keep it as is. It's the "Independent Feature" requested.

# 2. Diagonal Attraction DV
df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1

# RT Cleanup
if 'respDuration' in df.columns:        
     df.rename(columns={'respDuration': 'RT'}, inplace=True)
if isinstance(df['RT'], pd.DataFrame):
     rt_series = df['RT'].iloc[:, 0].copy()
     df = df.drop(columns=['RT'])
     df['RT'] = rt_series
df['RT'] = pd.to_numeric(df['RT'], errors='coerce')
df['absError'] = np.abs(df['errOri_scaled'])

# Type Conversion
df['TMS'] = df['TMS'].astype('category')
df['Hemifield'] = df['Hemifield'].astype('category')
df['OriCat'] = df['OriCat'].astype('category')
df['HemiStim'] = df['HemiStim'].astype('category')
df['subjID'] = df['subjID'].astype('category')

# Reorder
df['TMS'] = df['TMS'].cat.reorder_categories(['No TMS', 'Early TMS', 'Middle TMS'], ordered=True)
df['Hemifield'] = df['Hemifield'].cat.reorder_categories(['outPF', 'inPF'], ordered=True)


print("\n" + "="*80)
print("SECTION 1: TRIAL-LEVEL ANALYSIS (Standard DVs + InitOffset)")
print("Including 'initOffset' as Independent Variable")
print("="*80 + "\n")

dvs_trial = {
    "Accuracy (AbsError)": "absError",
    "Bias (Signed)": "errOri_scaled",
    "Reaction Time": "RT",
    "Diagonal Attraction": "BiasTowardsDiagonal"
}

for dv_label, dv_name in dvs_trial.items():
    print(f"\n\n{'='*100}")
    print(f"ANALYSIS FOR: {dv_label} ({dv_name})")
    print(f"{'='*100}\n")
    
    # Clean Data
    df_clean = df.dropna(subset=[dv_name, 'initOffset', 'TMS', 'Hemifield', 'OriCat', 'HemiStim'])
    
    # Models: Trial-Level with Subject Fixed Effects + InitOffset
    # Note: Adding initOffset as a main covariate.
    base_fe = "C(subjID) +"
    models = {
         # Simple Main Effects
         "M1": f"{dv_name} ~ {base_fe} TMS * Hemifield + initOffset",
         # Add Ori (Oblique Effect)
         "M2": f"{dv_name} ~ {base_fe} TMS * Hemifield + OriCat + initOffset",
         # Add Lateralized Interaction (Hypothesis)
         "M3": f"{dv_name} ~ {base_fe} TMS * Hemifield + OriCat * HemiStim + initOffset",
         # Full 4-Way
         "M4": f"{dv_name} ~ {base_fe} TMS * Hemifield * OriCat * HemiStim + initOffset"
    }

    results_list = []

    print(f"{'Model':<10} | {'AIC':<10} | {'BIC':<10} | {'R-squared':<10} | {'Adj. R2':<10} | {'Formula'}")
    print("-" * 120)

    try:
        for name, formula in models.items():
            model = smf.ols(formula, data=df_clean)
            res = model.fit()
            
            results_list.append({
                "Model": name,
                "Formula": formula,
                "AIC": res.aic,
                "BIC": res.bic,
                "R2": res.rsquared,
                "AdjR2": res.rsquared_adj,
                "Object": res
            })
            
            print(f"{name:<10} | {res.aic:<10.1f} | {res.bic:<10.1f} | {res.rsquared:<10.3f} | {res.rsquared_adj:<10.3f} | {formula}")

        if results_list:
            sorted_results = sorted(results_list, key=lambda x: x['AIC'])
            best_model = sorted_results[0]
            print(f"\n--- Best Model for {dv_label} ---")
            print(f"Best Model: {best_model['Model']}")
            print(f"Formula: {best_model['Formula']}")
            # Print Coefs for initOffset specifically
            print(best_model['Object'].summary())
            
    except Exception as e:
        print(f"Error analyzing {dv_name}: {e}")


print("\n\n" + "="*80)
print("SECTION 2: AGGREGATE ANALYSIS (Mixture Models)")
print("Note: initOffset cannot be used here (averages to ~0)")
print("="*80 + "\n")

# Grouping for Mixture
group_cols = ['subjID', 'TMS', 'Hemifield', 'OriCat', 'HemiStim']
fit_results = []
grouped = df.groupby(group_cols, observed=True)

for name, group in grouped:
    if len(group) < 10: continue
    data_deg = group['errOri_scaled'].dropna().values
    try:
        g, K = md.fit_model(data_deg, modelName='mixtureModel')
    except:
        g, K = np.nan, np.nan
    
    fit_results.append(list(name) + [g, K])

cols = group_cols + ['GuessRate', 'Precision']
df_agg = pd.DataFrame(fit_results, columns=cols)
df_agg = df_agg.replace([np.inf, -np.inf], np.nan).dropna()

dvs_agg = {
    "Precision (Kappa)": "Precision",
    "Guess Rate": "GuessRate"
}

for dv_label, dv_name in dvs_agg.items():
    print(f"\n\n{'='*100}")
    print(f"MIXTURE ANALYSIS FOR: {dv_label} ({dv_name})")
    print(f"{'='*100}\n")
    
    # Models (No initOffset, No C(subjID) because it's aggregate)
    # Correct. In aggregate analysis of Subject Means, we treat Subj as random/noise and just look at fixed effects of conditions.
    models = {
         "M1": f"{dv_name} ~ TMS * Hemifield",
         "M2": f"{dv_name} ~ TMS * Hemifield + OriCat",
         "M3": f"{dv_name} ~ TMS * Hemifield + OriCat * HemiStim",
         "M4": f"{dv_name} ~ TMS * Hemifield * OriCat * HemiStim"
    }
    
    results_list = []
    print(f"{'Model':<10} | {'AIC':<10} | {'BIC':<10} | {'R-squared':<10} | {'Adj. R2':<10}")
    print("-" * 100)
    
    try:
        for name, formula in models.items():
            model = smf.ols(formula, data=df_agg)
            res = model.fit()
            results_list.append({"Model":name, "AIC":res.aic, "Obj":res})
            print(f"{name:<10} | {res.aic:<10.1f} | {res.bic:<10.1f} | {res.rsquared:<10.3f} | {res.rsquared_adj:<10.3f}")

        best = sorted(results_list, key=lambda x: x['AIC'])[0]
        print(f"\nBest Model: {best['Model']}")
        print(best['Obj'].summary())
    except Exception as e:
        print(e)
        
print("\n\nAll Analyses Complete.")
