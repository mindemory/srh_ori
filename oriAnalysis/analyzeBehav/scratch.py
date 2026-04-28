import pandas as pd
import numpy as np
import generate_behavdf as gm

df = gm.master_df.copy()
df = df[df['LargeError'] == 0].copy()

df.rename(columns={
    'TMScond_label': 'TMS',        
    'stimPF_label': 'Hemifield',   
    'targOriCategory': 'OriCat',
    'hemisphereStimulated': 'HemiStim'
}, inplace=True)

df['TMS_Group'] = df['TMS'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})

df['OriCat45'] = df['OriCat'].replace({
    30: 'Acute (30+60)', 60: 'Acute (30+60)',
    120: 'Obtuse (120+150)', 150: 'Obtuse (120+150)'
})

def classify_diag(row):
    hemi_stim = row['HemiStim']
    in_pf = row['Hemifield'] == 'inPF'
    is_acute = row['OriCat45'] == 'Acute (30+60)'
    
    if hemi_stim == 'Left':
        return 'Radial' if (in_pf and not is_acute) or (not in_pf and is_acute) else 'Tangential'
    else: # Right TMS
        return 'Radial' if (in_pf and is_acute) or (not in_pf and not is_acute) else 'Tangential'

df['DiagType'] = df.apply(classify_diag, axis=1)

def make_tms_inpf_cat(row):
    if row['TMS_Group'] == 'No TMS':
        return 'NoTMS'
    else:
        return f"TMS_{row['Hemifield']}"
        
df['TMS_Condition'] = df.apply(make_tms_inpf_cat, axis=1)

for cond in ['NoTMS', 'TMS_inPF', 'TMS_outPF']:
    for diag in ['Radial', 'Tangential']:
        sub = df[(df['TMS_Condition']==cond) & (df['DiagType']==diag)]
        print(f"{cond} - {diag}: {sub['OriCat45'].value_counts().to_dict()}")

