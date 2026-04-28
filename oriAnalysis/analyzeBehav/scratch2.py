import pandas as pd
import numpy as np
import generate_behavdf as gm

df = gm.master_df.copy()
df = df[df['LargeError'] == 0].copy()
df.rename(columns={'TMScond_label': 'TMS', 'stimPF_label': 'Hemifield', 'targOriCategory': 'OriCat', 'hemisphereStimulated': 'HemiStim'}, inplace=True)
df['TMS_Group'] = df['TMS'].replace({'Early TMS': 'TMS', 'Middle TMS': 'TMS'})
df['OriCat45'] = df['OriCat'].replace({30: 'Acute', 60: 'Acute', 120: 'Obtuse', 150: 'Obtuse'})

def classify_diag(row):
    hemi_stim = row['HemiStim']
    in_pf = row['Hemifield'] == 'inPF'
    is_acute = row['OriCat45'] == 'Acute'
    if hemi_stim == 'Left': return 'Radial' if (in_pf and not is_acute) or (not in_pf and is_acute) else 'Tangential'
    else: return 'Radial' if (in_pf and is_acute) or (not in_pf and not is_acute) else 'Tangential'

df['DiagType'] = df.apply(classify_diag, axis=1)

def make_tms_inpf_cat(row): return 'NoTMS' if row['TMS_Group'] == 'No TMS' else f"TMS_{row['Hemifield']}"
df['TMS_Condition'] = df.apply(make_tms_inpf_cat, axis=1)

sub1 = df[df['subjID'] == 1]
for cond in ['NoTMS', 'TMS_inPF', 'TMS_outPF']:
    for diag in ['Radial', 'Tangential']:
        sub = sub1[(sub1['TMS_Condition']==cond) & (sub1['DiagType']==diag)]
        print(f"Subj 1, {cond} - {diag}: {sub['OriCat45'].value_counts().to_dict()}")

