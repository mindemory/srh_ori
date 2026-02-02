
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import generate_behavdf as gm
import warnings
import os

warnings.filterwarnings('ignore')
# User Request: No grid
sns.set_theme(style="white", context="notebook")
# Output Dir
output_dir = "/Users/mrugankdake/Documents/Clayspace/TMS_ori/Figures/exploratory"
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
df = gm.master_df.copy()
df.reset_index(drop=True, inplace=True)

# --- Feature Engineering ---
df.rename(columns={
    'TMScond_label': 'TMS',        
    'stimPF_label': 'Hemifield',   
    'targOriCategory': 'OriCat',
    'hemisphereStimulated': 'HemiStim'
}, inplace=True)

df['AdjustmentAbs'] = np.abs(df['initOffset'])
df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1
df['AbsError'] = np.abs(df['errOri_scaled'])

# Type Conversion
df['TMS'] = df['TMS'].astype('category')
df['Hemifield'] = df['Hemifield'].astype('category')
df['TMS'] = df['TMS'].cat.reorder_categories(['No TMS', 'Early TMS', 'Middle TMS'], ordered=True)
df['Hemifield'] = df['Hemifield'].cat.reorder_categories(['outPF', 'inPF'], ordered=True)

# =============================================================================
# FIGURE 3: Diagonal Attraction & Orientation Tuning
# =============================================================================
print("Generating Figure: Tuning & Diagonals...")

tms_conds = ['No TMS', 'Early TMS', 'Middle TMS']

fig2 = plt.figure(figsize=(18, 12))
gs = fig2.add_gridspec(3, 4) # 3 Rows (Bias, AbsErr, RT), 4 Cols (BarPlot + 3 TMS Columns)
fig2.suptitle("TMS Impact on Behavioral Metrics", fontsize=16)

# --- A. Diagonal Attraction (Left Column) ---
ax_bar = fig2.add_subplot(gs[:, 0])
sns.barplot(
    data=df, x='TMS', y='BiasTowardsDiagonal', hue='Hemifield',
    palette='Set2', errorbar='se', ax=ax_bar
)
ax_bar.set_title("Diag Attraction Magnitude")
ax_bar.set_ylabel("Bias Towards Diag (deg)")
ax_bar.axhline(0, color='k', linestyle='--')
ax_bar.legend(loc='upper right')

# --- B. Orientation Tuning Curves (Right 3 Columns) ---
# Metrics: Signed Bias (Row 0), AbsError (Row 1), RT (Row 2)
# Cols: No TMS, Early, Middle
metrics_tuning = [
    ('errOri_scaled', 'Signed Bias'),
    ('AbsError', 'Abs Error'),
    ('RT', 'Reaction Time')
]

for row_idx, (met, label) in enumerate(metrics_tuning):
    for col_idx, tms in enumerate(tms_conds):
        ax = fig2.add_subplot(gs[row_idx, col_idx+1]) # Shift right by 1
        
        subset = df[df['TMS'] == tms]
        
        sns.pointplot(
            data=subset, x='OriCat', y=met, hue='Hemifield',
            palette={'outPF': 'blue', 'inPF': 'red'},
            dodge=True, errorbar='se', ax=ax
        )
        
        ax.set_title(f"{label} | {tms}")
        if col_idx > 0: ax.set_ylabel("")
        else: ax.set_ylabel(label)
        
        if row_idx < 2: ax.set_xlabel("")
        else: ax.set_xlabel("Orientation")
        
        if row_idx == 0: ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        
        # Only legend in first col to save space
        if row_idx == 0 and col_idx == 0: pass 
        else: ax.get_legend().remove()

sns.despine()
plt.tight_layout()
save_path = os.path.join(output_dir, "Fig3_Mechamism_Tuning.png")
print(f"Saving to {save_path}")
plt.savefig(save_path, dpi=300)
plt.show()
