
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import generate_behavdf as gm

# Configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})
colors = ["#e74c3c", "#3498db", "#2ecc71"] # Custom palette

print("Loading data...")
df = gm.master_df.copy()

# Data Prep (Same as B02)
df['unique_block'] = df['subjID'].astype(str) + '_' + df['day'].astype(str) + '_' + df['block'].astype(str)
df['prevError'] = df.groupby('unique_block')['errOri_scaled'].shift(1)
df['absError'] = np.abs(df['errOri_scaled'])
df.rename(columns={'TMScond_label': 'TMS', 'stimPF_label': 'Hemifield', 'targOriCategory': 'OriCat'}, inplace=True)

# Order factors
df['TMS'] = pd.Categorical(df['TMS'], categories=['No TMS', 'Early TMS', 'Middle TMS'], ordered=True)
df['Hemifield'] = pd.Categorical(df['Hemifield'], categories=['outPF', 'inPF'], ordered=True)

# Remove outliers for clearer plots (optional, but good for visualization)
# df = df[np.abs(df['errOri_scaled']) < 45] 

print("Generating plots...")

# --- Plot 1: Bias Interaction (TMS x Hemifield) ---
# Question: Does Early TMS in InPF cause a bias shift?
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(
    data=df, x="TMS", y="errOri_scaled", hue="Hemifield",
    errorbar="se", capsize=.1, palette="muted", ax=ax1
)
ax1.set_title("Bias Interaction: TMS x Hemifield", fontsize=16)
ax1.set_ylabel("Signed Error (degrees) \n (+ = CW, - = CCW)", fontsize=12)
ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# --- Plot 2: Serial Dependence (PrevError vs Current Error) ---
# Question: Does PrevError predict Current Error? Does TMS change it?
# Binning prevError for cleaner visualization (DoG-like curve)
df['prevError_bin'] = pd.cut(df['prevError'], bins=np.linspace(-60, 60, 21), labels=False)
# Calculate mean of bins for X-axis
bin_means = np.linspace(-60, 60, 21)
# Create a temporary DF for bin plotting to map bin index back to degrees
bin_map = dict(enumerate((bin_means[:-1] + bin_means[1:]) / 2))

# We'll plot the raw scatter/regplot or binned means
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Option A: Regression Lines
sns.lmplot(
    data=df, x="prevError", y="errOri_scaled", hue="TMS",
    scatter=False, # Too noisy with scatter
    ci=95, palette="dark", height=6, aspect=1.5,
    x_bins=15 # This bins the scatter points for us!
)
plt.title("Serial Dependence by TMS Condition", fontsize=16)
plt.xlabel("Previous Error (degrees)", fontsize=12)
plt.ylabel("Current Error (degrees)", fontsize=12)
plt.xlim(-45, 45) # Focus on the central range where SD is strongest
plt.ylim(-10, 10)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()


# --- Plot 3: Orientation Bias (Cardinal Repulsion) ---
# Question: How does error vary across the 180 deg space?
fig3, ax3 = plt.subplots(figsize=(10, 6))
# We use targOriActual (continuous) rounded/binned or OriCat
sns.lineplot(
    data=df, x="targOriActual", y="errOri_scaled", hue="Hemifield",
    estimator='mean', errorbar='se', palette="viridis", ax=ax3
)
ax3.set_title("Systematic Orientation Bias", fontsize=16)
ax3.set_xlabel("Target Orientation (degrees)", fontsize=12)
ax3.set_ylabel("Signed Error (degrees)", fontsize=12)
ax3.set_xticks(np.arange(0, 181, 30))
ax3.axhline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()
