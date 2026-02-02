
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import generate_behavdf as gm

# Configuration
pd.set_option("display.max.columns", None)
pd.set_option("display.precision", 4)
msize = 10
axes_fontsize = 12
title_fontsize = 14

# Load dataframe
print("Loading data...")
df = gm.master_df
print("Data loaded.")

# Parameters
orientations = [30, 60, 120, 150]
conditions = ['No TMS', 'Early TMS', 'Middle TMS']
colors = {'inPF': 'red', 'outPF': 'blue'}
bins = np.linspace(-90, 90, 50) # Degree bins for error

'''
# Create figure: Rows = Orientations, Cols = TMS Conditions
fig, axes = plt.subplots(len(orientations), len(conditions), figsize=(15, 4 * len(orientations)), sharex=True, sharey=True)

for row_idx, ori in enumerate(orientations):
    for col_idx, tms_cond in enumerate(conditions):
        ax = axes[row_idx, col_idx]
        
        # Filter Data
        # Filter for InPF
        dfIn = df[(df['TMScond_label'] == tms_cond) & 
                  (df['stimPF_label'] == 'inPF') & 
                  (df['targOriCategory'] == ori)]
        
        # Filter for OutPF
        dfOut = df[(df['TMScond_label'] == tms_cond) & 
                   (df['stimPF_label'] == 'outPF') & 
                   (df['targOriCategory'] == ori)]
        
        # Plot Histograms (Density)
        # We use Histograms steps for outline
        if not dfIn.empty:
            ax.hist(dfIn['errOri_scaled'], bins=bins, density=True, histtype='step', 
                    color=colors['inPF'], linewidth=2, label='InPF', alpha=0.8)
            # Optional: Add filled KDE or semi-transparent fill
            # sns.kdeplot(dfIn['errOri_scaled'], ax=ax, color=colors['inPF'], fill=True, alpha=0.1)

        if not dfOut.empty:
            ax.hist(dfOut['errOri_scaled'], bins=bins, density=True, histtype='step', 
                    color=colors['outPF'], linewidth=2, label='OutPF', alpha=0.8)
            # sns.kdeplot(dfOut['errOri_scaled'], ax=ax, color=colors['outPF'], fill=True, alpha=0.1)

        # Formatting
        ax.set_title(f"Ori {ori}° | {tms_cond}", fontsize=title_fontsize)
        
        if row_idx == len(orientations) - 1:
            ax.set_xlabel('Error (degrees)', fontsize=axes_fontsize)
        if col_idx == 0:
            ax.set_ylabel('Density', fontsize=axes_fontsize)
            if row_idx == 0:
                ax.legend(loc='upper right')

        ax.set_xlim([-90, 90])
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
'''

# EDA Plots
# 1. distribution of initOri
# 2. distribution of finalOri
# 3. distribution of targOriActual
# 4. scatterplot of initOri vs finalOri
# 5. scatterplot of finalOri vs targOriActual

# Figure 1: Distributions
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

# initOri
sns.histplot(df['initOri'], bins=30, ax=axes1[0], kde=True)
axes1[0].set_title('Distribution of initOri', fontsize=title_fontsize)

# finalOri
sns.histplot(df['finalOri'], bins=30, ax=axes1[1], kde=True)
axes1[1].set_title('Distribution of
 finalOri', fontsize=title_fontsize)

# targOriActual
sns.histplot(df['targOriActual'], bins=30, ax=axes1[2], kde=True)
axes1[2].set_title('Distribution of targOriActual', fontsize=title_fontsize)

plt.tight_layout()
plt.show()


# Figure 2: Scatterplots
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))

# initOri vs finalOri
sns.scatterplot(x='initOri', y='finalOri', data=df, ax=axes2[0], alpha=0.3)
axes2[0].set_title('initOri vs finalOri', fontsize=title_fontsize)
axes2[0].set_aspect('equal')

# finalOri vs targOriActual
sns.scatterplot(x='finalOri', y='targOriActual', data=df, ax=axes2[1], alpha=0.3)
axes2[1].set_title('finalOri vs targOriActual', fontsize=title_fontsize)
axes2[1].set_aspect('equal')

plt.tight_layout()
plt.show()
