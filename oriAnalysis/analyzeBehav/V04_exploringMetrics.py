import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import generate_behavdf as gm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="white", context="notebook")

# Output Dir
output_dir = os.path.join(gm.p['datd'], 'Figures', 'clusteringExploration')
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

# Metrics
df['BiasTowardsDiagonal'] = df['errOri_scaled']
df.loc[df['OriCat'].isin([60, 150]), 'BiasTowardsDiagonal'] *= -1
df['AbsError'] = np.abs(df['errOri_scaled'])
df['SignedError'] = df['errOri_scaled'] 

# Features for Clustering
features = ['AbsError', 'BiasTowardsDiagonal', 'RT']
print(f"Features used for clustering: {features}")

# --- Outlier Analysis (Added Request) ---
# Subject-level means of Signed Error
subj_means = df.groupby('subjID', observed=True)['errOri_scaled'].mean()

# --- Outlier Analysis ---
# Using the 'LargeError' flag from generate_behavdf
n_large = df['LargeError'].sum()
n_total = len(df)
print(f"LargeError Flag Check: {n_large}/{n_total} ({(n_large/n_total)*100:.1f}%) trials marked as LargeErrors.")
print("Dropping LargeError trials for subsequent analysis...")
df = df[df['LargeError'] == 0].copy()
print(f"Remaining trials: {len(df)}")
print("------------------------------------------------------\n")

# --- Clustering Analysis per Subject ---
subjects = df['subjID'].unique()
n_subjs = len(subjects)

# Setup Grid
n_cols = 5
n_rows = int(np.ceil(n_subjs / n_cols))

# =============================================================================
# FIGURE 1: Elbow / Silhouette Curves
# =============================================================================
print("Generating Figure 1: Elbow Curves...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
axes = axes.flatten()

for i, subj in enumerate(subjects):
    ax = axes[i]
    subj_data = df[df['subjID'] == subj][features].dropna()
    
    if len(subj_data) < 20: 
        print(f"Skipping Subject {subj} (Not enough data: {len(subj_data)})")
        continue

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subj_data)
    
    inertias = []
    silhouettes = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        if k >= 2:
            try:
                sil = silhouette_score(X_scaled, kmeans.labels_)
                silhouettes.append(sil)
            except:
                silhouettes.append(0)
        else:
            silhouettes.append(0) 

    ax.plot(K_range, inertias, 'bo-', markersize=6, linewidth=2, label='Inertia')
    ax2 = ax.twinx()
    ax2.plot(K_range[1:], silhouettes[1:], 'r--s', alpha=0.6, label='Silhouette')
    
    ax.set_title(f"Subject {subj}")
    ax.set_xlabel('K')
    if i % n_cols == 0: ax.set_ylabel('Inertia', color='b')
    
    ax.grid(False)
    ax2.grid(False)
    
    if i == 0: 
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

for j in range(i+1, len(axes)): fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "V02_KMeans_Elbow_per_Subject.png"), dpi=300)
plt.close()


# =============================================================================
# PAIRWISE 2D PLOTS (K=2)
# =============================================================================
pairs = [
    ('AbsError', 'BiasTowardsDiagonal', 'V02_Pair1_Abs_Bias.png'),
    ('AbsError', 'RT',                  'V02_Pair2_Abs_RT.png'),
    ('BiasTowardsDiagonal', 'RT',       'V02_Pair3_Bias_RT.png')
]

for x_col, y_col, filename in pairs:
    print(f"Generating {filename} ({x_col} vs {y_col})...")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.flatten()
    
    for i, subj in enumerate(subjects):
        ax = axes[i]
        
        subj_data = df[df['subjID'] == subj][features].dropna()
        if len(subj_data) < 20: continue
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(subj_data)
        
        # Fit K=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Plot
        plot_data = subj_data.copy()
        plot_data['Cluster'] = labels.astype(str)
        
        sns.scatterplot(
            data=plot_data, x=x_col, y=y_col, hue='Cluster',
            ax=ax, palette='deep', alpha=0.6, s=15, legend=(i==0)
        )
        
        ax.set_title(f"Subject {subj}")
        if i % n_cols != 0: ax.set_ylabel("")
        if i < (n_rows-1)*n_cols: ax.set_xlabel("")
        
        ax.grid(False)
        
        if i == 0:
            ax.legend(loc='upper right', title="Cluster")
            
    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


# =============================================================================
# DISTRIBUTION PLOTS (Standardized)
# =============================================================================
# Config for fixed axes/bins
dist_config = {
    'AbsError':            {'xlim': (0, 30),   'bins': 30},
    'BiasTowardsDiagonal': {'xlim': (-45, 45), 'bins': 45},
    'SignedError':         {'xlim': (-30, 30), 'bins': 60},
    'RT':                  {'xlim': (0, 3),    'bins': 60}
}
dist_metrics = list(dist_config.keys())

for met in dist_metrics:
    conf = dist_config[met]
    print(f"Generating Distribution Plot for {met} (Range: {conf['xlim']})...")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    axes = axes.flatten()
    
    for i, subj in enumerate(subjects):
        ax = axes[i]
        
        # Select data for subject and metric
        # We need specific columns for hue
        subj_data = df[df['subjID'] == subj][[met, 'TMSstatus']].dropna()
        
        sns.histplot(
            data=subj_data,
            x=met,
            hue='TMSstatus',
            kde=True, 
            ax=ax, 
            palette={'No TMS': 'k', 'TMS': 'r'},
            alpha=0.3, 
            line_kws={'linewidth': 1.5},
            binrange=conf['xlim'],
            bins=conf['bins'],
            element="step"
        )
        
        ax.set_title(f"Subject {subj}")
        ax.set_xlabel(met)
        ax.set_xlim(conf['xlim'])
        
        if i % n_cols != 0: ax.set_ylabel("")
        
        ax.grid(False)
        
        # Legend only on first plot
        if i == 0:
            sns.move_legend(ax, "upper right")
        elif ax.get_legend():
            ax.get_legend().remove()

    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    
    plt.tight_layout()
    filename = f"V03_Dist_{met}.png" # Simplified filename
    save_path = os.path.join(output_dir, filename)
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

print("Script Complete.")
