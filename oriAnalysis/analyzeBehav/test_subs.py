import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import socket

# Hide deprecation warnings!
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

hostname = socket.gethostname()

p = {}
if hostname == 'zod.psych.nyu.edu' or hostname == 'zod':
    p['datd'] = '/d/DATD/datd/TMS_ori'
else:
    p['datd'] = '/Users/mrugankdake/Documents/Clayspace/TMS_ori'

p['data'] = os.path.join(p['datd'], 'data', 'ori_data')
p['analysis'] = os.path.join(p['datd'], 'analysis')
p['meta'] = os.path.join(p['analysis'], 'meta_analysis')

# Output Dir
output_dir = os.path.join(p['datd'], 'Figures', 'visualbiases')
os.makedirs(output_dir, exist_ok=True)

print('Creating a dataframe for all subjects including 27...')
# Find subjects and days that have been run so far
sub_dirs = [d for d in os.listdir(p['analysis']) if d.startswith('sub')]
subjIDs = sorted([int(ss[-2:]) for ss in sub_dirs])

# Explicitly KEEP subject 27 if they exist in the analysis folder
print(f"Loading {len(subjIDs)} subjects: {subjIDs}")
print()

summary_df = pd.read_csv(os.path.join(p['analysis'], 'V1ori_TMS_meta.csv'))
master_df = pd.DataFrame()

for ss in subjIDs:
    sub_csv_path = os.path.join(p['analysis'], f'sub{ss:02}', f'sub{ss:02}_behav.csv')
    if not os.path.exists(sub_csv_path):
        print(f"Skipping Subject {ss} - CSV not found.")
        continue
        
    temp_df = pd.read_csv(sub_csv_path)
    
    # Add metadata if it exists
    if ss in summary_df['Subject ID'].values:
        temp_df['subjInitials'] = summary_df[summary_df['Subject ID'] == ss].iloc[0]['Subject Initials']
        temp_df['hemisphereStimulated'] = summary_df[summary_df['Subject ID'] == ss].iloc[0]['Hemisphere stimulated']
    
    if ss == 7:
        # Drop rows for day == 2 and block == 7, 8, 12, 13
        temp_df = temp_df[~((temp_df['day'] == 2) & (temp_df['block'].isin([7, 8, 12, 13])))]
    
    master_df = pd.concat([master_df, temp_df], ignore_index=True)

# Calculate Angular Error and Absolute Error
errTemp = master_df['finalOri'] - master_df['targOriActual']
master_df['errOri_scaled'] = (errTemp + 90) % 180 - 90
master_df['AbsError'] = np.abs(master_df['errOri_scaled'])

# Mark Outliers
global_mean_err = master_df['errOri_scaled'].mean()
global_sd_err = master_df['errOri_scaled'].std()
lower_z = global_mean_err - 3 * global_sd_err
upper_z = global_mean_err + 3 * global_sd_err

master_df['LargeError'] = 0
master_df.loc[(master_df['errOri_scaled'] < lower_z) | (master_df['errOri_scaled'] > upper_z), 'LargeError'] = 1

print(f"Total LargeErrors Outliers defined by 3SD: {master_df['LargeError'].sum()}/{len(master_df)}")

# Remove LargeErrors for plotting
df_clean = master_df[master_df['LargeError'] == 0].copy()

# Treat Subject as Categorical for Plotting
df_clean['subjID'] = df_clean['subjID'].astype(str)

# Add TMS Condition Label
df_clean['TMScond_label'] = df_clean['TMScond'].apply(lambda x: 'No TMS' if x == 1 else 'Early TMS' if x == 2 else 'Middle TMS')

print("\nGenerating Subject Performance Plots...")
fig, axes = plt.subplots(3, 1, figsize=(14, 15))
fig.suptitle("Subject Performance (Including Subject 27)", fontsize=16)

# Highlight Subject 27
subjects = sorted(df_clean['subjID'].unique(), key=lambda x: int(x))
palette = {subj: 'red' if subj == '27' else 'gray' for subj in subjects}

# 1. Mean Absolute Error by Subject
sns.barplot(
    data=df_clean, x='subjID', y='AbsError', 
    palette=palette, errorbar='se', ax=axes[0], 
    capsize=0.1, errwidth=1.5, order=subjects
)
axes[0].set_title("Mean Absolute Angular Error by Subject")
axes[0].set_ylabel("Absolute Error (deg)")
axes[0].set_xlabel("Subject ID")
axes[0].axhline(df_clean['AbsError'].mean(), color='black', linestyle='--', alpha=0.5, label='Grand Mean')
axes[0].legend()

# 2. Mean Reaction Time by Subject
sns.barplot(
    data=df_clean, x='subjID', y='RT', 
    palette=palette, errorbar='se', ax=axes[1], 
    capsize=0.1, errwidth=1.5, order=subjects
)
axes[1].set_title("Mean Reaction Time by Subject")
axes[1].set_ylabel("Reaction Time (s)")
axes[1].set_xlabel("Subject ID")
axes[1].axhline(df_clean['RT'].mean(), color='black', linestyle='--', alpha=0.5, label='Grand Mean')
axes[1].legend()

# 3. Number of Trials by TMS Condition
tms_order = ['No TMS', 'Early TMS', 'Middle TMS']
sns.countplot(
    data=df_clean, x='subjID', hue='TMScond_label',
    palette={'No TMS': 'gray', 'Early TMS': 'blue', 'Middle TMS': 'green'},
    ax=axes[2], order=subjects, hue_order=tms_order
)
axes[2].set_title("Number of Trials (After Exclusion) by TMS Condition")
axes[2].set_ylabel("Trial Count")
axes[2].set_xlabel("Subject ID")
axes[2].legend(title='TMS Condition')

sns.despine(fig=fig)
plt.tight_layout()

save_path = os.path.join(output_dir, "test_subs_performance.png")
print(f"Saving to {save_path}")
plt.savefig(save_path, dpi=300)
plt.close(fig)

print("Done! Performance plot saved.")

# ==============================================================================
# New Figure: Distribution of Angular Error
# ==============================================================================
print("\nGenerating Distribution of Angular Error (errOri_scaled)...")
fig2, ax2 = plt.subplots(figsize=(16, 7))
fig2.suptitle("Distribution of Angular Error per Subject", fontsize=16)

sns.stripplot(
    data=df_clean,
    x='subjID',
    y='errOri_scaled',
    hue='TMScond_label',
    palette={'No TMS': 'gray', 'Early TMS': 'blue', 'Middle TMS': 'green'},
    order=subjects,
    hue_order=tms_order,
    dodge=True,
    alpha=0.5,
    size=3,
    jitter=True,
    ax=ax2
)

# Optional: Adding a zero line to show the target
ax2.axhline(0, color='red', linestyle='--', lw=1)
ax2.set_xlabel("Subject ID")
ax2.set_ylabel("Angular Error (deg)")
# Move legend out of the plot
ax2.legend(title="TMS Condition", bbox_to_anchor=(1.05, 1), loc='upper left')

sns.despine(fig=fig2)
plt.tight_layout()

save_path_dist = os.path.join(output_dir, "test_subs_error_distribution.png")
print(f"Saving to {save_path_dist}")
plt.savefig(save_path_dist, dpi=300, bbox_inches='tight')
plt.close(fig2)
# ==============================================================================
# New Figure: Timeout Rate (% of Trials reaching Max RT)
# ==============================================================================
print("\nGenerating Timeout Rate Plot...")

# We expect the max RT to be around 3.0s, let's create a binary timeout variable 
# If reaction time >= max_allowed_rt (typically 3.0s in typical experiments), flag as timeout
max_rt = round(df_clean['RT'].max(), 1)  # Usually 3.0 or similar
print(f"Assumed Max RT for Timeout cutoff is: {max_rt}s")

df_clean['is_timeout'] = (df_clean['RT'] >= max_rt).astype(int)

# Group by subject and calculate percentage of timeouts
timeout_rates = df_clean.groupby('subjID')['is_timeout'].mean().reset_index()
timeout_rates['timeout_pct'] = timeout_rates['is_timeout'] * 100

fig3, ax3 = plt.subplots(figsize=(14, 6))
fig3.suptitle(f"Percentage of Timeout Trials (RT >= {max_rt}s) per Subject", fontsize=16)

sns.barplot(
    data=timeout_rates, 
    x='subjID', 
    y='timeout_pct', 
    palette=palette, 
    order=subjects,
    ax=ax3
)

ax3.set_xlabel("Subject ID")
ax3.set_ylabel("% of Trials Timed Out")
ax3.axhline(timeout_rates['timeout_pct'].mean(), color='black', linestyle='--', alpha=0.5, label='Grand Mean')
ax3.legend()

sns.despine(fig=fig3)
plt.tight_layout()

save_path_timeout = os.path.join(output_dir, "test_subs_timeout_rates.png")
print(f"Saving to {save_path_timeout}")
plt.savefig(save_path_timeout, dpi=300)
plt.close(fig3)
# ==============================================================================
# New Figure: Target vs Reported Orientation Scatter Plots
# ==============================================================================
print("\nGenerating Target vs Reported Orientation Scatter Plots...")

# Using seaborn's relplot to create a faceted grid of scatter plots
# We want one subplot per subject
g = sns.relplot(
    data=df_clean,
    x='targOriActual',
    y='finalOri',
    hue='TMScond_label',
    col='subjID',
    col_wrap=6, # 6 columns per row, so 4 rows for 24 subjects
    palette={'No TMS': 'gray', 'Early TMS': 'blue', 'Middle TMS': 'green'},
    hue_order=tms_order,
    height=3,
    aspect=1,
    alpha=0.6,
    s=15, # dot size
    col_order=subjects
)

g.set_axis_labels("Target Orientation (deg)", "Reported Orientation (deg)")
g.set_titles("Subject {col_name}")

# Add the identity line (x=y) to each subplot
for ax in g.axes.flat:
    ax.plot([0, 180], [0, 180], color='red', linestyle='--', linewidth=1, zorder=0)
    ax.set_xlim([0, 180])
    ax.set_ylim([-90, 270]) # Adjusted as finalOri and targOriActual domains vary, 0-180 is typically fine
    
    # Force axes to be strictly [0, 180] if they wrap back around mathematically
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)

g.fig.suptitle("Target vs Reported Orientation by Subject", fontsize=20, y=1.02)

save_path_scatter = os.path.join(output_dir, "test_subs_scatter_target_vs_reported.png")
print(f"Saving to {save_path_scatter}")
g.savefig(save_path_scatter, dpi=300, bbox_inches='tight')
plt.close(g.fig)

print("Done! Scatter plot matrix saved.")
print("Done! Timeout plot saved.")
print("Done! Distribution plot saved.")
