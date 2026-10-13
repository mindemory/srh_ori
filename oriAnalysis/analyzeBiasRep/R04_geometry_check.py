"""
R04_geometry_check.py

Visual check of the Radial/Tangential labels. For every subject, draws the two
stimulus locations at their real screen positions and, at each location, the
category orientation labelled Radial (solid) and Tangential (dashed), using the
task's own drawing convention (drawTextures.m 'Response': x + cos(ori),
y - sin(ori) in screen pixels). A correct label means the solid line points
along the dotted line to fixation at both locations.

Usage: python R04_geometry_check.py [rep|orig]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from biasrep_common import COMPARE_DIR, STUDIES
from study_results import load_study

study = sys.argv[1] if len(sys.argv) > 1 else 'rep'
df = load_study(study)   # paper's Radial/Tangential labels, recorded stimulus positions
out_dir = COMPARE_DIR
os.makedirs(out_dir, exist_ok=True)

subs = sorted(df['subjID'].unique())
ncol = 4
fig, axs = plt.subplots(int(np.ceil(len(subs) / ncol)), ncol, figsize=(4 * ncol, 3.4 * np.ceil(len(subs) / ncol)))
for ax, s in zip(axs.flat, subs):
    sd = df[df['subjID'] == s]
    L = sd['stimSize'].iloc[0] * 1.5  # response-line length used by the task
    ax.plot(0, 0, 'k+', ms=10, mew=2)
    for hemi, col in [('inPF', 'tab:red'), ('outPF', 'tab:gray')]:
        h = sd[sd['Hemifield'] == hemi]
        x, y = h['targX'].iloc[0] - h['xCenter'].iloc[0], -(h['targY'].iloc[0] - h['yCenter'].iloc[0])  # y up
        ax.plot([0, 1.35 * x], [0, 1.35 * y], ':', color='0.5', lw=1)
        ax.add_patch(plt.Circle((x, y), sd['stimSize'].iloc[0] / 2, fill=False, color=col, lw=1.5))
        for dtype, ls, lw in [('Radial', '-', 3), ('Tangential', '--', 1.5)]:
            for ori in sorted(h[h['DiagType'] == dtype]['OriCat'].unique()):
                dx, dy = np.cos(np.radians(ori)) * L / 2, np.sin(np.radians(ori)) * L / 2
                ax.plot([x - dx, x + dx], [y - dy, y + dy], ls=ls, lw=lw, color=col)
        rad = '/'.join(str(o) for o in sorted(h[h.DiagType == 'Radial'].OriCat.unique()))
        ax.text(x, y - 0.75 * L, f'{hemi}\nR={rad}°', ha='center', va='top', fontsize=8, color=col)
    ax.set_aspect('equal'); ax.set_xlim(-520, 520); ax.set_ylim(-520, 160)
    ax.set_xticks([]); ax.set_yticks([])
    side = 'left' if sd[sd['Hemifield'] == 'inPF']['VF'].iloc[0] == 'LVF' else 'right'
    ax.set_title(f'sub{s:02d} · {sd["HemiStim"].iloc[0]} hemisphere · inPF on {side} · {sd["Elevation"].iloc[0]:.0f}° below horizontal', fontsize=8)
for ax in axs.flat[len(subs):]:
    ax.axis('off')
fig.suptitle(f'{STUDIES[study]["label"]}: Radial (solid) and Tangential (dashed) category orientations at each stimulus location\n'
             'dotted = line through fixation (+); red = inPF, grey = outPF', y=1.0)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(out_dir, f'GeometryCheck_{study}.{ext}'), dpi=200, bbox_inches='tight')
print(f'Saved {os.path.join(out_dir, f"GeometryCheck_{study}.png")}')
