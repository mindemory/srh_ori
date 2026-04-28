import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os
import generate_behavdf as gm

# Subject-Location Mapping (1-based index)
# (Session, LocationOrdinalIndex)
LOCATIONS = {
    1: (1, 1), 2: (1, 1), 3: (1, 8), 5: (1, 4), 6: (1, 2),
    7: (2, 2), 8: (1, 5), 9: (1, 2), 11: (1, 2), 12: (1, 2),
    14: (1, 2), 16: (2, 3), 17: (1, 4), 18: (1, 3), 19: (1, 5),
    20: (2, 4), 21: (2, 1), 22: (1, 2), 23: (2, 6), 24: (1, 1),
    25: (2, 1), 26: (1, 1), 28: (2, 3)
}

subjects = sorted(gm.master_df['subjID'].unique())
print(f"Plotting for {len(subjects)} subjects...")

# Calculate grid size
n_subs = len(subjects)
n_cols = 5
n_rows = int(np.ceil(n_subs / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()

phosphene_results = {}

for i, subj_id in enumerate(subjects):
    ax = axes[i]
    sess, loc_ord = LOCATIONS[subj_id]
    subj_str = f"sub{int(subj_id):02d}"
    
    # Get hemisphere stimulated from master_df
    try:
        hemi = gm.master_df[gm.master_df['subjID'] == subj_id]['hemisphereStimulated'].iloc[0]
    except:
        hemi = "?"

    file_path = f'/d/DATD/datd/TMS_ori/data/phosphene_data/{subj_str}/tmsRtnTpy_{subj_str}_sess{sess:02d}.mat'
    
    if not os.path.exists(file_path):
        ax.set_title(f"{subj_str} ({hemi}) - Missing File", fontsize=10, color='red')
        continue
    
    try:
        data = scipy.io.loadmat(file_path)
        tms = data['tmsRtnTpy']
        resp = tms['Response'][0,0]
        params = tms['Params'][0,0]
        screen = params['screen'][0,0]
        
        res_x = float(np.squeeze(screen['screenXpixels']))
        res_y = float(np.squeeze(screen['screenYpixels']))
        
        coil_locs = resp['CoilLocation'][0,0].flatten()
        detections = resp['Detection'][0,0].flatten()
        drawings = resp['Drawing'][0,0][0]
        
        # FIND THE N-TH ACTIVE LOCATION (with > 1 detection)
        unique_locs = np.unique(coil_locs)
        active_locs = []
        for l in unique_locs:
            if np.sum(detections[coil_locs == l]) > 1:
                active_locs.append(l)
        active_locs = sorted(active_locs)
        
        if len(active_locs) < loc_ord:
            ax.set_title(f"{subj_str} ({hemi}) - Missing Loc", color='orange')
            continue
            
        real_loc_idx = active_locs[loc_ord - 1]
        trials = np.where((coil_locs == real_loc_idx) & (detections == 1))[0]
        
        polygons = []
        for t_idx in trials:
            pts = drawings[t_idx]
            if pts.size < 6: continue 
            
            poly = Polygon(pts)
            if not poly.is_valid: poly = poly.buffer(0)
            polygons.append(poly)
            
            ax.plot(pts[:, 0], pts[:, 1], color='gray', alpha=0.3, lw=0.5)
            ax.fill(pts[:, 0], pts[:, 1], color='gray', alpha=0.05)

        union_area = 0
        if polygons:
            # Intersection (for plotting)
            intersect = polygons[0]
            for p in polygons[1:]:
                intersect = intersect.intersection(p)
            
            # Union (for size calculation)
            all_union = unary_union(polygons)
            union_area = all_union.area
            phosphene_results[subj_id] = union_area

            if not intersect.is_empty:
                if intersect.geom_type == 'Polygon':
                    ix, iy = intersect.exterior.xy
                    ax.plot(ix, iy, 'r-', lw=2)
                    ax.fill(ix, iy, color='red', alpha=0.2)
                elif intersect.geom_type == 'MultiPolygon':
                    for p in intersect.geoms:
                        ix, iy = p.exterior.xy
                        ax.plot(ix, iy, 'r-', lw=2)
                        ax.fill(ix, iy, color='red', alpha=0.2)

        # Plot fixation cross
        cx, cy = res_x / 2, res_y / 2
        cs = 20
        ax.plot([cx-cs, cx+cs], [cy, cy], 'k-', lw=1, alpha=0.3)
        ax.plot([cx, cx], [cy-cs, cy+cs], 'k-', lw=1, alpha=0.3)

        ax.set_title(f"{subj_str} ({hemi})\nArea: {union_area:.1f}", fontsize=11, fontweight='bold')
        ax.set_xlim(0, res_x)
        ax.set_ylim(0, res_y)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        
    except Exception as e:
        ax.set_title(f"{subj_str} ({hemi}) - Error", fontsize=10, color='orange')
        print(f"Error processing {subj_str}: {e}")

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
output_grid = "Phosphene_Maps_Grid_AllSubjects.png"
plt.savefig(output_grid, dpi=300)
print(f"Final corrected grid saved to {output_grid}")

# PRINT AREAS FOR HARD-CODING
print("PHOSPHENE_AREAS = {")
for s_id, area in phosphene_results.items():
    print(f"    {s_id}: {area:.2f},")
print("}")
