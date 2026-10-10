import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import os
import glob

# Set subject
sub_id = 'sub01'

# Paths
base_dir = f'/d/DATD/datd/TMS_ori/SIMNIBS_output/{sub_id}/simstandard/subject_overlays/'

surf_lh_path = os.path.join(base_dir, 'lh.central')
surf_rh_path = os.path.join(base_dir, 'rh.central')

# Find data files
try:
    data_lh_path = glob.glob(os.path.join(base_dir, 'lh.*.central.E.magn'))[0]
    data_rh_path = glob.glob(os.path.join(base_dir, 'rh.*.central.E.magn'))[0]
except IndexError:
    print(f"Error: Could not find data files in {base_dir}")
    exit()

# Load meshes and data
coords_lh, faces_lh = nib.freesurfer.read_geometry(surf_lh_path)
vals_lh = nib.freesurfer.read_morph_data(data_lh_path)
coords_rh, faces_rh = nib.freesurfer.read_geometry(surf_rh_path)
vals_rh = nib.freesurfer.read_morph_data(data_rh_path)

# Merge meshes
coords_total = np.vstack([coords_lh, coords_rh])
faces_total = np.vstack([faces_lh, faces_rh + len(coords_lh)])
vals_total = np.concatenate([vals_lh, vals_rh])

# Scale
vmax = 1.5
print(f"Subject: {sub_id}")

# --- 1. Create the figure ---
fig = plt.figure(figsize=(12, 10))
plotting.plot_surf_stat_map(
    [coords_total, faces_total], vals_total, 
    view='posterior',
    colorbar=True, 
    threshold=None,
    title=f'{sub_id.upper()} E-field - Posterior View',
    cmap='jet',
    vmax=vmax
)

# Set custom ticks
for axes in fig.get_axes():
    if axes.get_label() == '<colorbar>':
        axes.set_yticks([0, 0.75, 1.5])
        axes.set_yticklabels(['0', '0.75', '1.5'])

# --- 2. Save Mesh-Only PNG (Transparent) ---
# We'll hide the colorbar and title for this one if we wanted "mesh only", 
# but usually it's easier to just save the whole thing at high res.
# Let's hide the text for the "brain only" file:
title_obj = plt.gca().get_title()
plt.title('') # Hide title temporarily
for axes in fig.get_axes():
    if axes.get_label() == '<colorbar>':
        axes.set_visible(False)

plt.savefig('efield_viz_brain.png', transparent=True, dpi=600, bbox_inches='tight')
print("High-res brain PNG saved (efield_viz_brain.png)")

# --- 3. Save Labels-Only SVG ---
# Show text again, hide mesh
plt.title(f'{sub_id.upper()} E-field - Posterior View')
for axes in fig.get_axes():
    if axes.get_label() == '<colorbar>':
        axes.set_visible(True)
    
    # Hide the mesh artists
    for artist in axes.get_children():
        if isinstance(artist, PolyCollection) or 'Poly3DCollection' in str(type(artist)):
            artist.set_visible(False)

plt.savefig('efield_viz_labels.svg', transparent=True, bbox_inches='tight')
print("Vector labels/colorbar SVG saved (efield_viz_labels.svg)")

plt.show()
