import os
import glob
from simnibs import sim_struct, brainsight, run_simnibs

# Set subjects to run
subs = [14, 16, 17, 18, 19, 22, 23, 24, 25, 26, 28]

# Paths
masterpath = '/d/DATD/datd/TMS_ori/SIMNIBS_output/'
coilpath = '/Users/mrugank/Applications/SimNIBS-4.1/simnibs_env/lib/python3.9/site-packages/simnibs/resources/coil_models/Drakaki_BrainStim_2022/MagVenture_Cool-B70.ccd'

for sub in subs:
    sub_id = f"sub{sub:02d}"
    navfile = f"/d/DATD/datd/TMS_ori/brainsight/session_txts/{sub_id}.txt"
    
    if not os.path.exists(navfile):
        print(f"Warning: Brainsight file {navfile} not found. Skipping {sub_id}.")
        continue

    # Check if overlays already exist
    simfoldpath = os.path.join(masterpath, sub_id, "simstandard")
    overlay_path = os.path.join(simfoldpath, "subject_overlays")
    if os.path.exists(overlay_path):
        print(f"Skipping {sub_id}: Simulation results already exist in {overlay_path}")
        continue

    print(f"Reading Brainsight file: {navfile}")
    try:
        tms_list_targets, tms_list_samples = brainsight().read(navfile)
    except Exception as e:
        print(f"Error reading brainsight file for {sub_id}: {e}")
        continue

    m2mfoldpath = os.path.join(masterpath, sub_id, f"m2m_{sub_id}")

    if not os.path.exists(m2mfoldpath):
        print(f"Warning: m2m folder {m2mfoldpath} not found. Skipping {sub_id}.")
        continue

    if not os.path.exists(simfoldpath):
        os.makedirs(simfoldpath)

    # Delete existing .mat files to avoid OSError
    for mat_file in glob.glob(os.path.join(simfoldpath, "*.mat")):
        print(f"Deleting existing .mat file: {mat_file}")
        os.remove(mat_file)

    print(f"Starting simulation for {sub_id} in {simfoldpath}")
    S = sim_struct.SESSION()
    S.fnamehead = os.path.join(m2mfoldpath, f"{sub_id}.msh")
    S.subpath = m2mfoldpath
    S.pathfem = simfoldpath
    S.open_in_gmsh = False
    S.map_to_surf = True
    S.map_to_vol = True
    S.map_to_fsavg = False
    S.map_to_mni = False
    S.fields = 'eEjJvDs'
    
    tmslist = S.add_tmslist(tms_list_targets)
    tmslist.fnamecoil = coilpath 
    
    try:
        run_simnibs(S, cpus=4)
    except Exception as e:
        print(f"Error running simulation for {sub_id}: {e}")

print("Simulation run completed.")
