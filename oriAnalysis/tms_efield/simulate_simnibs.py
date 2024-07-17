from simnibs import sim_struct, brainsight, run_simnibs
import os
#subs = [1, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27]
subs = [1, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27]

# stim_intensities = [55, 47, 56, 63, 60, 53, 54, 55, 60, 44, 46, 58, 47, 55, 50, 53, 50, 48, 50, 50]
# stim_dIdt = [round((a / 60) * 87 *1e6) for a in stim_intensities]

stim_dIdt = [1e6 for a in subs]

coilpath = '/Users/mrugank/Applications/SimNIBS-4.0/simnibs_env/lib/python3.9/site-packages/simnibs/resources/coil_models/Drakaki_BrainStim_2022/MagVenture_Cool-B70.ccd'
masterpath = "/d/DATD/datd/MD_TMS_EEG/SIMNIBS_output/"
brainsightpath = masterpath+"session_txts/"

#s.add_tmslist(tms_list_targets)
for idx in range(len(subs)):
    sub = subs[idx]
    sub_id = f"sub{sub:02d}"
    navfile = f"{brainsightpath}{sub_id}.txt"
    tms_list_targets, tms_list_samples = brainsight().read(navfile, stim_dIdt[idx])

    # print(tms_list_samples)
    m2mfoldpath = f"{masterpath}{sub_id}/m2m_{sub_id}"
    #simfoldpath = f"{masterpath}{sub_id}/simulation"
    simfoldpath = f"{masterpath}{sub_id}/simstandard"

    if not os.path.exists(simfoldpath):
        os.mkdir(simfoldpath)

        S = sim_struct.SESSION()
        S.fnamehead = os.path.join(m2mfoldpath, f"{sub_id}.msh")
        S.subpath = m2mfoldpath
        S.pathfem = simfoldpath
        S.open_in_gmsh = False
        S.map_to_surf = True
        S.map_to_vol = True
        S.map_to_fsavg = False
        S.map_to_mni = False
        S.fields= 'eEjJvDs'
        #S.tissues_in_niftis = "all"
        tmslist = S.add_tmslist(tms_list_targets)
        tmslist.fnamecoil = coilpath 
        run_simnibs(S, cpus = 4)