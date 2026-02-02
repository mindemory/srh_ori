import os, socket
from shutil import copyfile


def copyFiles(from_path, to_path):
    f_list = os.listdir(from_path)
    f_list = [f for f in f_list if f.endswith('.vhdr') or f.endswith('.vmrk') or f.endswith('.eeg')]
    for f in f_list:
        copyfile(os.path.join(from_path, f), os.path.join(to_path, f))
    

def load_paths(subjID, day):
    p = {}
    hostname = socket.gethostname()
    if hostname == 'syndrome' or hostname == 'zod.psych.nyu.edu' or hostname == 'zod':
        p['datd'] = '/d/DATD/datd/TMS_ori'
    else:
        p['datd'] = '/Users/mrugankdake/Documents/Clayspace/EEG_TMS/datd/TMS_ori'
    
    p['data'] = os.path.join(p['datd'], 'data')
    p['analysis'] = os.path.join(p['datd'], 'analysis')
    p['meta'] = os.path.join(p['analysis'], 'meta_analysis')
    p['EEGData'] = f"{p['datd']}/EEGData/sub{subjID:02d}/day{day:02d}"
    p['EEGfiles'] = f"{p['datd']}/EEGfiles/sub{subjID:02d}/day{day:02d}"
    # if not os.path.exists(p['EEGfiles']):
    #     os.makedirs(p['EEGfiles'])
        # copyFiles(p['EEGData'], p['EEGfiles'])
    
    p['EEGpy'] = f"{p['datd']}/EEGpy/sub{subjID:02d}/day{day:02d}"

    if not os.path.exists(p['EEGpy']):
        os.makedirs(p['EEGpy'])
    p['EEGroot'] = f"{p['EEGpy']}/sub{subjID:02d}_day{day:02d}"
    # p['raw_fpath'] = f"{p['EEGroot']}/sub{subjID:02d}_day{day:02d}.vhdr"
    # if ~os.path.exists(p['raw_fpath']):

    return p
