# Load the modules
from scipy.io import loadmat
import numpy as np
import pandas as pd
import os

# Hide deprecation warnings!
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import socket
hostname = socket.gethostname()

p = {}
if hostname == 'zod.psych.nyu.edu' or hostname == 'zod':
    p['datd'] = '/d/DATD/datd/srh_ori'
else:
    p['datd'] = '/Users/mrugankdake/Documents/Clayspace/EEG_TMS/datd/srh_ori'

p['data'] = os.path.join(p['datd'], 'data')
p['analysis'] = os.path.join(p['datd'], 'analysis')
p['meta'] = os.path.join(p['analysis'], 'meta_analysis')