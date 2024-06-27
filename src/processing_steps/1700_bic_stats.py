import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, hdf5_root_fast, binary_root
import h5py
from lib.cpp.cpu_seq.io import load_slice, write_slice
from lib.py.helpers import block_info, load_block, commandline_args
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# sample : [new_start, new_end, old_start, old_end]
ranges = {
    '770_pag' : np.array([ 0, 1, 0, 150*8 ]),
    '770c_pag' : np.array([ 1000, 2000, 2250, 3000 ]),
    '771c_pag' : np.array([ 1200, 2000, 0, 1000 ]),
    '772_pag' : np.array([ 100*8, 300*8, 300*8, 400*8 ]),
    '775c_pag' : np.array([ 800, 2400, 2600, 3200 ]),
    '810c_pag' : np.array([ 700*2, 850*2, 0, 600*2 ]),
    '811_pag' : np.array([ 300*2, 800*2, 1000*2, 1500*2 ]),
    'novisim' : np.array([ 200, 500, 600, 800 ]),
}

if __name__ == '__main__':
    sample, scale = commandline_args({
        "sample" : "<required>",
        "scale" : 1}
    )

    bics = np.load(f'{hdf5_root}/processed/bic/{sample}_bics.npy')

    range = ranges[sample] // scale
    new_bics = bics[range[0]:range[1]]
    old_bics = bics[range[2]:range[3]]

    new_mean, new_std = np.mean(new_bics), np.std(new_bics)
    old_mean, old_std = np.mean(old_bics), np.std(old_bics)

    print (f'New BIC: {new_mean:.04f} +/- {new_std:.04f}')
    print (f'Old BIC: {old_mean:.04f} +/- {old_std:.04f}')