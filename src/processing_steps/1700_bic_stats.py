#! /usr/bin/python3
'''
This script computes statistics on the Bone Implant Contact (BIC) metric computed in the previous step.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.constants import *
from config.paths import hdf5_root
from lib.py.commandline_args import default_parser
import numpy as np

# sample : [new_start, new_end, old_start, old_end]
# Determines the z-range of new and old bone for each sample.
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
    args = default_parser(__doc__).parse_args()

    bics_path = f'{hdf5_root}/processed/bics/{args.sample}/{args.sample_scale}x/bics.npy'
    if args.verbose >= 1: print (f'Loading BICs for {bics_path}')
    bics = np.load(bics_path)

    range = ranges[args.sample] // args.sample_scale
    new_bics = bics[range[0]:range[1]]
    old_bics = bics[range[2]:range[3]]

    new_mean, new_std = np.mean(new_bics), np.std(new_bics)
    old_mean, old_std = np.mean(old_bics), np.std(old_bics)

    print (f'New BIC: {new_mean:.04f} +/- {new_std:.04f}')
    print (f'Old BIC: {old_mean:.04f} +/- {old_std:.04f}')