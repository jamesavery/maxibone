import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, hdf5_root_fast, binary_root
import edt
import h5py
from lib.cpp.cpu_seq.io import load_slice, write_slice
from lib.cpp.cpu.analysis import bic
from lib.py.helpers import block_info, load_block, commandline_args
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    sample, scale, m, scheme, threshold_prob, threshold_distance = commandline_args({
        "sample" : "<required>",
        "scale" : 1,
        "material": 0,
        "scheme": "gauss+edt",
        "threshold_prob" : 0, # For whether voxel is blood or not
        "threshold_distance" : 100
    })

    probs_dir = f'{binary_root}/segmented/{scheme}'
    soft_path = f'{probs_dir}/P{m}/{scale}x/{sample}.uint16'
    bone_path = f'{probs_dir}/P{np.abs(m-1)}/{scale}x/{sample}.uint16'
    output_dir = f"{binary_root}/fields/soft-tissue-edt/{scale}x"

    bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5')
    shape = np.array(bi["dimensions"][:3])
    shape //= scale

    os.makedirs(output_dir, exist_ok=True)

    soft = np.memmap(soft_path, dtype=np.uint16, mode='r').reshape(shape)

    soft_threshed = ~(soft > threshold_prob)
    del soft

    hyperthreading = True
    n_cores = mp.cpu_count() // (2 if hyperthreading else 1) # Only count physical cores
    edted = edt.edt(soft_threshed, parallel=n_cores)
    del soft_threshed

    disted = edted < threshold_distance
    del edted

    bone = np.memmap(bone_path, dtype=np.uint16, mode='r').reshape(shape)

    bone_threshed = bone > threshold_prob
    del bone

    bone_count = np.sum(bone_threshed)
    dist_count = np.sum(bone_threshed & disted)

    print (f"Bone count: {bone_count}, Distance count: {dist_count}, Ratio: {dist_count/bone_count}")
