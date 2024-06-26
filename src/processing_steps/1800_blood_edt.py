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
    sample, scale, m, scheme, threshold_prob, threshold_distance, verbose = commandline_args({
        "sample" : "<required>",
        "scale" : 1,
        "material": 0,
        "scheme": "gauss+edt",
        "threshold_prob" : 0, # For whether voxel is blood or not
        "threshold_distance" : 5,
        'verbose': 1
    })

    probs_dir = f'{binary_root}/segmented/{scheme}'
    soft_path = f'{probs_dir}/P{m}/{scale}x/{sample}.uint16'
    bone_path = f'{probs_dir}/P{np.abs(m-1)}/{scale}x/{sample}.uint16'
    output_dir = f"{binary_root}/fields/soft-tissue-edt/{scale}x"
    image_output_dir = f"{hdf5_root}/processed/soft-tissue-edt/{scale}x/{sample}"

    if verbose >= 1: os.makedirs(image_output_dir, exist_ok=True)

    bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5')
    shape = np.array(bi["dimensions"][:3])
    shape //= scale
    nz, ny, nx = shape

    os.makedirs(output_dir, exist_ok=True)

    soft = np.memmap(soft_path, dtype=np.uint16, mode='r').reshape(shape)

    soft_threshed = (soft > threshold_prob)
    del soft

    if verbose >= 1:
        print (f'Writing soft tissue debug plane images to {image_output_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [soft_threshed[nz//2,:,:], soft_threshed[:,ny//2,:], soft_threshed[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.imshow(plane)
            plt.savefig(f'{image_output_dir}/{sample}_soft_{name}.png')
            plt.clf()

    hyperthreading = True
    n_cores = mp.cpu_count() // (2 if hyperthreading else 1) # Only count physical cores
    edted = edt.edt(soft_threshed, parallel=n_cores)
    del soft_threshed

    if verbose >= 1:
        print (f'Writing EDT debug plane images to {image_output_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [edted[nz//2,:,:], edted[:,ny//2,:], edted[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.imshow(plane)
            plt.colorbar()
            plt.savefig(f'{image_output_dir}/{sample}_edt_{name}.png')
            plt.clf()

    disted = edted < threshold_distance
    del edted

    if verbose >= 1:
        print (f'Writing thresholded EDT debug plane images to {image_output_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [disted[nz//2,:,:], disted[:,ny//2,:], disted[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.imshow(plane)
            plt.savefig(f'{image_output_dir}/{sample}_dist_{name}.png')
            plt.clf()

    bone = np.memmap(bone_path, dtype=np.uint16, mode='r').reshape(shape)

    bone_threshed = bone > threshold_prob
    del bone

    if verbose >= 1:
        print (f'Writing bone debug plane images to {image_output_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [bone_threshed[nz//2,:,:], bone_threshed[:,ny//2,:], bone_threshed[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.imshow(plane)
            plt.savefig(f'{image_output_dir}/{sample}_bone_{name}.png')
            plt.clf()

    bone_count = np.sum(bone_threshed)
    dist_count = np.sum(bone_threshed & disted)

    print (f"Bone count: {bone_count}, Distance count: {dist_count}, Ratio: {dist_count/bone_count}")
