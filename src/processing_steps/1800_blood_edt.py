import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, hdf5_root_fast, binary_root
import edt
import h5py
from lib.cpp.cpu.analysis import bic
from lib.cpp.gpu.bitpacking import encode as bp_encode, decode as bp_decode
from lib.cpp.gpu.morphology import erode_3d_sphere_bitpacked as erode, dilate_3d_sphere_bitpacked as dilate
from lib.cpp.cpu_seq.io import load_slice, write_slice
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

    soft_bp = np.empty((nz,ny,nx//32),dtype=np.uint32)
    bp_encode(soft_threshed.astype(np.uint8), soft_bp)
    del soft_threshed

    soft_tmp = np.empty_like(soft_bp)
    # Open, close, then dilate
    dilate(soft_bp, 1, soft_tmp)
    erode(soft_tmp, 3, soft_bp)
    dilate(soft_bp, 5, soft_tmp)
    soft_bp[:] = soft_tmp[:]
    del soft_tmp

    if verbose >= 1:
        print (f'Writing soft tissue debug plane images to {image_output_dir}')
        soft = np.empty((nz,ny,nx),dtype=np.uint8)
        bp_decode(soft_bp, soft)
        names = ['yx', 'zx', 'zy']
        planes = [soft[nz//2,:,:], soft[:,ny//2,:], soft[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.figure(figsize=(10,10))
            plt.imshow(plane)
            plt.savefig(f'{image_output_dir}/{sample}_soft_{name}.png', bbox_inches='tight')
            plt.clf()
        del soft

    bone = np.memmap(bone_path, dtype=np.uint16, mode='r').reshape(shape)

    bone_threshed = bone > threshold_prob
    del bone

    if verbose >= 1:
        print (f'Writing bone debug plane images to {image_output_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [bone_threshed[nz//2,:,:], bone_threshed[:,ny//2,:], bone_threshed[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.figure(figsize=(10,10))
            plt.imshow(plane)
            plt.savefig(f'{image_output_dir}/{sample}_bone_{name}.png', bbox_inches='tight')
            plt.clf()

    bone_bp = np.empty((nz,ny,nx//32),dtype=np.uint32)
    bp_encode(bone_threshed.astype(np.uint8), bone_bp)

    disted_bp = soft_bp & bone_bp
    disted = np.empty((nz,ny,nx),dtype=np.uint8)
    bp_decode(disted_bp, disted)

    if verbose >= 1:
        print (f'Writing distance debug plane images to {image_output_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [disted[nz//2,:,:], disted[:,ny//2,:], disted[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.figure(figsize=(10,10))
            plt.imshow(plane)
            plt.savefig(f'{image_output_dir}/{sample}_dist_{name}.png', bbox_inches='tight')
            plt.clf()

    bone_count = np.sum(bone_threshed)
    dist_count = np.sum(disted)

    print (f"Bone count: {bone_count}, Distance count: {dist_count}, Ratio: {dist_count/bone_count}")
