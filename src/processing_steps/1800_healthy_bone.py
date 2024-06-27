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

# close = dilate then erode
# open = erode then dilate
def morph(image, r, fa, fb):
    if r == 0:
        return image
    I1 = image.copy().astype(image.dtype)
    I2 = np.empty(image.shape, dtype=image.dtype)
    rmin = 15
    rmins = r // rmin
    rrest = r % rmin
    for _ in range(rmins):
        fa(I1, rmin, I2)
        I1, I2 = I2, I1
    if rrest > 0:
        fa(I1, rrest, I2)
        I1, I2 = I2, I1

    for i in range(rmins):
        fb(I1, rmin, I2)
        I1, I2 = I2, I1
    if rrest > 0:
        fb(I1, rrest, I2)
        I1, I2 = I2, I1

    return I1

def close(image, r):
    return morph(image, r, dilate, erode)

def open(image, r):
    return morph(image, r, erode, dilate)

def encode_ooc(src, dst):
    bs = 32
    for i in range(src.shape[0] // bs):
        start, end = i*bs, (i+1)*bs
        bp_encode(src[start:end], dst[start:end])

def decode_ooc(src, dst):
    bs = 32
    for i in range(src.shape[0] // bs):
        start, end = i*bs, (i+1)*bs
        bp_decode(src[start:end], dst[start:end])

if __name__ == '__main__':
    sample, scale, m, scheme, threshold_prob, threshold_distance, verbose = commandline_args({
        "sample" : "<required>",
        "scale" : 1,
        "material": 0,
        "scheme": "gauss+edt",
        "threshold_prob" : 0, # For whether voxel is blood or not
        "threshold_distance" : 20, # In micrometers (µm)
        'verbose': 1
    })

    probs_dir = f'{binary_root}/segmented/{scheme}'
    soft_path = f'{probs_dir}/P{m}/{scale}x/{sample}.uint16'
    bone_path = f'{probs_dir}/P{np.abs(m-1)}/{scale}x/{sample}.uint16'
    output_dir = f"{binary_root}/fields/healthy_bone/{scale}x"
    image_output_dir = f"{hdf5_root}/processed/healthy_bone/{scale}x/{sample}"

    if verbose >= 1: os.makedirs(image_output_dir, exist_ok=True)

    bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5')
    voxel_size = bi["voxel_size"] * scale
    shape = np.array(bi["dimensions"][:3])
    shape //= scale
    nz, ny, nx = shape

    if verbose >= 1:
        print (f'Processing {sample} with threshold {threshold_prob} and distance {threshold_distance} at scale {scale}x')
        print (f'Voxel size: {voxel_size} µm')

    os.makedirs(output_dir, exist_ok=True)

    soft = np.memmap(soft_path, dtype=np.uint16, mode='r').reshape(shape)

    soft_threshed = (soft > threshold_prob)
    del soft

    soft_bp = np.empty((nz,ny,nx//32),dtype=np.uint32)
    encode_ooc(soft_threshed.astype(np.uint8), soft_bp)
    del soft_threshed

    # Close, open, then dilate. The sizes are in micrometers
    closing = 10
    opening = 10

    closing_voxels = int(closing / voxel_size)
    opening_voxels = int(opening / voxel_size)
    distance_voxels = int(threshold_distance / voxel_size)

    if verbose >= 1:
        print (f'Closing: {closing_voxels}, Opening: {opening_voxels}, Distance: {distance_voxels}')

    # Close
    closed = close(soft_bp, closing_voxels)

    # Open
    opened = open(closed, opening_voxels)
    del closed

    # Dilate
    dilate(opened, distance_voxels, soft_bp)
    del opened

    if verbose >= 1:
        print (f'Writing soft tissue debug plane images to {image_output_dir}')
        soft = np.empty((nz,ny,nx),dtype=np.uint8)
        decode_ooc(soft_bp, soft)
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
    encode_ooc(bone_threshed.astype(np.uint8), bone_bp)

    bone_bp_opened = open(bone_bp, opening_voxels)
    bone_opened = np.empty((nz,ny,nx),dtype=np.uint8)
    decode_ooc(bone_bp_opened, bone_opened)

    if verbose >= 1:
        print (f'Writing opened bone debug plane images to {image_output_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [bone_opened[nz//2,:,:], bone_opened[:,ny//2,:], bone_opened[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.figure(figsize=(10,10))
            plt.imshow(plane)
            plt.savefig(f'{image_output_dir}/{sample}_bone_opened_{name}.png', bbox_inches='tight')
            plt.clf()

    disted_bp = soft_bp & bone_bp_opened
    disted = np.empty((nz,ny,nx),dtype=np.uint8)
    decode_ooc(disted_bp, disted)

    if verbose >= 1:
        print (f'Writing distance debug plane images to {image_output_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [disted[nz//2,:,:], disted[:,ny//2,:], disted[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.figure(figsize=(10,10))
            plt.imshow(plane)
            plt.savefig(f'{image_output_dir}/{sample}_dist_{name}.png', bbox_inches='tight')
            plt.clf()

    bone_count = np.sum(bone_opened)
    dist_count = np.sum(disted)

    print (f"Bone count: {bone_count}, Distance count: {dist_count}, Ratio: {dist_count/bone_count}")

    if verbose >= 1: print (f'Saving the distance field to {output_dir}/{sample}.npy')
    np.save(f'{output_dir}/{sample}.npy', disted)
