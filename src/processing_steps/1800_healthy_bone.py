import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')

from config.constants import *
from config.paths import hdf5_root, binary_root
from lib.py.helpers import bitpack_decode, bitpack_encode, chunk_info, close_3d, commandline_args, dilate_3d, open_3d
import matplotlib.pyplot as plt
import numpy as np
import os

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

    bi = chunk_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5')
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

    soft_bp = bitpack_encode(soft_threshed)
    del soft_threshed

    # Close, open, then dilate. The sizes are in micrometers
    # TODO adjust accordingly
    closing = 5
    opening = 5

    closing_voxels = int(closing / voxel_size)
    opening_voxels = int(opening / voxel_size)
    distance_voxels = int(threshold_distance / voxel_size)

    if verbose >= 1:
        print (f'Closing: {closing_voxels}, Opening: {opening_voxels}, Distance: {distance_voxels}')

    # Close
    closed = close_3d(soft_bp, closing_voxels)

    # Open
    opened = open_3d(closed, opening_voxels)
    del closed

    # Dilate
    soft_bp = dilate_3d(opened, distance_voxels)
    del opened

    if verbose >= 1:
        print (f'Writing soft tissue debug plane images to {image_output_dir}')
        soft = bitpack_decode(soft_bp)
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

    bone_bp = bitpack_encode(bone_threshed)
    bone_bp_opened = open_3d(bone_bp, opening_voxels)
    bone_opened = bitpack_decode(bone_bp_opened)
    del bone_bp

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
    disted = bitpack_decode(disted_bp)

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
