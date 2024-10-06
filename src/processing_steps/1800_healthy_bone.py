#! /usr/bin/python3
'''
This script finds the amount of "healthy bone".
Healthy bone is defined as the amount of bone that is within a certain distance of soft tissue.
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
from config.paths import hdf5_root, binary_root, get_plotting_dir
from lib.py.commandline_args import add_volume, default_parser
from lib.py.helpers import bitpack_decode, bitpack_encode, chunk_info, close_3d, dilate_3d, open_3d
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    argparser = default_parser(__doc__)
    argparser = add_volume(argparser, 'field', 2, 'gauss+edt')
    argparser.add_argument('-d', '--distance', action='store', type=int, default=20,
        help='The distance threshold (in micrometers) for the distance field. Default is 20.')
    argparser.add_argument('-t', '--threshold', action='store', type=int, default=0,
        help='The probability threshold for whether a voxel is the given material. Default is 0.')
    argparser.add_argument('-m', '--material', action='store', type=int, default=0,
        help='The material to segment. Default is 0, which should be soft tissue.')
    args = argparser.parse_args()

    probs_dir = f'{binary_root}/segmented/{args.field}'
    soft_path = f'{probs_dir}/P{args.material}/{args.sample_scale}x/{args.sample}.uint16'
    bone_path = f'{probs_dir}/P{np.abs(args.material-1)}/{args.sample_scale}x/{args.sample}.uint16'
    output_dir = f"{binary_root}/fields/healthy_bone/{args.sample_scale}x"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    plotting_dir = get_plotting_dir(args.sample, args.sample_scale)
    if args.plotting:
        pathlib.Path(plotting_dir).mkdir(parents=True, exist_ok=True)

    bi = chunk_info(f'{hdf5_root}/hdf5-byte/msb/{args.sample}.h5', args.sample_scale, verbose=args.verbose)
    voxel_size = bi["voxel_size"] * args.sample_scale
    shape = np.array(bi["dimensions"][:3])
    shape //= args.sample_scale
    nz, ny, nx = shape

    if args.verbose >= 1:
        print (f'Processing {args.sample} with threshold {args.threshold} and distance {args.distance} at scale {args.sample_scale}x')
        print (f'Voxel size: {voxel_size} µm')

    os.makedirs(output_dir, exist_ok=True)

    if args.verbose >= 1: print (f'Loading the soft tissue probabilities from {soft_path}')
    soft = np.memmap(soft_path, dtype=np.uint16, mode='r').reshape(shape)

    if args.verbose >= 1: print (f'Thresholding the probabilities at {args.threshold}')
    soft_threshed = (soft > args.threshold)
    del soft

    soft_bp = bitpack_encode(soft_threshed, verbose=args.verbose)
    del soft_threshed

    # Close, open, then dilate. The sizes are in micrometers - adjust accordingly
    closing = 5
    opening = 5

    closing_voxels = int(closing / voxel_size)
    opening_voxels = int(opening / voxel_size)
    distance_voxels = int(args.distance / voxel_size)

    if args.verbose >= 1:
        print (f'Closing: {closing_voxels}, Opening: {opening_voxels}, Distance: {distance_voxels}')

    # Close
    closed = close_3d(soft_bp, closing_voxels, args.verbose)

    # Open
    opened = open_3d(closed, opening_voxels, args.verbose)
    del closed

    # Dilate
    soft_bp = dilate_3d(opened, distance_voxels, args.verbose)
    del opened

    if args.plotting:
        if args.verbose >= 1: print (f'Writing soft tissue debug plane images to {plotting_dir}')
        soft = bitpack_decode(soft_bp, verbose=args.verbose)
        names = ['yx', 'zx', 'zy']
        planes = [soft[nz//2,:,:], soft[:,ny//2,:], soft[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.figure(figsize=(10,10))
            plt.imshow(plane)
            plt.savefig(f'{plotting_dir}/soft_{name}.pdf', bbox_inches='tight')
            plt.close()
        del soft

    if args.verbose >= 1: print (f'Loading the bone probabilities from {bone_path}')
    bone = np.memmap(bone_path, dtype=np.uint16, mode='r').reshape(shape)

    if args.verbose >= 1: print (f'Thresholding the probabilities at {args.threshold}')
    bone_threshed = bone > args.threshold
    del bone

    if args.plotting:
        if args.verbose >= 1: print (f'Writing bone debug plane images to {plotting_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [bone_threshed[nz//2,:,:], bone_threshed[:,ny//2,:], bone_threshed[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.figure(figsize=(10,10))
            plt.imshow(plane)
            plt.savefig(f'{plotting_dir}/bone_{name}.pdf', bbox_inches='tight')
            plt.close()

    bone_bp = bitpack_encode(bone_threshed, verbose=args.verbose)
    bone_bp_opened = open_3d(bone_bp, opening_voxels, args.verbose)
    bone_opened = bitpack_decode(bone_bp_opened, verbose=args.verbose)
    del bone_bp

    if args.plotting:
        if args.verbose >= 1: print (f'Writing opened bone debug plane images to {plotting_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [bone_opened[nz//2,:,:], bone_opened[:,ny//2,:], bone_opened[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.figure(figsize=(10,10))
            plt.imshow(plane)
            plt.savefig(f'{plotting_dir}/bone_opened_{name}.pdf', bbox_inches='tight')
            plt.close()

    if args.verbose >= 1: print (f'Computing the distance field')
    disted_bp = soft_bp & bone_bp_opened
    disted = bitpack_decode(disted_bp, verbose=args.verbose)

    if args.plotting:
        if args.verbose >= 1: print (f'Writing distance debug plane images to {plotting_dir}')
        names = ['yx', 'zx', 'zy']
        planes = [disted[nz//2,:,:], disted[:,ny//2,:], disted[:,:,nx//2]]
        for name, plane in zip(names, planes):
            plt.figure(figsize=(10,10))
            plt.imshow(plane)
            plt.savefig(f'{plotting_dir}/dist_{name}.pdf', bbox_inches='tight')
            plt.close()

    bone_count = np.sum(bone_opened)
    dist_count = np.sum(disted)

    print (f"Bone count: {bone_count}, Distance count: {dist_count}, Ratio: {dist_count/bone_count}")

    if args.verbose >= 1: print (f'Saving the distance field to {output_dir}/{args.sample}.npy')
    np.save(f'{output_dir}/{args.sample}.npy', disted)
