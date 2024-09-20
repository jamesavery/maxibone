#! /usr/bin/python3
'''
This script finds the osteocytes in the segmented soft tissue mask.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')

from config.constants import *
from config.paths import hdf5_root
import h5py
from lib.cpp.cpu.general import bincount, where_in
from lib.cpp.cpu.geometry import center_of_masses, inertia_matrices, outside_ellipsoid
from lib.py.commandline_args import default_parser
from lib.py.helpers import plot_middle_planes, update_hdf5
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import scipy.ndimage as ndi
import PIL.Image

if __name__ == '__main__':
    args = default_parser(__doc__).parse_args()

    # Define and create directories
    plot_dir = f"{hdf5_root}/processed/osteocyt_mask/{args.sample_scale}x"
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
    mask_dir = f'{hdf5_root}/masks/{args.sample_scale}x'
    sample_path = f'{mask_dir}/{args.sample}.h5'

    # Load the blood mask
    with h5py.File(f"{sample_path}", 'r') as f:
        blood_mask = f['blood']['mask'][:]
        Nz, Ny, Nx = blood_mask.shape
        voxel_size = f['blood'].attrs['voxel_size']
        voxel_volume = voxel_size**3

    if voxel_volume > osteocyte_Vmax:
        raise ValueError(f"Voxel volume for scale {args.sample_scale} is {voxel_volume}, which is larger than the maximum osteocyte volume {osteocyte_Vmax}. Please run on a finer scale.")

    # Load and threshold the soft tissue mask
    voxels = np.memmap(f'{hdf5_root}/binary/segmented/gauss+edt/P0/{args.sample_scale}x/{args.sample}.uint16', dtype='uint16', mode='r', shape=(Nz, Ny, Nx))
    as_mask = voxels > 0
    as_mask *= ~blood_mask

    # Label the potential osteocytes
    hole_id, num_holes = ndi.label(as_mask, output=np.uint64) # TODO out-of-core
    if args.verbose > 0:
        print (f"Found {num_holes} potential osteocytes")

    # Compute the volumes and sort out unrealistic osteocytes
    volumes = np.zeros(num_holes+1, dtype=np.uint64)
    bincount(hole_id, volumes)
    volumes = volumes * voxel_volume
    small_unknown = volumes < osteocyte_Vmin
    large_unknown = volumes > osteocyte_Vmax
    osteocyte_sized = (volumes >= osteocyte_Vmin) & (volumes <= osteocyte_Vmax)
    if args.verbose > 0:
        print (f"Found {np.sum(small_unknown)} small and {np.sum(large_unknown)} large osteocytes")
        print (f"Found {np.sum(osteocyte_sized)} potential osteocytes")

    # Compute ellipsoid fits
    cms = np.zeros((num_holes+1, 3), dtype=np.float32)
    center_of_masses(hole_id, cms)
    ims = np.zeros((num_holes+1, 3, 3), dtype=np.float32)
    inertia_matrices(hole_id, cms, ims)
    principal_lambdas = np.linalg.eigvals(ims)
    abc = 1 / np.sqrt(principal_lambdas)
    a, b, c = abc.T
    weirdly_long = (a / c) > 3
    nans = np.isnan(a) | np.isnan(b) | np.isnan(c)
    weirdly_long |= nans
    if args.verbose > 0:
        print (f"Found {np.sum(weirdly_long)} weirdly long osteocytes")

    # Test that the osteocytes are not too different from the best ellipsoid
    ellipsoid_errors = np.zeros(num_holes+1, dtype=np.uint64)
    ellipsoid_volumes = (4/3) * np.pi * a * b * c
    outside_ellipsoid(hole_id, cms, abc, ellipsoid_errors)
    ellipsoid_error_threshold = .3 * 1e9
    weirdly_shaped = (ellipsoid_errors / ellipsoid_volumes) > ellipsoid_error_threshold
    if args.verbose > 0:
        print (f"Found {np.sum(weirdly_shaped)} weirdly shaped osteocytes")
        print (f'Plotting histogram of ellipsoid errors to {plot_dir}/')
        errors = ellipsoid_errors[1:] / ellipsoid_volumes[1:]
        errors = errors[~np.isnan(errors) & ~np.isinf(errors) & ~weirdly_long[1:] & osteocyte_sized[1:]]
        print (f"Mean ellipsoid error: {np.mean(errors)}")
        print (f'Std ellipsoid error: {np.std(errors)}')
        print (f'min/max ellipsoid error: {np.min(errors)}/{np.max(errors)}')
        plt.hist(errors, bins=100, log=True)
        plt.savefig(f'{plot_dir}/{args.sample}_ellipsoid_errors.png')
        plt.clf()

    # Final osteocyte segmentation
    osteocyte_segments = np.argwhere(osteocyte_sized & (~weirdly_long) & (~weirdly_shaped)).flatten().astype(np.uint64)
    osteocyte_mask = hole_id.copy()
    if args.verbose > 0:
        print (f"Found {len(osteocyte_segments)} osteocytes")

    where_in(osteocyte_mask, osteocyte_segments)

    if args.verbose > 0:
        # Plot the debug images
        red = [255,0,0]
        yellow = [255,255,0]
        green = [0,255,0]

        print(f"Saving osteocyt mask to {sample_path}")
        print(f'Plotting osteocyt mask to {plot_dir}/')
        hnz, hny, hnx = Nz//2, Ny//2, Nx//2
        plot_middle_planes(osteocyte_mask, plot_dir, f'{args.sample}_osteocyt_mask', verbose=args.verbose)

        yx = np.zeros((Ny, Nx, 3), dtype=np.uint8)
        yx[as_mask[hnz] > 0] = yellow
        yx[blood_mask[hnz] > 0] = red
        yx[osteocyte_mask[hnz] > 0] = green
        PIL.Image.fromarray(yx).save(f'{plot_dir}/{args.sample}_yx_overlay.png')

        zx = np.zeros((Nz, Nx, 3), dtype=np.uint8)
        zx[as_mask[:,hny] > 0] = yellow
        zx[blood_mask[:,hny] > 0] = red
        zx[osteocyte_mask[:,hny] > 0] = green
        PIL.Image.fromarray(zx).save(f'{plot_dir}/{args.sample}_zx_overlay.png')

        zy = np.zeros((Nz, Ny, 3), dtype=np.uint8)
        zy[as_mask[:,:,hnx] > 0] = yellow
        zy[blood_mask[:,:,hnx] > 0] = red
        zy[osteocyte_mask[:,:,hnx] > 0] = green
        PIL.Image.fromarray(zy).save(f'{plot_dir}/{args.sample}_zy_overlay.png')

    # Save the mask
    update_hdf5(sample_path,
                group_name='osteocyt',
                datasets={'mask': osteocyte_mask},
                attributes={
                    'scale': args.sample_scale,
                    'voxel_size': voxel_size,
                    'sample': args.sample,
                    'name': 'osteocyt mask'
                })