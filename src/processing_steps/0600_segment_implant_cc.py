#! /usr/bin/python3
'''
Segment the implant using coarse thresholding followed by connected components.
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
import h5py
from lib.cpp.cpu.io import load_slice
from lib.py.commandline_args import default_parser
from lib.py.helpers import largest_cc_of, update_hdf5_mask, plot_middle_planes
import numpy as np

if __name__ == "__main__":
    args = default_parser(__doc__).parse_args()

    # Load metadata. TODO: Clean up, make automatic function.
    meta_filename = f"{hdf5_root}/hdf5-byte/msb/{args.sample}.h5"
    h5meta     = h5py.File(meta_filename,'r')
    vm_shifts  = h5meta['volume_matching_shifts'][:]
    full_Nz, Ny, Nx = h5meta['voxels'].shape               # Full image resolution
    Nz         = full_Nz - np.sum(vm_shifts)               # Full volume matched image resolution
    nz,ny,nx   = np.array([Nz,Ny,Nx]) // args.sample_scale # Volume matched image resolution at chosen scale

    intermediate_folder = f"/tmp/maxibone/labels_implant/{args.sample_scale}x/"
    pathlib.Path(f"{intermediate_folder}").mkdir(parents=True, exist_ok=True)

    if args.plotting:
        plotting_dir = get_plotting_dir(args.sample, args.sample_scale)
        pathlib.Path(plotting_dir).mkdir(parents=True, exist_ok=True)

    voxel_size  = h5meta['voxels'].attrs['voxelsize'] * args.sample_scale
    global_vmin = np.min(h5meta['subvolume_range'][:,0])
    global_vmax = np.max(h5meta['subvolume_range'][:,1])
    values      = np.linspace(global_vmin,global_vmax,2**16)
    implant_threshold_u16 = np.argmin(np.abs(values-implant_threshold))
    if 'novisim' in args.sample:
        implant_threshold_u16 = implant_threshold_u16_novisim

    if args.verbose >= 2: print(f"""
        Reading metadata from {meta_filename}.
        volume_matching_shifts = {vm_shifts}
        Implant threshold {implant_threshold} -> {implant_threshold_u16} as uint16
        full_Nz,Ny,Nx    = {full_Nz,Ny,Nx}
        Nz               = {Nz}
        nz,ny,nx         = {nz,ny,nx}
        voxel_size       = {voxel_size}
        vmin,vmax        = {global_vmin,global_vmax}
    """)
    h5meta.close()

    if args.verbose >= 1: print(f"Reading full volume {args.sample} at {args.sample_scale}x")
    voxels = np.empty((nz,ny,nx), dtype=np.uint16)
    load_slice(voxels, f"{binary_root}/voxels/{args.sample_scale}x/{args.sample}.uint16", (0,0,0), (nz,ny,nx))
    if args.plotting:
        plot_middle_planes(voxels, plotting_dir, "voxels")
    noisy_implant = (voxels > implant_threshold_u16)
    if args.plotting:
        plot_middle_planes(noisy_implant, plotting_dir, "noisy_implant")

    del voxels

    implant_mask = largest_cc_of(args.sample, args.sample_scale, noisy_implant, 'implant', args.plotting, plotting_dir, args.verbose)

    if args.plotting:
        if args.verbose >= 1: print (f"Plotting middle planes to {plotting_dir}")
        plot_middle_planes(implant_mask, plotting_dir, args.sample, verbose=args.verbose)

    output_dir = f"{hdf5_root}/masks/{args.sample_scale}x/"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.verbose >= 1: print(f"Writing largest connected component to {output_dir}/{args.sample}.h5")

    update_hdf5_mask(f"{output_dir}/{args.sample}.h5",
        group_name = "implant",
        datasets = { 'mask': implant_mask },
        attributes = {
            'scale': args.sample_scale,
            'voxel_size': voxel_size,
            'sample': args.sample,
            'name': "implant_mask"
        })
