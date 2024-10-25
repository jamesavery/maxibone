#! /usr/bin/python3
'''
This script computes the bone region of the implant.

The bone region is defined as the region of the implant that is not the implant shell, and is not the solid implant.
Or in other words, the bone region covers bone and soft tissue.
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
import datetime
import h5py
from lib.cpp.cpu.geometry import compute_front_back_masks
from lib.py.commandline_args import default_parser
from lib.py.helpers import bitpack_decode, bitpack_encode, close_3d, dilate_3d, largest_cc_of, open_3d, plot_middle_planes, update_hdf5_mask
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.signal as signal

if __name__ == "__main__":
    args = default_parser(__doc__, default_scale=4).parse_args()

    if args.verbose >= 1: print(f"Loading {args.sample_scale}x implant mask from {hdf5_root}/masks/{args.sample_scale}x/{args.sample}.h5")
    implant_file = h5py.File(f"{hdf5_root}/masks/{args.sample_scale}x/{args.sample}.h5",'r')
    implant      = implant_file["implant/mask"][:].astype(np.uint8)
    voxel_size   = implant_file["implant"].attrs["voxel_size"]
    novisimflag = implant_file.attrs["novisim"]
    implant_file.close()

    nz, ny, nx = implant.shape

    if args.plotting:
        plotting_dir = get_plotting_dir(args.sample, args.sample_scale)
        if args.verbose >= 1: print(f"Storing all debug-images to {plotting_dir}")
        pathlib.Path(plotting_dir).mkdir(parents=True, exist_ok=True)

    if args.verbose >= 1: print(f"Loading {args.sample_scale}x voxels from {binary_root}/voxels/{args.sample_scale}x/{args.sample}.uint16")
    voxels  = np.fromfile(f"{binary_root}/voxels/{args.sample_scale}x/{args.sample}.uint16",dtype=np.uint16).reshape(implant.shape)
    if args.plotting: plot_middle_planes(voxels, plotting_dir, 'voxels', verbose=args.verbose)

    if args.verbose >= 1: print (f'Loading FoR values from {hdf5_root}/hdf5-byte/msb/{args.sample}.h5')
    with h5py.File(f"{hdf5_root}/hdf5-byte/msb/{args.sample}.h5",'r') as f:
        UVWp = f['implant-FoR/UVWp'][:]
        cp = f['implant-FoR/center_of_cylinder_UVW'][:]
        cm = (f['implant-FoR/center_of_mass'][:]) / voxel_size
        E = f['implant-FoR/E'][:]

    if args.verbose >= 1: print(f"Computing front/back/implant_shell/solid_implant masks")
    front_mask = np.empty_like(implant, dtype=np.uint8)
    back_mask = np.empty_like(implant, dtype=np.uint8)
    implant_shell_mask = np.empty_like(implant, dtype=np.uint8)
    solid_implant = np.empty_like(implant, dtype=np.uint8)
    if args.verbose >= 1: start = datetime.datetime.now()
    compute_front_back_masks(implant, voxel_size, E, cm, cp, UVWp, front_mask, back_mask, implant_shell_mask, solid_implant)
    if args.verbose >= 1: end = datetime.datetime.now()
    if args.verbose >= 1: print (f'Computing front/back/implant_shell/solid_implant masks took {end-start}')

    front_mask = largest_cc_of(args.sample, args.sample_scale, front_mask, 'front', args.plotting, plotting_dir, args.verbose)
    front_part = voxels * front_mask

    output_dir = f"{hdf5_root}/masks/{args.sample_scale}x"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.verbose >= 1: print(f"Saving implant_solid mask to {output_dir}/{args.sample}.h5")
    update_hdf5_mask(f"{output_dir}/{args.sample}.h5",
                     group_name="implant_solid",
                     datasets={"mask": solid_implant},
                     attributes={"sample": args.sample, "scale": args.sample_scale, "voxel_size": voxel_size})
    if args.plotting: plot_middle_planes(solid_implant, plotting_dir, 'implant-solid-sanity')

    if args.verbose >= 1: print(f"Saving implant_shell mask to {output_dir}/{args.sample}.h5")
    update_hdf5_mask(f"{output_dir}/{args.sample}.h5",
                     group_name="implant_shell",
                     datasets={"mask":implant_shell_mask},
                     attributes={"sample":args.sample,"scale":args.sample_scale,"voxel_size":voxel_size})
    if args.plotting: plot_middle_planes(implant_shell_mask, plotting_dir, 'implant-shell-sanity', verbose=args.verbose)
    del implant_shell_mask

    if args.verbose >= 1: print(f"Saving cut_cylinder_air mask to {output_dir}/{args.sample}.h5")
    update_hdf5_mask(f"{output_dir}/{args.sample}.h5",
                     group_name="cut_cylinder_air",
                     datasets={"mask":back_mask},
                     attributes={"sample":args.sample,"scale":args.sample_scale,"voxel_size":voxel_size})
    if args.plotting: plot_middle_planes(back_mask, plotting_dir, 'implant-back-sanity', verbose=args.verbose)
    del back_mask

    if args.verbose >= 1: print(f"Saving cut_cylinder_bone mask to {output_dir}/{args.sample}.h5")
    update_hdf5_mask(f"{output_dir}/{args.sample}.h5",
                     group_name="cut_cylinder_bone",
                     datasets={"mask":front_mask},
                     attributes={"sample":args.sample, "scale":args.sample_scale, "voxel_size":voxel_size})
    if args.plotting: plot_middle_planes(front_mask, plotting_dir, 'implant-front-sanity', verbose=args.verbose)
    del front_mask

    front_part_implanted = front_part.copy()
    front_part_implanted[implant == 1] = 0
    fpmin = front_part_implanted
    fpmin[fpmin==0] = 65535
    vmin = fpmin.min()
    fpmin[fpmin==65535] = vmin
    vmax = fpmin.max()
    del fpmin, front_part_implanted

    if args.verbose >= 1: print(f"Computing bone region")
    hist, bins = np.histogram(front_part, 2048, range=(vmin,vmax))
    hist[0] = 0
    hist_raw = hist.copy()
    hist = gaussian_filter1d(hist, 3)
    peaks, info = signal.find_peaks(hist, height=0.1*hist.max()) # Although, wouldn't the later argsort filter the smaller peaks away anyways?

    if args.plotting:
        plt.figure(figsize=(20,10))
        plt.plot(bins[1:], hist_raw)
        plt.plot(bins[1:], hist)
        plt.savefig(f'{plotting_dir}/bone_histogram.pdf', bbox_inches='tight')
        plt.close()
        if args.verbose >= 1: print (f'peaks: {peaks}')

    two_largest_peaks = peaks[np.argsort(info['peak_heights'])[::-1][:2]]
    p1, p2 = sorted(two_largest_peaks)
    midpoint = bins[np.argmin(hist[p1:p2]) + p1]
    if args.verbose >= 1: print(f"p1, p2 = ({p1,bins[p1]}), ({p2,bins[p2]}); midpoint = {midpoint}")

    bone_mask1 = front_part > midpoint
    del front_part
    if args.plotting: plot_middle_planes(bone_mask1, plotting_dir, 'implant-bone1-sanity', verbose=args.verbose)

    if novisimflag:
        closing_diameter, opening_diameter, implant_dilate_diameter = 400, 300, 15 # micrometers
    else:
        closing_diameter, opening_diameter, implant_dilate_diameter = 400, 300, 5 # micrometers
    closing_voxels = 2 * int(round(closing_diameter / (2 * voxel_size))) + 1 # Scale & ensure odd length
    opening_voxels = 2 * int(round(opening_diameter / (2 * voxel_size))) + 1 # Scale & ensure odd length
    implant_dilate_voxels = 2 * int(round(implant_dilate_diameter / (2 * voxel_size))) + 1 # Scale & ensure odd length
    bitpacked = nx % 32 == 0
    print_progress = 2 if args.verbose >= 1 else 0
    if bitpacked:
        bone_region_tmp = bitpack_encode(bone_mask1, verbose=args.verbose)
    else:
        bone_region_tmp = bone_mask1.astype(np.uint8)
    del bone_mask1

    if args.verbose >= 1: print (f"Closing with sphere of diameter {closing_diameter} micrometers, {closing_voxels} voxels.")
    bone_region_tmp = close_3d(bone_region_tmp, closing_voxels // 2, verbose=print_progress)

    if args.verbose >= 1: print (f"Opening with sphere of diameter {opening_diameter} micrometers, {opening_voxels} voxels.")
    bone_region_tmp = open_3d(bone_region_tmp, opening_voxels // 2, verbose=print_progress)

    if args.verbose >= 1: print (f"Dilating and removing implant with {implant_dilate_diameter} micrometers, {implant_dilate_voxels} voxels.")
    if bitpacked:
        packed_implant = bitpack_encode(solid_implant, verbose=args.verbose)
    else:
        packed_implant = solid_implant
    del solid_implant
    dilated_implant = dilate_3d(packed_implant, implant_dilate_voxels, verbose=print_progress)
    bone_region_tmp &= ~dilated_implant

    if bitpacked:
        bone_region_mask = bitpack_decode(bone_region_tmp, verbose=args.verbose)
    else:
        bone_region_mask = bone_region_tmp.astype(bool)
    del bone_region_tmp

    bone_region_mask = largest_cc_of(args.sample, args.sample_scale, bone_region_mask, 'bone_region', args.plotting, plotting_dir, args.verbose)

    if args.plotting:
        if bitpacked:
            dilated_implant_unpacked = bitpack_decode(dilated_implant, verbose=args.verbose)
        else:
            dilated_implant_unpacked = dilated_implant
        voxels_implanted = voxels.copy()
        voxels_implanted[dilated_implant_unpacked == 0] = 0

        plot_middle_planes(voxels_implanted, plotting_dir, 'implant-dilated-sanity', verbose=args.verbose)
        plot_middle_planes(bone_region_mask, plotting_dir, 'implant-bone-sanity', verbose=args.verbose)

        voxels[~bone_region_mask] = 0
        plot_middle_planes(voxels, plotting_dir, 'voxels-boned', verbose=args.verbose)

    if args.verbose >= 1: print(f"Saving bone_region mask to {output_dir}/{args.sample}.h5")
    update_hdf5_mask(f"{output_dir}/{args.sample}.h5",
                        group_name="bone_region",
                        datasets={"mask": bone_region_mask},
                        attributes={"sample": args.sample, "scale": args.sample_scale, "voxel_size": voxel_size})
