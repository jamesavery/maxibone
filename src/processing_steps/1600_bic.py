#! /usr/bin/python3
'''
This script computes the Bone Implant Contact (BIC) for each layer along the z-axis.
The BIC is the ratio of voxels within a distance threshold to the implant surface that are also within the soft tissue mask.
It is applied to each layer as different z-ranges indicate old and new bone.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')

from config.constants import *
from config.paths import hdf5_root, binary_root
import h5py
from lib.cpp.cpu.analysis import bic
from lib.py.commandline_args import add_volume, default_parser
from lib.py.helpers import plot_middle_planes
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    argparser = default_parser(__doc__)
    argparser = add_volume(argparser, 'field', 2, 'gauss+edt')
    argparser = add_volume(argparser, 'mask', 2, 'bone_region')
    argparser.add_argument('-t', '--threshold', action='store', type=int, default=500,
        help='The threshold for the field. Default is 500.')
    args = argparser.parse_args()

    image_output_dir = f"{hdf5_root}/processed/bic/"
    os.makedirs(image_output_dir, exist_ok=True)

    if args.verbose >= 1: print (f'Processing {args.sample} with threshold {args.threshold} and scales {args.sample_scale}x, {args.field_scale}x, {args.mask_scale}x (voxels, field, mask)')
    blood_file = h5py.File(f'{hdf5_root}/masks/{args.sample_scale}x/{args.sample}.h5', 'r')
    blood = blood_file['blood']['mask']
    voxel_size = blood_file["implant"].attrs["voxel_size"]
    if args.verbose >= 1: print (f'Loaded blood mask with shape {blood.shape} and voxel size {blood_file["blood"].attrs["voxel_size"]}')
    field = np.load(f'{binary_root}/fields/implant-{args.field}/{args.field_scale}x/{args.sample}.npy', mmap_mode='r')
    if args.verbose >= 1: print (f'Loaded field with shape {field.shape}')
    mask_file = h5py.File(f'{hdf5_root}/masks/{args.mask_scale}x/{args.sample}.h5', 'r')
    mask = mask_file[args.mask]['mask']
    if args.verbose >= 1: print (f'Loaded mask with shape {mask.shape}')

    # Compute the divisable shapes
    if args.verbose >= 1: print (f'Computing divisable shapes')
    assert blood.shape[0] >= field.shape[0] and blood.shape[0] >= mask.shape[0] # blood is the largest, handle others later
    assert field.shape[0] >= mask.shape[0] # field is the largest, handle mask later
    min_factor = max(args.field_scale, args.mask_scale) // args.sample_scale
    bnz = (blood.shape[0] // min_factor) * min_factor
    fnz = bnz // (args.field_scale // args.sample_scale)
    mnz = bnz // (args.mask_scale // args.sample_scale)

    blood = blood[:bnz]
    field = field[:fnz]
    mask = mask[:mnz]

    blood_file.close()
    mask_file.close()

    if args.verbose >= 2:
        plot_middle_planes(blood, f'{image_output_dir}', f'{args.sample}_blood', verbose=args.verbose)
        plot_middle_planes(field, f'{image_output_dir}', f'{args.sample}_field', verbose=args.verbose)

    if args.verbose >= 1: print (f'Calling into C++ to compute BICs')
    bics = np.zeros(blood.shape[0], dtype=np.float32)
    bic(blood, field, mask, args.threshold * args.sample_scale, bics, args.verbose)

    if args.verbose >= 1: print (f'Saving BICs to {image_output_dir}')
    plt.plot(bics); plt.savefig(f"{image_output_dir}/{args.sample}_bics.png", bbox_inches='tight'); plt.clf()
    np.save(f"{image_output_dir}/{args.sample}_bics.npy", bics)