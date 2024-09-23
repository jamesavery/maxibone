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
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    argparser = default_parser(__doc__)
    argparser = add_volume(argparser, 'field', 2)
    argparser = add_volume(argparser, 'mask', 2)
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
    if args.verbose >= 1: print (f'Loaded mask with shape {mask.shape}')

    # Compute the divisable shapes
    if args.verbose >= 1: print (f'Computing divisable shapes')
    min_factor = max(args.field_scale, args.mask_scale) // args.sample_scale
    bnz = (blood.shape[0] // min_factor) * min_factor
    fnz = bnz // (args.field_scale // args.sample_scale)
    mnz = bnz // (args.mask_scale // args.sample_scale)

    blood = blood[:bnz]
    field = field[:fnz]
    args.mask = args.mask[:mnz]

    blood_file.close()
    mask_file.close()

    if args.verbose >= 2:
        print (f'Writing debug plane images to {image_output_dir}')
        nz, ny, nx = blood.shape
        names = ['yx', 'zx', 'zy']
        planes = [blood[nz//2,:,:], blood[:,ny//2,:], blood[:,:,nx//2]]

        for name, plane in zip(names, planes):
            plt.imshow(plane)
            plt.savefig(f"{image_output_dir}/{args.sample}_blood_{name}.png", bbox_inches='tight')
            plt.clf()

        nz, ny, nx = field.shape
        planes = [field[nz//2,:,:], field[:,ny//2,:], field[:,:,nx//2]]

        for name, plane in zip(names, planes):
            plt.imshow((plane < args.threshold) & (plane > 0))
            plt.savefig(f"{image_output_dir}/{args.sample}_field_{name}.png", bbox_inches='tight')
            plt.clf()

    if args.verbose >= 1: print (f'Calling into C++ to compute BICs')
    bics = np.zeros(blood.shape[0], dtype=np.float32)
    bic(blood, field, mask, args.threshold * args.sample_scale, bics, args.verbose)

    if args.verbose >= 1: print (f'Saving BICs to {image_output_dir}')
    plt.plot(bics); plt.savefig(f"{image_output_dir}/{args.sample}_bics.png", bbox_inches='tight'); plt.clf()
    np.save(f"{image_output_dir}/{args.sample}_bics.npy", bics)