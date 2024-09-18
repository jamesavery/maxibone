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
from lib.py.helpers import commandline_args
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    sample, scale, field_scale, threshold,  mask, mask_scale, verbose = commandline_args({
        "sample" : "<required>",
        "scale" : 1,
        "field_scale" : 2,
        "threshold" : 500,
        "mask" : "bone_region",
        "mask_scale" : 2,
        "verbose" : 2
    })

    field_name = "edt"
    image_output_dir = f"{hdf5_root}/processed/bic/"
    os.makedirs(image_output_dir, exist_ok=True)

    if verbose >= 1: print (f'Processing {sample} with threshold {threshold} and scales {scale}x, {field_scale}x, {mask_scale}x (voxels, field, mask)')
    blood_file = h5py.File(f'{hdf5_root}/masks/{scale}x/{sample}.h5', 'r')
    blood = blood_file['blood']['mask']
    voxel_size = blood_file["implant"].attrs["voxel_size"]
    field = np.load(f'{binary_root}/fields/implant-edt/{field_scale}x/{sample}.npy', mmap_mode='r')
    mask_file = h5py.File(f'{hdf5_root}/masks/{mask_scale}x/{sample}.h5', 'r')
    mask = mask_file[mask]['mask']

    # Compute the divisable shapes
    assert blood.shape[0] >= field.shape[0] and blood.shape[0] >= mask.shape[0] # blood is the largest, handle others later
    assert field.shape[0] >= mask.shape[0] # field is the largest, handle mask later
    min_factor = max(field_scale, mask_scale) // scale
    bnz = (blood.shape[0] // min_factor) * min_factor
    fnz = bnz // (field_scale // scale)
    mnz = bnz // (mask_scale // scale)

    blood = blood[:bnz]
    field = field[:fnz]
    mask = mask[:mnz]

    blood_file.close()
    mask_file.close()

    if verbose >= 2:
        print (f'Writing debug plane images to {image_output_dir}')
        nz, ny, nx = blood.shape
        names = ['yx', 'zx', 'zy']
        planes = [blood[nz//2,:,:], blood[:,ny//2,:], blood[:,:,nx//2]]

        for name, plane in zip(names, planes):
            plt.imshow(plane)
            plt.savefig(f"{image_output_dir}/{sample}_blood_{name}.png", bbox_inches='tight')
            plt.clf()

        nz, ny, nx = field.shape
        planes = [field[nz//2,:,:], field[:,ny//2,:], field[:,:,nx//2]]

        for name, plane in zip(names, planes):
            plt.imshow((plane < threshold) & (plane > 0))
            plt.savefig(f"{image_output_dir}/{sample}_field_{name}.png", bbox_inches='tight')
            plt.clf()

    bics = np.zeros(blood.shape[0], dtype=np.float32)

    bic(blood, field, mask, threshold * scale, bics)

    plt.plot(bics); plt.savefig(f"{image_output_dir}/{sample}_bics.png", bbox_inches='tight'); plt.clf()
    np.save(f"{image_output_dir}/{sample}_bics.npy", bics)