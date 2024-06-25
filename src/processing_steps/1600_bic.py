import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, hdf5_root_fast, binary_root
import h5py
from lib.cpp.cpu_seq.io import load_slice, write_slice
from lib.cpp.cpu.analysis import bic
from lib.py.helpers import block_info, load_block, commandline_args
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    sample, threshold, scale, field_scale, material, block_size, mask, mask_scale, verbose = commandline_args({"sample" : "<required>",
                                                            "threshold" : 500,
                                                            "scale" : 1,
                                                            "field_scale" : 2,
                                                            "material" : 0,
                                                            "block_size" : 8,
                                                            "mask" : "bone_region",
                                                            "mask_scale" : 2,
                                                            "verbose" : 2})

    field_name = "edt"

    image_output_dir = f"{hdf5_root}/processed/bic/"
    os.makedirs(image_output_dir, exist_ok=True)

    blood_file = h5py.File(f'{hdf5_root}/masks/{scale}x/{sample}.h5', 'r')
    blood = blood_file['blood']['mask'][:]
    voxel_size = blood_file["implant"].attrs["voxel_size"]
    field_file = np.load(f'{binary_root}/fields/implant-edt/{field_scale}x/{sample}.npy', mmap_mode='r')
    front_file = h5py.File(f'{hdf5_root}/masks/{mask_scale}x/{sample}.h5', 'r')
    front = front_file['bone_region']['mask']
    field = field_file * front
    del field_file, front

    if verbose >= 2:
        print (f'Writing debug plane images to {image_output_dir}')
        nz, ny, nx = blood.shape
        plt.imshow(blood[-100,:,:]); plt.savefig(f"{image_output_dir}/{sample}_blood_yx.png", bbox_inches='tight'); plt.clf()
        plt.imshow(blood[:,ny//2,:]); plt.savefig(f"{image_output_dir}/{sample}_blood_zx.png", bbox_inches='tight'); plt.clf()
        plt.imshow(blood[:,:,nx//2]); plt.savefig(f"{image_output_dir}/{sample}_blood_zy.png", bbox_inches='tight'); plt.clf()
        nz, ny, nx = field.shape
        plt.imshow((field[nz//2,:,:] < threshold) & (field[nz//2,:,:] > 0)); plt.savefig(f"{image_output_dir}/{sample}_field_yx.png", bbox_inches='tight'); plt.clf()
        plt.imshow(field[:,ny//2,:]); plt.savefig(f"{image_output_dir}/{sample}_field_zx.png", bbox_inches='tight'); plt.clf()
        plt.imshow(field[:,:,nx//2]); plt.savefig(f"{image_output_dir}/{sample}_field_zy.png", bbox_inches='tight'); plt.clf()

    bics = np.zeros(blood.shape[0], dtype=np.float32)

    bic(blood, field, threshold, bics)

    plt.plot(bics); plt.savefig(f"{image_output_dir}/{sample}_bics.png", bbox_inches='tight'); plt.clf()
    np.save(f"{image_output_dir}/{sample}_bics.npy", bics)