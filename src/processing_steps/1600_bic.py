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
    bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', block_size, 0, 0)
    (Nz,Ny,Nx,Nr) = bi['dimensions']
    block_size    = bi['block_size']
    n_blocks      = bi['n_blocks']
    blocks_are_subvolumes = bi['blocks_are_subvolumes']

    #n_blocks = 1
    #n_blocks -= 1

    image_output_dir = f"{hdf5_root}/processed/bic/"
    os.makedirs(image_output_dir, exist_ok=True)

    blood_file = h5py.File(f'{hdf5_root}/masks/{scale}x/{sample}.h5', 'r')
    blood = blood_file['blood']['mask'][:]
    field = np.load(f'{binary_root}/fields/implant-edt/{field_scale}x/{sample}.npy', mmap_mode='r')
    front_file = h5py.File(f'{hdf5_root}/masks/{mask_scale}x/{sample}.h5', 'r')
    front = front_file['bone_region']['mask']

    xs = np.linspace(-1,1,Nx)
    rs = np.sqrt(xs[np.newaxis,np.newaxis,:]**2 + xs[np.newaxis,:,np.newaxis]**2)
    cylinder_mask = (rs<=1)

    print (blood.shape, field.shape)

    bics = np.zeros(Nz, dtype=np.float32)

    print (bic(blood, field, threshold))
    quit()

    #for b in tqdm(range(n_blocks), desc='Computing histograms', position=0):
        #for l in tqdm(range(0, min(block_size, Nz - b*block_size)), desc='Computing layer', position=1, leave=False):

            #i_blood = b*block_size + l
            #bics[i_blood] = bic(blood[i_blood:i_blood+1], bfield[i_blood:i_blood+1] < threshold)
            #i_field = i_blood // field_scale
            #i_front = min(front.shape[0]-1, i_blood // mask_scale)
            ##blood_zstart = i*block_size
            ##blood_zend = (i+1)*block_size
            ##field_zstart = i*(block_size//field_scale)
            ##field_zend = (i+1)*(block_size//field_scale)
            ##front_zstart = i*(block_size//mask_scale)
            ##front_zend = (i+1)*(block_size//mask_scale)

            ##this_blood = blood[blood_zstart:blood_zend].astype(np.uint8)
            ##this_field = field[field_zstart:field_zend]
            ##this_front = front[front_zstart:front_zend].astype(np.uint8)

            #this_blood = blood[i_blood:i_blood+1].astype(np.uint8)
            #this_field = field[i_field:i_field+1]
            #this_front = front[i_front:i_front+1].astype(np.uint8)

            ##print (this_blood.shape, this_field.shape, this_front.shape)

            #this_field = (this_field < threshold).astype(np.uint8)

            ## Upscale fields to the same resolution as the voxels
            ##this_field = np.repeat(np.repeat(np.repeat(this_field, field_scale, axis=0), field_scale, axis=1), field_scale, axis=2)
            #this_field = np.repeat(np.repeat(this_field, field_scale, axis=1), field_scale, axis=2)
            #this_field *= cylinder_mask

            ## Upscale front to the same resolution as the voxels
            ##this_front = np.repeat(np.repeat(np.repeat(this_front, mask_scale, axis=0), mask_scale, axis=1), mask_scale, axis=2)
            #this_front = np.repeat(np.repeat(this_front, mask_scale, axis=1), mask_scale, axis=2)

            ##print (this_blood.shape, this_field.shape, this_front.shape)

            #this_field *= this_front

            #bloods = np.sum(this_blood * this_field)
            #totals = max(1, np.sum(this_field))
            #bics[i_blood] = bloods / totals
            ##print ('BIC', bics[b])

            #if verbose >= 1 and i_blood == 1000:
            #    nz, ny, nx = this_blood.shape
            #    plt.figure(figsize=(10,10)); plt.imshow(this_blood[nz//2,:,:]); plt.colorbar(); plt.savefig(f"{image_output_dir}/{sample}_{b}_blood_yx.png", bbox_inches='tight'); plt.clf()
            #    plt.figure(figsize=(10,10)); plt.imshow(this_blood[:,ny//2,:]); plt.colorbar(); plt.savefig(f"{image_output_dir}/{sample}_{b}_blood_zx.png", bbox_inches='tight'); plt.clf()
            #    plt.figure(figsize=(10,10)); plt.imshow(this_blood[:,:,nx//2]); plt.colorbar(); plt.savefig(f"{image_output_dir}/{sample}_{b}_blood_zy.png", bbox_inches='tight'); plt.clf()
            #    plt.figure(figsize=(10,10)); plt.imshow(this_field[nz//2,:,:]); plt.colorbar(); plt.savefig(f"{image_output_dir}/{sample}_{b}_field_yx.png", bbox_inches='tight'); plt.clf()
            #    plt.figure(figsize=(10,10)); plt.imshow(this_field[:,ny//2,:]); plt.colorbar(); plt.savefig(f"{image_output_dir}/{sample}_{b}_field_zx.png", bbox_inches='tight'); plt.clf()
            #    plt.figure(figsize=(10,10)); plt.imshow(this_field[:,:,nx//2]); plt.colorbar(); plt.savefig(f"{image_output_dir}/{sample}_{b}_field_zy.png", bbox_inches='tight'); plt.clf()
            #    plt.figure(figsize=(10,10)); plt.imshow(this_front[nz//2,:,:]); plt.colorbar(); plt.savefig(f"{image_output_dir}/{sample}_{b}_front_yx.png", bbox_inches='tight'); plt.clf()
            #    plt.figure(figsize=(10,10)); plt.imshow(this_front[:,ny//2,:]); plt.colorbar(); plt.savefig(f"{image_output_dir}/{sample}_{b}_front_zx.png", bbox_inches='tight'); plt.clf()
            #    plt.figure(figsize=(10,10)); plt.imshow(this_front[:,:,nx//2]); plt.colorbar(); plt.savefig(f"{image_output_dir}/{sample}_{b}_front_zy.png", bbox_inches='tight'); plt.clf()

    bics = 1 - bics # Convert to bone volume fraction
    plt.plot(bics); plt.savefig(f"{image_output_dir}/{sample}_bics.png", bbox_inches='tight'); plt.clf()
    np.save(f"{image_output_dir}/{sample}_bics.npy", bics)