#! /usr/bin/python3
'''
This script computes the implant data for a given sample. The implant data is
computed from the implant mask and the principal axis of the implant. The implant
data consists of the following:
1. The quarter profile of the implant.
2. The maximum radius of the implant in each bin.
3. The solid implant mask.
4. The front mask of the implant.
'''
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root
import h5py
from lib.cpp.gpu.geometry import fill_implant_mask_pre, fill_implant_mask, compute_front_mask
from lib.py.helpers import commandline_args, plot_middle_planes, update_hdf5, update_hdf5_mask
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import tqdm

if __name__ == "__main__":
    sample, scale, block_size, verbose = commandline_args({
        "sample" : "<required>",
        "scale" : 2,
        "block_size" : 256,
        "verbose" : 1
    })

    output_image_dir = f"{hdf5_root}/processed/implant-data/{sample}"
    os.makedirs(output_image_dir, exist_ok=True)

    if verbose >= 1: print(f"Loading principal axis and cylinder frame-of-references")
    h5meta = h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5","r")
    try:
        h5g = h5meta["implant-FoR"]
        Muvwp = h5g["UVWp_transform"][:] # Actually all that's needed
        bbox = h5g["bounding_box_UVWp"][:]
        implant_radius = h5g.attrs["implant_radius"]
        theta_range = h5g["theta_range"][:]
        h5meta.close()
    except Exception as e:
        print(f"Cant't read implant frame-of-reference: {e}")
        print(f"Make sure you have run implant-FoR.py for {sample} at scale {scale}x")
        sys.exit(-1)

    if verbose >= 1: print(f"Loading {scale}x implant mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
    try:
        implant_file = h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5",'r')
        implant      = implant_file["implant/mask"]
        voxel_size   = implant_file["implant"].attrs["voxel_size"]
        nz,ny,nx     = implant.shape
    except Exception as e:
        print(f"Can't read implant mask {e}.\nDid you run segment-implant-cc.py?")
        sys.exit(-1)

    ((Up_min, Up_max), (Vp_min, Vp_max), (Wp_min, Wp_max)) = bbox

    n_bins = 2048//scale
    rsqr_fraction = 0.7 # Fill in whenever W>Wmin and r<rsqr_fraction*rsqr_maxs[U_i]
    solid_implant_mask = np.empty(implant.shape, dtype=np.uint8)
    rsqr_maxs          = np.zeros((n_bins,), dtype=np.float32)
    profile            = np.zeros((n_bins,), dtype=np.float32)

    bbox_flat  = tuple(bbox.flatten())
    Muvwp_flat = tuple(Muvwp.flatten())
    #if verbose >= 1: print(f"Filling implant mask")
    n_blocks = (nz + block_size-1) // block_size
    mask = np.zeros((block_size,ny,nx), np.uint8)
    thetas = np.zeros((2,), np.float32)

    for i in tqdm.tqdm(range(n_blocks), desc="Filling implant mask pre"):
        z0 = i * block_size
        z1 = (i+1) * block_size
        if z1 > nz: z1 = nz
        mask[:z1-z0,:,:] = implant[z0:z1,:,:].astype(np.uint8)
        fill_implant_mask_pre(mask[:z1-z0], z0*ny*nx, voxel_size, bbox_flat, Muvwp_flat, thetas, rsqr_maxs)

    for i in tqdm.tqdm(range(n_blocks), desc="Filling implant mask"):
        z0 = i*block_size
        z1 = (i+1)*block_size
        if z1 > nz: z1 = nz
        mask[:z1-z0,:,:] = implant[z0:z1,:,:].astype(np.uint8)
        fill_implant_mask(mask[:z1-z0], z0*ny*nx, voxel_size, bbox_flat, rsqr_fraction, Muvwp_flat, thetas, rsqr_maxs, solid_implant_mask, profile)

    implant_file.close()

    update_hdf5(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5",
                group_name="implant-data",
                datasets  = {"quarter_profile": profile,
                            "r_maxs": np.sqrt(rsqr_maxs)}
    )
    plt.plot(profile)
    plt.savefig(f"{output_image_dir}/profile.png", bbox_inches='tight')
    plt.clf()

    update_hdf5_mask(f"{hdf5_root}/masks/{scale}x/{sample}.h5",
                    group_name="implant_solid",
                    datasets={"mask": solid_implant_mask.astype(bool, copy=False)},
                    attributes={"sample": sample, "scale": scale, "voxel_size": voxel_size})
    plot_middle_planes(solid_implant_mask, output_image_dir, "solid_implant_mask")

    # Compute front mask
    print ("Computing front mask")
    front_mask = np.zeros_like(solid_implant_mask)
    compute_front_mask(solid_implant_mask, voxel_size, Muvwp_flat, bbox_flat, front_mask)

    del solid_implant_mask

    update_hdf5_mask(f"{hdf5_root}/masks/{scale}x/{sample}.h5",
                    group_name="cut_cylinder_bone",
                    datasets={"mask":front_mask.astype(bool,copy=False)},
                    attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plot_middle_planes(front_mask, output_image_dir, "front_mask")
