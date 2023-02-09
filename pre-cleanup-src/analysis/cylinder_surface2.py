#!/usr/bin/env python3
import os, sys, h5py, numpy as np, pathlib, tqdm, vedo, matplotlib.pyplot as plt, edt, vedo.pointcloud as pc, scipy.ndimage as ndi
sys.path.append(sys.path[0]+"/../")
from config.paths import *
from helper_functions import *
from pybind_kernels.geometry import cylinder_projection
NA = np.newaxis


def sphere(n):
    xs = np.linspace(-1,1,n)
    return (xs[:,NA,NA]**2 + xs[NA,:,NA]**2 + xs[NA,NA,:]**2) <= 1


def homogeneous_transform(xs, M):
    shape = np.array(xs.shape)
    assert(shape[-1] == 3)
    shape[-1] = 4
    hxs = np.empty(shape,dtype=xs.dtype)
    hxs[...,:3] = xs;
    hxs[..., 3]  = 1

    print(hxs.shape, M.shape)
    return (hxs @ M.T)[...,:3]

def np_save(path,data):
    output_dir = os.path.dirname(path)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)        
    np.save(path,data)

    
# Requires: implant-FoR
#           soft-tissue/bone segmentation + blood analysis
#           EDT-field
sample, mask_scale, segment_scale = commandline_args({"sample":"<required>","mask_scale":8,"segment_Scale":1}) #,"segment_scale":8}) #TODO: Implement higher resolution segment scale


h5meta = h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5","r")
h5mask = h5py.File(f"{hdf5_root}/masks/{mask_scale}x/{sample}.h5","r")


Cs_voxel_size     = h5meta["voxels"].attrs["voxelsize"]*segment_scale
mask_voxel_size   = h5meta["voxels"].attrs["voxelsize"]*mask_scale
(Nz,Ny,Nx)     = h5meta["voxels"].shape
Nz            -= np.sum(h5meta["volume_matching_shifts"][:])
(nz,ny,nx)     = np.array([Nz,Ny,Nx])//segment_scale


try:
    h5g = h5meta["implant-FoR"]
    Muvwp                = h5g["UVWp_transform"][:]
    bbox                 = h5g["bounding_box_UVWp"][:]
    theta_min, theta_max = h5g["theta_range"][:]
except Exception as e:
    print(f"Cant't read implant frame-of-reference: {e}")
    print(f"Make sure you have run segment-implant-cc.py and implant-FoR.py for {sample} at scale {mask_scale}x")
    sys.exit(-1)
    
try:
    blood_mask    = h5mask["blood/mask"][:]
    solid_implant = h5mask["implant_solid/mask"][:]
    h5mask.close()    
except Exception as e:
    print(f"Cant't read masks: {e}")
    print("Make sure you have run compute_histograms.py, generate_xx_probabilities.py, segment_from_distributions,\n"+
          "and segment-blood-cc.py")
    sys.exit(-1)

    
P0_binfile            = f"{binary_root}/segmented/P0/{segment_scale}x/{sample}.uint16"
P1_binfile            = f"{binary_root}/segmented/P1/{segment_scale}x/{sample}.uint16"
edt_binfile           = f"{binary_root}/fields/implant-edt/{mask_scale}x/{sample}.uint16"


M = h5meta["implant-FoR/UVWp_transform"][:]

print(f"Loading {segment_scale}x segmentation from {P0_binfile} and {P1_binfile}")
Cs = np.empty((2,nz,ny,nx),dtype=np.uint8)
Cs[0] = np.memmap(P0_binfile,dtype=np.uint16,mode="r",shape=(nz,ny,nx)) >> 8
Cs[1] = np.memmap(P1_binfile,dtype=np.uint16,mode="r",shape=(nz,ny,nx)) >> 8

print(f"Computing {solid_implant.shape} EDT on implant_solid mask")
edt_field = edt.edt(~solid_implant,parallel=32)*mask_voxel_size

print(f"Computing cylinder projection images")
n_theta, n_Up = 1800//segment_scale,3200//segment_scale
images = np.zeros((2,n_theta, n_Up), dtype=np.float32)
counts = np.zeros((2,n_theta, n_Up), dtype=np.uint64)
d_min, d_max = 0.1, 50*Cs_voxel_size

cylinder_projection(edt_field, Cs, Cs_voxel_size,
                    d_min, d_max, theta_min, theta_max,
                    tuple(bbox.flatten()), tuple(Muvwp.flatten()),
                    images, counts)
                    
