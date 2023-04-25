import h5py, sys, os.path, pathlib
import numpy as np
import os
import sys
import h5py
import scipy.ndimage as ndi

from blockmap         import *
from config.constants import *
from config.paths import hdf5_root, commandline_args

NA = np.newaxis

sample, scale, chunk_size = commandline_args({"sample":"<required>","scale":1,'chunk_size':256})



h5meta = h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5",'r')
h5in   = h5py.File(f"{hdf5_root}/processed/volume_matched/{scale}x/{sample}.h5",'r')

output_dir = f"{hdf5_root}/processed/implant/{scale}x"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
h5out  = h5py.File(f"{output_dir}/{sample}.h5",'w')

subvolume_nz = h5meta['subvolume_dimensions'][:,0]
n_subvolumes = len(subvolume_nz)

voxelsize   = h5meta['voxels'].attrs['voxelsize'] * scale
global_vmin = np.min(h5meta['subvolume_range'][:,0])
global_vmax = np.max(h5meta['subvolume_range'][:,1])
values      = np.linspace(global_vmin,global_vmax,255)
h5meta.close()

byte_implant_threshold = np.argmin(np.abs(values-implant_threshold))

voxels_in  = h5in['voxels']
voxels_out = h5out.create_dataset("voxels",voxels_in.shape,dtype=np.uint8,compression='lzf')
# TODO: Transfer metadata consistently
voxels_out.dims[0].label = 'z';
voxels_out.dims[1].label = 'y';
voxels_out.dims[2].label = 'x';
voxels_out.attrs['voxelsize'] = voxelsize

(Nz,Ny,Nx) = voxels_in.shape

# Hvor skal denne hen?
def sphere(n):
    xs = np.linspace(-1,1,n)
    return (xs[:,NA,NA]**2 + xs[NA,:,NA]**2 + xs[NA,NA,:]**2) <= 1

sphere_diameter = 2*int(20/voxelsize+0.5)+1
if(sphere_diameter>1):
    sph5 = sphere(sphere_diameter)

print(f"Reading {voxels_in.shape} voxels of type {voxels_in.dtype} from "+f"{hdf5_root}/processed/volume_matched/{scale}x/{sample}.h5")
print(f"Implant threshold {implant_threshold} -> {byte_implant_threshold} as byte")

for z0 in range(0,Nz,chunk_size):
    z1 = min(Nz,z0+chunk_size)
    print(f"Reading and thresholding chunk {z0}:{z1} of {voxels_in.shape} {voxels_in.dtype}.")
    implant_chunk       = voxels_in[z0:z1] >= byte_implant_threshold
    print(f"Max inddata: {voxels_in[z0:z1].max()}; Number of matching voxels: {np.sum(implant_chunk)}")
    if(sphere_diameter>1):
        print(f"Binary opening with {sphere_diameter*voxelsize} micrometer sphere ({sphere_diameter} voxel radius).")
        implant_chunk[sphere_diameter//2:-sphere_diameter//2] = ndi.binary_opening(implant_chunk,sph5)[sphere_diameter//2:-sphere_diameter//2]
    print("Writing chunk")
    voxels_out[z0:z1]  = implant_chunk
    
h5in.close()
h5out.close()
