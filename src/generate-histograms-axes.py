import jax
import numpy as np
import jax.numpy as jp
import h5py
from functools import partial
jax.config.update("jax_enable_x64", True)
NA = np.newaxis

from static_constants import *;
from esrf_read import *;
import glob;
import sys;

sample, nbits, chunk_length, hdf5_root = sys.argv[1:];

nbits, chunk_length = int(nbits), int(chunk_length);
shift = np.uint64(16-nbits)
nbins = np.uint64(2**nbits)
print(shift,nbins)

h5_msb = h5py.File(f"{hdf5_root}/hdf5-byte/msb/1x/{sample}.h5",'r');
h5_lsb = h5py.File(f"{hdf5_root}/hdf5-byte/lsb/1x/{sample}.h5",'r');


voxels_hi = h5_msb['voxels']
voxels_lo = h5_lsb['voxels']
subvolume_dimensions = h5_msb['subvolume_dimensions'];

(Nz,Ny,Nx) = voxels_hi.shape

hist_z = np.zeros((Nz,nbins),dtype=jp.uint64);
hist_y = np.zeros((Ny,nbins),dtype=jp.uint64);
hist_x = np.zeros((Nx,nbins),dtype=jp.uint64);

def cylinder_mask(Ny,Nx):
    ys = jp.linspace(-1,1,Ny)
    xs = jp.linspace(-1,1,Nx)
    return (xs[NA,:]**2 + ys[:,NA]**2) < 1 

def merge_short(chunk_hi,chunk_lo):
    hi = chunk_hi.astype(jp.uint16)
    lo = chunk_lo.astype(jp.uint16)
    return ((hi<<8) + (lo&0xff)).astype(jp.uint16)

merge_short_jit = jax.jit(merge_short)


import cupy as cp
mask = cylinder_mask(Ny,Nx)

for z in range(0,Nz,chunk_length):
    chunk_end = min(z+chunk_length,Nz);
    print(f"Reading and merging slice {z} to {z+chunk_length}")    
    chunk = (merge_short(voxels_hi[z:chunk_end], voxels_lo[z:chunk_end]) >> shift)
#    chunk = merge_short(cp.array(voxels_hi[z:chunk_end]), cp.array(voxels_lo[z:chunk_end])) >> shifte
    print(f"Copying chunk ({chunk.shape} {chunk.dtype}) to GPU")
    chunk = cp.array(np.array(chunk))
    mask  = cp.array(mask)
    print("masking")
    chunk *= mask[NA,:,:]


    print("Calculating z-histograms");
    for i in range(z,chunk_end): hist_z[i] = cp.bincount(chunk[i-z][mask],minlength=nbins).get()
#    hist_z[z:chunk_end] = jax.vmap(lambda x: jp.bincount(x[x!=0].flatten(),length=nbins))(chunk)
    

    print("Calculating y-histograms");
    hist_y_cp = cp.array(hist_y)
    for i in range(Ny):
        slice_data = chunk[:,i]
        slice_data = slice_data[slice_data!=0]
        if(len(slice_data)>0):
            hist_y_cp[i] = hist_y_cp[i] + cp.bincount(slice_data,minlength=nbins)
    hist_y = hist_y_cp.get()
#    chunk_y = jp.transpose(chunk,(1,0,2))
#    print(chunk_y.shape, hist_y.shape)    
#    hist_y[:] += jax.vmap(lambda x: jp.bincount(x[x!=0].flatten(),length=nbins))(chunk_y)
#    del chunk_y

    print("Calculating x-histograms");
    hist_x_cp = cp.array(hist_x)
    for i in range(Nx):
        slice_data = chunk[:,:,i]
        slice_data = slice_data[slice_data!=0]
        if(len(slice_data)>0):
            hist_x_cp[i] = hist_x_cp[i] + cp.bincount(slice_data,minlength=nbins)
    hist_x = hist_x_cp.get()


np.save(f"{hdf5_root}/processed/z-histograms/{sample}.npy",hist_z)
np.save(f"{hdf5_root}/processed/y-histograms/{sample}.npy",hist_y)
np.save(f"{hdf5_root}/processed/x-histograms/{sample}.npy",hist_x)
