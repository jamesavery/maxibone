import jax
import numpy as np
import jax.numpy as jp
import h5py
from functools import partial
jax.config.update("jax_enable_x64", True)

from static_constants import *;
from esrf_read import *;
import glob;
import sys;

sample, nbins, chunk_length, hdf5_root = sys.argv[1:];

nbins, chunk_length = int(nbins), int(chunk_length);
bin_edges = np.linspace(0, 2**16-1, nbins+1);

h5_msb = h5py.File(f"{hdf5_root}/hdf5-byte/scale/1x/{sample}.h5",'r');
h5_lsb = h5py.File(f"{hdf5_root}/hdf5-byte/lsb/1x/{sample}.h5",'r');


voxels_hi = h5_msb['voxels']
voxels_lo = h5_lsb['voxels']
subvolume_dimensions = h5_msb['subvolume_dimensions'];

(Nz,Ny,Nx) = voxels_hi.shape

hist_z = np.zeros((Nz,nbins),dtype=jp.uint32);
hist_y = np.zeros((Ny,nbins),dtype=jp.uint32);
hist_x = np.zeros((Nx,nbins),dtype=jp.uint32);

def merge_short(chunk_hi,chunk_lo):
    hi = chunk_hi.astype(jp.uint16)
    lo = chunk_lo.astype(jp.uint16)
    return (hi<<8) | lo

merge_short_jit = jax.jit(merge_short)

for z in range(0,Nz,chunk_length):
    chunk_end = min(z+chunk_length,Nz);
    print(f"Reading and merging slice {z} to {z+chunk_length}")    
    chunk = merge_short_jit(voxels_hi[z:chunk_end], voxels_lo[z:chunk_end])
    print("Calculating x-histograms");
    chunk_x = jp.transpose(chunk,(2,0,1))
    print(chunk_x.shape, hist_x.shape)
    hist_x[:] += jax.vmap(lambda x: jp.histogram(x,bin_edges)[0])(chunk_x)
    del chunk_x    
    print("Calculating y-histograms");
    chunk_y = jp.transpose(chunk,(1,0,2))
    hist_y[:] += jax.vmap(lambda x: jp.histogram(x,bin_edges)[0])(chunk_y)
    del chunk_y
    print("Calculating z-histograms");
    hist_z[z:chunk_end] = jax.vmap(lambda x: jp.histogram(x,bin_edges)[0])(chunk)


np.save(f"{hdf5_root}/processed/z-histograms/{sample}.npy",hist_z)
np.save(f"{hdf5_root}/processed/y-histograms/{sample}.npy",hist_y)
np.save(f"{hdf5_root}/processed/x-histograms/{sample}.npy",hist_x)
