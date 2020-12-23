import histograms, numpy as np
from time import time;
import sys

def axes_histogram(voxels, ranges=None, voxel_bins=256):
    print("Hello\n",flush=True)
    print("again\n",flush=True)
    (Nz,Ny,Nx) = voxels.shape
    Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1
    
    x_bins   = np.zeros((Nx,voxel_bins),dtype=np.uint64)
    y_bins   = np.zeros((Ny,voxel_bins),dtype=np.uint64)
    z_bins   = np.zeros((Nz,voxel_bins),dtype=np.uint64)
    r_bins   = np.zeros((Nr,voxel_bins),dtype=np.uint64)
    print("my old\n",flush=True)
    
    if ranges is None:
        vmin, vmax = 1, 255
    else:
        vmin, vmax = ranges

    print(f"friend: {vmin}, {vmax}\n",flush=True)        

    histograms.axis_histogram(np.ascontiguousarray(voxels), x_bins, y_bins, z_bins, r_bins, vmin, vmax);

    return x_bins, y_bins, z_bins, r_bins


def field_histogram(voxels, field, ranges=None,field_bins=256, voxel_bins=256):
    assert(voxels.shape == field.shape)

    
    bins   = np.zeros((field_bins,voxel_bins),dtype=np.uint64)

    # TODO: Don't scan over array 4 times - perhaps roll into user kernel
    if ranges is None:
        vmin, vmax = 1, 255
        fmin, fmax = field.min(), field.max()
    else:
        ((vmin,vmax),(fmin,fmax)) = ranges

    histograms.field_histogram(voxels,field,bins,vmin,vmax,fmin,fmax)
    
    return bins
