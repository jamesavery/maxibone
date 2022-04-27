import histograms, numpy as np
from time import time;
import sys

# TODO: Currently specialized to uint16_t
def masked_minmax(voxels):
    return histograms.masked_minmax(voxels)

def axes_histogram(voxels, ranges=None, voxel_bins=256):
    (Nz,Ny,Nx) = voxels.shape
    Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1
    
    x_bins   = np.zeros((Nx,voxel_bins),dtype=np.uint64)
    y_bins   = np.zeros((Ny,voxel_bins),dtype=np.uint64)
    z_bins   = np.zeros((Nz,voxel_bins),dtype=np.uint64)
    r_bins   = np.zeros((Nr,voxel_bins),dtype=np.uint64)
    
    if ranges is None:
        vmin, vmax = 1, 4095
    else:
        vmin, vmax = ranges

    histograms.axis_histogram(voxels, x_bins, y_bins, z_bins, r_bins, vmin, vmax);
    return x_bins, y_bins, z_bins, r_bins


def field_histogram(voxels, field, ranges=None,field_bins=256, voxel_bins=256):
    assert(voxels.dtype == np.uint16)
    
    bins   = np.zeros((field_bins,voxel_bins),dtype=np.uint64)

    if ranges is None:
        vmin, vmax = masked_minmax(voxels)
    else:
        (vmin,vmax) = ranges

    print("Calculating field histogram",flush=True);        
    histograms.field_histogram(voxels,field,bins,vmin,vmax)
    
    return bins
