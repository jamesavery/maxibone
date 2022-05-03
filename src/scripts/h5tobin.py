#!/usr/bin/env python3
import sys, h5py, numpy as np
sys.path.append(sys.path[0]+"/../")
import pybind_kernels.histograms
from config.paths import hdf5_root, commandline_args
from tqdm import tqdm

# It dumps them chunked in block_size of xy-planes
def dump_file_to_binary(sample, y_cutoff, block_size):
    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    dl = h5py.File(f'{hdf5_root}/hdf5-byte/lsb/{sample}.h5', 'r')
    outfile = f'{hdf5_root}/binary/{sample}_voxels.bin'
    Nz, Ny, Nx = dm['voxels'].shape
    Ny -= y_cutoff
    block_size = block_size if block_size > 0 else Nz
    blocks = (Nz // block_size) + (1 if Nz % block_size > 0 else 0)
    for i in tqdm(range(blocks), desc='Dumping voxels'):
        rng = i*block_size, min((i+1)*block_size, Nz)
        tmp = np.empty((rng[1]-rng[0], Ny, Nx), np.uint16)
        tmp[:,:,:] = \
            (dm['voxels'][rng[0]:rng[1],y_cutoff:,:].astype(np.uint16) << 8) | \
            (dl['voxels'][rng[0]:rng[1],y_cutoff:,:].astype(np.uint16))
        histograms.append_slice(tmp, outfile)

    fi = h5py.File(f'{hdf5_root}/processed/implant-edt/2x/{sample}.h5', 'r')
    outfile = f'{hdf5_root}/binary/{sample}_field.bin'
    Nz, Ny, Nx = fi['voxels'].shape
    Ny -= y_cutoff // 2
    block_size = block_size if block_size > 0 else Nz
    blocks = (Nz // block_size) + (1 if Nz % block_size > 0 else 0)
    for i in tqdm(range(blocks), desc='Dumping field'):
        rng = i*block_size, min((i+1)*block_size, Nz)
        tmp = np.empty((rng[1]-rng[0], Ny, Nx), np.uint16)
        tmp[:,:,:] = fi['voxels'][rng[0]:rng[1],(y_cutoff//2):,:].astype(np.uint16)
        histograms.append_slice(tmp, outfile)


if __name__ == "__main__":
    sample, y_cutoff, block_size = commandline_args({"sample":"<required>",
                                                     "y_cutoff": 0,
                                                     "block_size":256})

    dump_file_to_binary(sample,y_cutoff,block_size)
    
