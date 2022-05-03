#!/usr/bin/env python3
import sys, h5py, numpy as np
sys.path.append(sys.path[0]+"/../")
import pybind_kernels.histograms as histograms
from config.paths import hdf5_root, commandline_args
from tqdm import tqdm

# It dumps them chunked in block_size of xy-planes
def dump_file_to_binary(sample, y_cutoff, block_size):
    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    dl = h5py.File(f'{hdf5_root}/hdf5-byte/lsb/{sample}.h5', 'r')
    outfile = f'{hdf5_root}/binary/{sample}_voxels.uint16'
    Nz, Ny, Nx = dm['voxels'].shape
    Ny -= y_cutoff
    block_size = block_size if block_size > 0 else Nz
    blocks = (Nz // block_size) + (1 if Nz % block_size > 0 else 0)
    for i in tqdm(range(blocks), desc='Dumping voxels'):
        chunk_range = slice(i*block_size, min((i+1)*block_size, Nz))
        chunk = np.empty((chunk_range.stop-chunk_range.start, Ny, Nx), np.uint16)
        chunk[:,:,:] = \
            (dm['voxels'][chunk_range,y_cutoff:,:].astype(np.uint16) << 8) | \
            (dl['voxels'][chunk_range,y_cutoff:,:].astype(np.uint16))
        histograms.append_slice(chunk, outfile)

    fi = h5py.File(f'{hdf5_root}/processed/implant-edt/2x/{sample}.h5', 'r')
    Nz, Ny, Nx  = fi['voxels'].shape
    field_dtype = fi['voxels'].dtype
    Ny -= y_cutoff // 2
    block_size = block_size if block_size > 0 else Nz
    blocks = (Nz // block_size) + (1 if Nz % block_size > 0 else 0)
    outfile = f'{hdf5_root}/binary/{sample}_field.{field_dtype}'
    
    for i in tqdm(range(blocks), desc='Dumping field'):
        chunk_range = slice(i*block_size, min((i+1)*block_size, Nz))
        chunk = np.empty((chunk_range.stop-chunk_range.start, Ny, Nx), field_dtype)
        chunk[:,:,:] = fi['voxels'][chunk_range,(y_cutoff//2):,:]
        histograms.append_slice(chunk, outfile)

if __name__ == "__main__":
    sample, y_cutoff, block_size = commandline_args({"sample":"<required>",
                                                     "y_cutoff": 0,
                                                     "block_size":256})

    dump_file_to_binary(sample,y_cutoff,block_size)
    
