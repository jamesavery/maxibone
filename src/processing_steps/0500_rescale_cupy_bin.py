#! /usr/bin/python3
'''
Rescale a 16-bit binary file to 2x, 4x, 8x, 16x, and 32x. Each rescaling is performed on all dimensions.
'''
import sys
sys.path.append(sys.path[0]+"/../")

from config.paths import hdf5_root, binary_root
import h5py
import importlib
from lib.py.helpers import commandline_args
from lib.py.resample import downsample2x
import numpy as np
import pathlib
import tqdm
import traceback

cupy_available = importlib.util.find_spec("cupy") is not None
if cupy_available:
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
else:
    import numpy as cp

if __name__ == "__main__":
    sample, image, chunk_size, dtype, verbose = commandline_args({
        "sample" : "<required>",
        "image" : "voxels",
        "chunk_size" : 32*2,
        "dtype" : "uint16",
        "verbose" : 1
    })

    # Can do 6, 9, 12, 24, 27, etc. as well, but we currently don't. See old rescaly-cupy.py
    scales = [2, 4, 8, 16, 32]
    T = np.dtype(dtype)

    input_meta  = f'{hdf5_root}/hdf5-byte/msb/{sample}.h5'
    input_bin   = f"{binary_root}/{image}/1x/{sample}.{dtype}"
    output_root = f"{binary_root}/{image}"

    if verbose >= 1:
        print(f"Generating power-of-twos rescalings for sample {sample}")
        print(f"Input metadata from {input_meta}")
        print(f"Input flat binary {dtype} data from {input_bin}")
        print(f"Output flat binary {dtype} data to {output_root}/[1,2,4,8,16,32]x/{sample}.{dtype}")

    meta_h5         = h5py.File(input_meta, 'r')
    full_Nz, Ny, Nx = meta_h5['voxels'].shape
    shifts          = meta_h5['volume_matching_shifts'][:]
    Nz              = full_Nz - np.sum(shifts)
    meta_h5.close()

    if verbose >= 1: print(f"Downscaling from 1x {(Nz,Ny,Nx)} to 2x {(Nz//2,Ny//2,Nx//2)}")

    if (chunk_size % 32):
        if verbose >= 1: print(f"Chunk size {chunk_size} is invalid: must be divisible by 32.")
        sys.exit(-1)

    # TODO: Just iterate now we do powers of two
    voxels2x  = np.empty((Nz//2,  Ny//2,  Nx//2),  dtype=T)
    voxels4x  = np.empty((Nz//4,  Ny//4,  Nx//4),  dtype=T)
    voxels8x  = np.empty((Nz//8,  Ny//8,  Nx//8),  dtype=T)
    voxels16x = np.empty((Nz//16, Ny//16, Nx//16), dtype=T)
    voxels32x = np.empty((Nz//32, Ny//32, Nx//32), dtype=T)
    voxels    = [voxels2x, voxels4x, voxels8x, voxels16x, voxels32x]

    for z in tqdm.tqdm(range(0, Nz, chunk_size), f"{sample}: Reading and scaling {chunk_size}-layer chunks"):
        zend = min(z+chunk_size, Nz)
        chunk_items = (zend-z) * Ny * Nx
        # TODO Use lib calls instead of fromfile
        try:
            voxels1x_chunk = cp.fromfile(input_bin, dtype=T, count=chunk_items, offset=z*Ny*Nx*T.itemsize).reshape(zend-z,Ny,Nx)
        except:
            traceback.print_exc()
            if verbose >= 1: print(f"Read failed. chunk_items = {chunk_items} = {(zend-z)*Ny*Nx}, z = {z}, zend-z = {zend-z}")
            sys.exit(-1)

        voxels2x_chunk = downsample2x(voxels1x_chunk)
        del voxels1x_chunk
        voxels4x_chunk  = downsample2x(voxels2x_chunk)
        voxels8x_chunk  = downsample2x(voxels4x_chunk)
        voxels16x_chunk = downsample2x(voxels8x_chunk)
        voxels32x_chunk = downsample2x(voxels16x_chunk)

        if cupy_available:
            voxels2x[z//2:zend//2]  = voxels2x_chunk.get()
            voxels4x[z//4:zend//4]  = voxels4x_chunk.get()
            voxels8x[z//8:zend//8]  = voxels8x_chunk.get()
            voxels16x[z//16:zend//16] = voxels16x_chunk.get()
            voxels32x[z//32:zend//32] = voxels32x_chunk.get()
        else:
            voxels2x[z//2:zend//2]  = voxels2x_chunk
            voxels4x[z//4:zend//4]  = voxels4x_chunk
            voxels8x[z//8:zend//8]  = voxels8x_chunk
            voxels16x[z//16:zend//16] = voxels16x_chunk
            voxels32x[z//32:zend//32] = voxels32x_chunk

        del voxels4x_chunk
        del voxels8x_chunk
        del voxels16x_chunk
        del voxels32x_chunk

    if verbose >= 1: print(f"Allocating {(Nz//2,Ny//2,Nx//2)}={Nz//2*Ny//2*Nx//2} {dtype} for voxels2x on GPU")

    for i in tqdm.tqdm(range(len(scales)), f"{sample}: Downscaling to all smaller scales: {scales[2:]}"):
        output_dir = f"{output_root}/{scales[i]}x/"
        pathlib.Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
        if verbose >= 1: print(f"Writing out scale {scales[i]}x {(voxels[i].shape)} to {output_dir}/{sample}.uint16")
        voxels[i].tofile(f"{output_dir}/{sample}.uint16")
