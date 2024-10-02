#! /usr/bin/python3
'''
Rescale a 16-bit binary file to 2x, 4x, 8x, 16x, and 32x. Each rescaling is performed on all dimensions.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.paths import hdf5_root, binary_root
import h5py
import importlib
from lib.py.commandline_args import default_parser
from lib.py.resample import downsample2x
import numpy as np
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
    argparser = default_parser(description=__doc__)
    argparser.add_argument('image', action='store', type=str, default='voxels', nargs='?',
        help='The image type to rescale. Default is voxels.')
    argparser.add_argument('dtype', action='store', type=str, default='uint16', nargs='?',
        help='The data type of the binary file. Default is uint16.')
    argparser.add_argument('--scales', action='store', type=int, default=[2, 4, 8, 16, 32], nargs='+', metavar='N',
        help='The scales to rescale down to. Default is 2 4 8 16 32.')
    args = argparser.parse_args()

    # Can do 6, 9, 12, 24, 27, etc. as well, but we currently don't. See old rescaly-cupy.py
    scales = args.scales
    T = np.dtype(args.dtype)

    input_meta  = f'{hdf5_root}/hdf5-byte/msb/{args.sample}.h5'
    input_bin   = f"{binary_root}/{args.image}/1x/{args.sample}.{args.dtype}"
    output_root = f"{binary_root}/{args.image}"

    if args.verbose >= 1:
        print(f"Generating power-of-twos rescalings for sample {args.sample}")
        print(f"Input metadata from {input_meta}")
        print(f"Input flat binary {args.dtype} data from {input_bin}")
        print(f"Output flat binary {args.dtype} data to {output_root}/[1,2,4,8,16,32]x/{args.sample}.{args.dtype}")

    meta_h5         = h5py.File(input_meta, 'r')
    full_Nz, Ny, Nx = meta_h5['voxels'].shape
    shifts          = meta_h5['volume_matching_shifts'][:]
    Nz              = full_Nz - np.sum(shifts)
    meta_h5.close()

    if args.verbose >= 1: print(f"Downscaling from 1x {(Nz,Ny,Nx)} to 2x {(Nz//2,Ny//2,Nx//2)}")

    if (args.chunk_size % 32):
        if args.verbose >= 1: print(f"Chunk size {args.chunk_size} is invalid: must be divisible by 32.")
        sys.exit(-1)

    # TODO: Just iterate now we do powers of two
    voxels2x  = np.empty((Nz//2,  Ny//2,  Nx//2),  dtype=T)
    voxels4x  = np.empty((Nz//4,  Ny//4,  Nx//4),  dtype=T)
    voxels8x  = np.empty((Nz//8,  Ny//8,  Nx//8),  dtype=T)
    voxels16x = np.empty((Nz//16, Ny//16, Nx//16), dtype=T)
    voxels32x = np.empty((Nz//32, Ny//32, Nx//32), dtype=T)
    voxels    = [voxels2x, voxels4x, voxels8x, voxels16x, voxels32x]

    chunk_rng = range(0, Nz, args.chunk_size)
    chunk_iter = tqdm.tqdm(chunk_rng, f"{args.sample}: Reading and scaling {args.chunk_size}-layer chunks") if args.verbose >= 1 else chunk_rng
    for z in chunk_iter:
        zend = min(z+args.chunk_size, Nz)
        chunk_items = (zend-z) * Ny * Nx
        # TODO Use lib calls instead of fromfile
        try:
            voxels1x_chunk = cp.fromfile(input_bin, dtype=T, count=chunk_items, offset=z*Ny*Nx*T.itemsize).reshape(zend-z,Ny,Nx)
        except:
            traceback.print_exc()
            if args.verbose >= 1: print(f"Read failed. chunk_items = {chunk_items} = {(zend-z)*Ny*Nx}, z = {z}, zend-z = {zend-z}")
            sys.exit(-1)

        voxels2x_chunk = downsample2x(voxels1x_chunk)
        del voxels1x_chunk
        voxels4x_chunk  = downsample2x(voxels2x_chunk)
        voxels8x_chunk  = downsample2x(voxels4x_chunk)
        voxels16x_chunk = downsample2x(voxels8x_chunk)
        voxels32x_chunk = downsample2x(voxels16x_chunk)

        if cupy_available:
            voxels2x[z//2:zend//2]    = voxels2x_chunk.get()
            voxels4x[z//4:zend//4]    = voxels4x_chunk.get()
            voxels8x[z//8:zend//8]    = voxels8x_chunk.get()
            voxels16x[z//16:zend//16] = voxels16x_chunk.get()
            voxels32x[z//32:zend//32] = voxels32x_chunk.get()
        else:
            voxels2x[z//2:zend//2]    = voxels2x_chunk
            voxels4x[z//4:zend//4]    = voxels4x_chunk
            voxels8x[z//8:zend//8]    = voxels8x_chunk
            voxels16x[z//16:zend//16] = voxels16x_chunk
            voxels32x[z//32:zend//32] = voxels32x_chunk

        del voxels4x_chunk
        del voxels8x_chunk
        del voxels16x_chunk
        del voxels32x_chunk

    if args.verbose >= 1: print(f"Allocating {(Nz//2,Ny//2,Nx//2)}={Nz//2*Ny//2*Nx//2} {args.dtype} for voxels2x on GPU")

    scales_rng = range(len(scales))
    scales_iter = tqdm.tqdm(scales_rng, f"{args.sample}: Downscaling to all smaller scales: {scales[2:]}") if args.verbose >= 1 else scales_rng
    for i in scales_iter:
        output_dir = f"{output_root}/{scales[i]}x/"
        pathlib.Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
        if args.verbose >= 1: print(f"Writing out scale {scales[i]}x {(voxels[i].shape)} to {output_dir}/{args.sample}.{args.dtype}")
        voxels[i].tofile(f"{output_dir}/{args.sample}.{args.dtype}")
