#! /usr/bin/python3
'''
This script computes the connected components of the segmented blood mask.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')

from config.constants import *
from config.paths import hdf5_root, binary_root
import datetime
from functools import partial
from lib.cpp.cpu.connected_components import largest_connected_component
from lib.cpp.cpu_seq.io import load_slice
from lib.py.commandline_args import add_volume, default_parser
from lib.py.helpers import chunk_info, plot_middle_planes, update_hdf5
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import numpy as np
import os.path
import pathlib
import scipy.ndimage as ndi
import tqdm

if __name__ == '__main__':
    argparser = default_parser(__doc__, 0)
    argparser = add_volume(argparser, 'field', 2, 'gauss+edt')
    argparser.add_argument('-m', '--material', action='store', type=int, default=0,
        help='The material to segment. Default is 0, which should be soft tissue.')
    args = argparser.parse_args()

    scales = [32, 16, 8, 4, 2, 1] if args.sample_scale <= 0 else [args.sample_scale]
    bi = chunk_info(f'{hdf5_root}/hdf5-byte/msb/{args.sample}.h5', 1)
    Nz, Ny, Nx, _ = bi["dimensions"]

    for scale in tqdm.tqdm(scales, desc= 'Computing connected components'):
        data = f'{binary_root}/segmented/{args.field}/P{args.material}/{scale}x/{args.sample}.uint16'
        output_dir = f'{hdf5_root}/masks/{scale}x'
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        nz, ny, nx = Nz // scale, Ny // scale, Nx // scale
        voxel_size = bi["voxel_size"]*scale

        if args.verbose >= 1:
            plot_dir = f"{hdf5_root}/processed/blood_mask/"
            pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

        layer_size = ny*nx
        hyperthreading = True # TODO check if hyperthreading is enabled
        n_cores = mp.cpu_count() // (2 if hyperthreading else 1) # Only count physical cores
        available_memory = 1024**3 * 4 * n_cores # 1 GB per core-ish
        memory_per_core = available_memory // n_cores
        elements_per_core = memory_per_core // 8 # 8 bytes per element
        layers_per_core = elements_per_core // layer_size
        n_chunks = int(2**np.ceil(np.log2(nz // layers_per_core)))
        layers_per_chunk = nz // n_chunks
        chunk_size_elements = layers_per_chunk * layer_size
        chunk_size_bytes = chunk_size_elements * 8

        if layers_per_chunk == 0 or layers_per_chunk >= nz:
            voxels = np.empty((nz, ny, nx), dtype=np.uint16)
            load_slice(voxels, data, (0,0,0), (nz, ny, nx))
            label, n_features = ndi.label(voxels)
            counts = np.bincount(label[label > 0], minlength=n_features+1)
            largest_cc = np.argmax(counts)
            mask = (label == largest_cc)
        else:
            intermediate_folder = f"/tmp/maxibone/labels_blood/{scale}x/"
            os.makedirs(intermediate_folder, exist_ok=True)

            def label_chunk(i, chunk_size, chunk_prefix):
                start = i*chunk_size
                end   = (i+1)*chunk_size if i < n_chunks-1 else nz # Last chunk gets the rest
                chunk_length = end-start
                voxel_chunk   = np.empty((chunk_length,ny,nx),dtype=np.uint16)
                load_slice(voxel_chunk, data, (start,0,0), voxel_chunk.shape)
                if args.verbose >= 3:
                    plot_middle_planes(voxel_chunk, plot_dir, f'{args.sample}_{scale}_{i}_voxels')
                label, n_features = ndi.label(voxel_chunk, output=np.int64)
                del voxel_chunk
                label.tofile(f'{chunk_prefix}{i}.int64')
                del label
                return n_features

            start = datetime.datetime.now()
            with ThreadPool(n_cores) as pool:
                label_chunk_partial = partial(label_chunk, chunk_size=layers_per_chunk, chunk_prefix=f"{intermediate_folder}/{args.sample}_")
                n_labels = pool.map(label_chunk_partial, range(n_chunks))
            end = datetime.datetime.now()
            flat_size = nz*ny*nx
            # load uint16, threshold (uint16 > uint8), label (int64), write int64
            total_bytes_processed = flat_size*2 + flat_size*2 + flat_size*8 + flat_size*8
            gb_per_second = total_bytes_processed / (end-start).total_seconds() / 1024**3
            print (f'Loading and labelling took {end-start}. (throughput: {gb_per_second:.02f} GB/s)')

            np.array(n_labels, dtype=np.int64).tofile(f"{intermediate_folder}/{args.sample}_n_labels.int64")

            mask = np.zeros((nz,ny,nx), dtype=bool)
            largest_connected_component(mask, f"{intermediate_folder}/{args.sample}_", n_labels, (nz,ny,nx), (layers_per_chunk,ny,nx), args.verbose)

        if args.verbose >= 1:
            plot_middle_planes(mask, plot_dir, f'{args.sample}_{scale}_{args.field}_mask', verbose=args.verbose)

        update_hdf5(f"{output_dir}/{args.sample}.h5",
                    group_name=f"blood",
                    datasets={'mask':mask},
                    attributes={
                        'scale': scale,
                        'voxel_size': voxel_size,
                        'sample': args.sample,
                        'name': "blood_mask"
                    })
