#! /usr/bin/python3
'''
Segment the implant using coarse thresholding followed by connected components.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')

from config.constants import *
from config.paths import hdf5_root, binary_root
import datetime
from functools import partial
import h5py
from lib.cpp.cpu.connected_components import largest_connected_component
from lib.cpp.cpu.io import load_slice
from lib.py.commandline_args import default_parser
from lib.py.helpers import update_hdf5_mask, plot_middle_planes
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import os.path
import pathlib
import scipy.ndimage as ndi

if __name__ == "__main__":
    args = default_parser(__doc__).parse_args()

    # Load metadata. TODO: Clean up, make automatic function.
    meta_filename = f"{hdf5_root}/hdf5-byte/msb/{args.sample}.h5"
    h5meta     = h5py.File(meta_filename,'r')
    vm_shifts  = h5meta['volume_matching_shifts'][:]
    full_Nz, Ny, Nx = h5meta['voxels'].shape               # Full image resolution
    Nz         = full_Nz - np.sum(vm_shifts)               # Full volume matched image resolution
    nz,ny,nx   = np.array([Nz,Ny,Nx]) // args.sample_scale # Volume matched image resolution at chosen scale
    intermediate_folder = f"/tmp/maxibone/labels_implant/{args.sample_scale}x/"
    os.makedirs(intermediate_folder, exist_ok=True)
    if args.verbose >= 1:
        plot_dir = f"{hdf5_root}/processed/implant_mask/"
        pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    voxel_size  = h5meta['voxels'].attrs['voxelsize'] * args.sample_scale
    global_vmin = np.min(h5meta['subvolume_range'][:,0])
    global_vmax = np.max(h5meta['subvolume_range'][:,1])
    values      = np.linspace(global_vmin,global_vmax,2**16)
    implant_threshold_u16 = np.argmin(np.abs(values-implant_threshold))
    if 'novisim' in args.sample:
        implant_threshold_u16 = implant_threshold_u16_novisim

    # Automatic chunk size calculation.
    # Should be that fmod(log2(n_chunks),1.0) == 0 and chunk_size * n_cores < available memory
    layer_size = ny*nx
    hyperthreading = True # TODO check if hyperthreading is enabled
    n_cores = mp.cpu_count() // (2 if hyperthreading else 1) # Only count physical cores
    available_memory = 1024**3 * 4 * n_cores # 1 GB per core-ish
    memory_per_core  = available_memory // n_cores
    elements_per_core = memory_per_core // 8 # 8 bytes per element
    layers_per_core = elements_per_core // layer_size
    n_chunks = int(2**np.ceil(np.log2(nz // layers_per_core)))
    layers_per_chunk = nz // n_chunks
    chunk_size_elements = layers_per_chunk * layer_size
    chunk_size_bytes = chunk_size_elements * 8

    if args.verbose >= 1: print(f"""
        Reading metadata from {meta_filename}.
        volume_matching_shifts = {vm_shifts}
        Implant threshold {implant_threshold} -> {implant_threshold_u16} as uint16
        full_Nz,Ny,Nx    = {full_Nz,Ny,Nx}
        Nz               = {Nz}
        nz,ny,nx         = {nz,ny,nx}
        voxel_size       = {voxel_size}
        vmin,vmax        = {global_vmin,global_vmax}
        Layers per core  = {layers_per_core}
        Chunk size       = {chunk_size_bytes / 1024**2} MB
        Layers per chunk = {layers_per_chunk}
        Number of chunks = {n_chunks}
        Number of cores  = {n_cores}
    """)
    h5meta.close()

    if layers_per_chunk == 0 or layers_per_chunk >= nz:
        voxels = np.empty((nz,ny,nx), dtype=np.uint16)
        if args.verbose >= 1: print(f"Reading full volume {args.sample} at {args.sample_scale}x")
        load_slice(voxels, f"{binary_root}/voxels/{args.sample_scale}x/{args.sample}.uint16", (0,0,0), (nz,ny,nx))
        noisy_implant = (voxels > implant_threshold_u16)
        del voxels

        if args.verbose >= 1: print(f"Labelling connected components")
        label, n_features = ndi.label(noisy_implant, output=np.int64)
        if args.verbose >= 1: print (f"Found {n_features} connected components. Now finding largest.")
        bincnts           = np.bincount(label[label>0],minlength=n_features+1)
        largest_cc_ix     = np.argmax(bincnts)
        implant_mask      = (label == largest_cc_ix)
    else:
        if args.verbose >= 1: print(f"Labelling connected components in chunks to {intermediate_folder}")
        use_cache = False

        if use_cache:
            n_labels = np.fromfile(f"{intermediate_folder}/{args.sample}_n_labels.int64", dtype=np.int64)
        else:
            def label_chunk(i, chunk_size, chunk_prefix, implant_threshold_u16, global_shape):
                start = i*chunk_size
                end   = (i+1)*chunk_size if i < n_chunks-1 else nz # Last chunk gets the rest
                chunk_length = end-start
                voxel_chunk   = np.empty((chunk_length,ny,nx),dtype=np.uint16)
                load_slice(voxel_chunk, f"{binary_root}/voxels/{args.sample_scale}x/{args.sample}.uint16", (start,0,0), voxel_chunk.shape)
                noisy_implant = (voxel_chunk > implant_threshold_u16)
                del voxel_chunk
                label, n_features = ndi.label(noisy_implant, output=np.int64)
                label.tofile(f'{chunk_prefix}{i}.int64')
                del label
                return n_features

            start = datetime.datetime.now()
            with ThreadPool(n_cores) as pool:
                label_chunk_partial = partial(label_chunk, chunk_size=layers_per_chunk, chunk_prefix=f"{intermediate_folder}/{args.sample}_", implant_threshold_u16=implant_threshold_u16, global_shape=(nz,ny,nx))
                n_labels = pool.map(label_chunk_partial, range(n_chunks))
            end = datetime.datetime.now()
            flat_size = nz*ny*nx
            # load uint16, threshold (uint16 > uint8), label (int64), write int64
            total_bytes_processed = flat_size*2 + flat_size*2 + flat_size*8 + flat_size*8
            gb_per_second = total_bytes_processed / (end-start).total_seconds() / 1024**3
            print (f'Loading and labelling took {end-start}. (throughput: {gb_per_second:.02f} GB/s)')

            np.array(n_labels, dtype=np.int64).tofile(f"{intermediate_folder}/{args.sample}_n_labels.int64")

        if args.verbose >= 1: print (f"Finding largest connected component")
        implant_mask = np.zeros((nz,ny,nx), dtype=bool)
        largest_connected_component(implant_mask, f"{intermediate_folder}/{args.sample}_", n_labels, (nz,ny,nx), (layers_per_chunk,ny,nx), args.verbose)

    if args.verbose >= 2:
        print (f"Plotting middle planes to {plot_dir}")
        plot_middle_planes(implant_mask, plot_dir, args.sample)

    output_dir = f"{hdf5_root}/masks/{args.sample_scale}x/"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.verbose >= 1: print(f"Writing largest connected component to {output_dir}/{args.sample}.h5")

    update_hdf5_mask(f"{output_dir}/{args.sample}.h5",
                    group_name="implant",
                    datasets={'mask': implant_mask},
                    attributes={'scale': args.sample_scale, 'voxel_size': voxel_size,
                                'sample': args.sample, 'name': "implant_mask"})
