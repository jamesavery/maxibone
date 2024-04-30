import matplotlib
matplotlib.use('Agg')
import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi, tqdm, matplotlib.pyplot as plt
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, hdf5_root_fast, binary_root
from lib.cpp.cpu_seq.io import load_slice, write_slice
from lib.cpp.cpu.connected_components import largest_connected_component
from scipy import ndimage as ndi
from lib.py.helpers import block_info, commandline_args, update_hdf5
from lib.py.resample import downsample2x, downsample3x
import cupy as cp
import multiprocessing as mp
import datetime
from functools import partial

sample, m, scheme, chunk_size, verbose = commandline_args({"sample" : "<required>",
                                                           "material" : 0,
                                                           "scheme" : "edt",
                                                           "chunk_size" : 256,
                                                           "verbose" : 2})

#scales = [32, 16, 8, 4]#, 2, 1]
scales = [1]

bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5')
Nz, Ny, Nx, _ = bi["dimensions"]

#Du skal til at køre den nye cc, saa vi kan faa et 1x blood mask!

for scale in tqdm.tqdm(scales, desc= 'Computing connected components'):
    data = f'{binary_root}/segmented/{scheme}/P{m}/{scale}x/{sample}.uint16'
    output_dir = f'{hdf5_root}/masks/{scale}x'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    nz, ny, nx = Nz // scale, Ny // scale, Nx // scale
    voxel_size = bi["voxel_size"]*scale

    if verbose >= 1:
        plot_dir = f"{hdf5_root}/processed/blood_mask/"
        pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

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

    if layers_per_chunk == 0 or layers_per_chunk >= nz:
        voxels = np.empty((nz, ny, nx), dtype=np.uint16)
        load_slice(voxels, data, (0,0,0), (nz, ny, nx))
        label, n_features = ndi.label(voxels)
        counts = np.bincount(label[label>0], minlength=n_features+1)
        largest_cc = np.argmax(counts)
        mask = (label == largest_cc)
    else:
        intermediate_folder = f"/tmp/maxibone/labels_blood/{scale}x/"
        os.makedirs(intermediate_folder, exist_ok=True)

        def label_chunk(i, chunk_size, chunk_prefix, global_shape):
            start = i*chunk_size
            end   = (i+1)*chunk_size if i < n_chunks-1 else nz # Last chunk gets the rest
            chunk_length = end-start
            voxel_chunk   = np.empty((chunk_length,ny,nx),dtype=np.uint16)
            load_slice(voxel_chunk, data, (start,0,0), voxel_chunk.shape)
            if verbose >= 3:
                plt.imshow(voxel_chunk[chunk_length//2,:,:]); plt.savefig(f"{plot_dir}/{sample}_{scale}_{i}_yx.png"); plt.clf()
                plt.imshow(voxel_chunk[:,ny//2,:]); plt.savefig(f"{plot_dir}/{sample}_{scale}_{i}_zx.png"); plt.clf()
                plt.imshow(voxel_chunk[:,:,nx//2]); plt.savefig(f"{plot_dir}/{sample}_{scale}_{i}_zy.png"); plt.clf()
            label, n_features = ndi.label(voxel_chunk, output=np.int64)
            del voxel_chunk
            label.tofile(f'{chunk_prefix}{i}.int64')
            del label
            return n_features

        start = datetime.datetime.now()
        with mp.Pool(n_cores) as pool:
            label_chunk_partial = partial(label_chunk, chunk_size=layers_per_chunk, chunk_prefix=f"{intermediate_folder}/{sample}_", global_shape=(nz,ny,nx))
            n_labels = pool.map(label_chunk_partial, range(n_chunks))
        end = datetime.datetime.now()
        flat_size = nz*ny*nx
        # load uint16, threshold (uint16 > uint8), label (int64), write int64
        total_bytes_processed = flat_size*2 + flat_size*2 + flat_size*8 + flat_size*8
        gb_per_second = total_bytes_processed / (end-start).total_seconds() / 1024**3
        print (f'Loading and labelling took {end-start}. (throughput: {gb_per_second:.02f} GB/s)')

        np.array(n_labels, dtype=np.int64).tofile(f"{intermediate_folder}/{sample}_n_labels.int64")

        mask = np.zeros((nz,ny,nx),dtype=bool)
        largest_connected_component(mask, f"{intermediate_folder}/{sample}_", n_labels, (nz,ny,nx), (layers_per_chunk,ny,nx), True)

    if verbose >= 1:
        print(f"Writing mask to {output_dir}/{sample}.h5")
        plt.imshow(mask[nz//2,:,:]); plt.savefig(f"{plot_dir}/{sample}_{scale}_yx.png"); plt.clf()
        plt.imshow(mask[:,ny//2,:]); plt.savefig(f"{plot_dir}/{sample}_{scale}_zx.png"); plt.clf()
        plt.imshow(mask[:,:,nx//2]); plt.savefig(f"{plot_dir}/{sample}_{scale}_zy.png"); plt.clf()

    update_hdf5(f"{output_dir}/{sample}.h5",
                group_name=f"blood",
                datasets={'mask':mask},
                attributes={'scale':scale,'voxel_size':voxel_size,
                            'sample':sample, 'name':"blood_mask"})
