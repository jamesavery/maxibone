import matplotlib
matplotlib.use('Agg')
import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi, tqdm, matplotlib.pyplot as plt
sys.path.append(sys.path[0]+"/../")
from config.constants import osteocyte_*
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

sample, scale, chunk_size, verbose = commandline_args({"sample" : "<required>",
                                                           "scale" : 1,
                                                           "chunk_size" : 256,
                                                           "verbose" : 2})

# Define and create directories
plot_dir = f"{hdf5_root}/processed/osteocyt_mask/"
pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
mask_dir = f'{hdf5_root}/masks/{scale}x'
sample_path = f'{mask_dir}/{sample}.h5'

# Load the blood mask
with h5py.File(f"{sample_path}", 'r') as f:
    blood_mask = f['blood']['mask'][:]
    Nz, Ny, Nx = blood_mask.shape
    voxel_size = f['blood'].attrs['voxel_size']

# Load and threshold the soft tissue mask
voxels = np.memmap(f'{hdf5_root}/binary/segmented/gauss+edt/P0/{scale}x/{sample}.uint16', dtype='uint16', mode='r', shape=(Nz, Ny, Nx))
as_mask = voxels > 0
as_mask *= ~blood_mask

if verbose > 0:
    # Plot the debug images
    snz, sny, snx = max(10, Nz // 100), max(10, Ny // 100), max(10, Nx // 100)
    print(f"Saving osteocyt mask to {sample_path}")
    print(f'Plotting osteocyt mask to {plot_dir}/')
    hnz, hny, hnx = Nz//2, Ny//2, Nx//2
    plt.figure(figsize=(snx,sny)); plt.imshow(as_mask[hnz,:,:]); plt.savefig(f'{plot_dir}/{sample}_yx.png'); plt.clf()
    plt.figure(figsize=(snx,snz)); plt.imshow(as_mask[:,hny,:]); plt.savefig(f'{plot_dir}/{sample}_zx.png'); plt.clf()
    plt.figure(figsize=(sny,snz)); plt.imshow(as_mask[:,:,hnx]); plt.savefig(f'{plot_dir}/{sample}_zy.png'); plt.clf()

    yx = np.zeros((Ny, Nx, 3), dtype=np.uint8)
    yx[blood_mask[hnz] > 0] = [255, 0, 0]
    yx[as_mask[hnz] > 0] = [0, 255, 0]
    plt.figure(figsize=(snx,sny)); plt.imshow(yx); plt.savefig(f'{plot_dir}/{sample}_yx_overlay.png'); plt.clf()

    zx = np.zeros((Nz, Nx, 3), dtype=np.uint8)
    zx[blood_mask[:,hny] > 0] = [255, 0, 0]
    zx[as_mask[:,hny] > 0] = [0, 255, 0]
    plt.figure(figsize=(snx,snz)); plt.imshow(zx); plt.savefig(f'{plot_dir}/{sample}_zx_overlay.png'); plt.clf()

    zy = np.zeros((Nz, Ny, 3), dtype=np.uint8)
    zy[blood_mask[:,:,hnx] > 0] = [255, 0, 0]
    zy[as_mask[:,:,hnx] > 0] = [0, 255, 0]
    plt.figure(figsize=(sny,snz)); plt.imshow(zy); plt.savefig(f'{plot_dir}/{sample}_zy_overlay.png'); plt.clf()

# Save the mask
update_hdf5(sample_path,
            group_name='osteocyt',
            datasets={'mask': as_mask},
            attributes={
                'scale': scale,
                'voxel_size': voxel_size,
                'sample': sample,
                'name': 'osteocyt mask'
            })