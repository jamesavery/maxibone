import matplotlib
matplotlib.use('Agg')
import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi, tqdm, matplotlib.pyplot as plt
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, hdf5_root_fast, binary_root
from lib.cpp.cpu_seq.io import load_slice, write_slice
from lib.cpp.cpu.general import bincount, where_in
from lib.cpp.cpu.geometry import center_of_masses, inertia_matrices, outside_ellipsoid
from lib.cpp.cpu.connected_components import largest_connected_component
from scipy import ndimage as ndi
from lib.py.helpers import block_info, commandline_args, update_hdf5
from lib.py.resample import downsample2x, downsample3x
import cupy as cp
import multiprocessing as mp
import datetime
from functools import partial
import PIL.Image

sample, scale, chunk_size, verbose = commandline_args({"sample" : "<required>",
                                                           "scale" : 1,
                                                           "chunk_size" : 256,
                                                           "verbose" : 2})

# Define and create directories
plot_dir = f"{hdf5_root}/processed/osteocyt_mask/{scale}x"
pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
mask_dir = f'{hdf5_root}/masks/{scale}x'
sample_path = f'{mask_dir}/{sample}.h5'

# Load the blood mask
with h5py.File(f"{sample_path}", 'r') as f:
    blood_mask = f['blood']['mask'][:]
    Nz, Ny, Nx = blood_mask.shape
    voxel_size = f['blood'].attrs['voxel_size']
    voxel_volume = voxel_size**3

if voxel_volume > osteocyte_Vmax:
    raise ValueError(f"Voxel volume for scale {scale} is {voxel_volume}, which is larger than the maximum osteocyte volume {osteocyte_Vmax}. Please run on a finer scale.")

# Load and threshold the soft tissue mask
voxels = np.memmap(f'{hdf5_root}/binary/segmented/gauss+edt/P0/{scale}x/{sample}.uint16', dtype='uint16', mode='r', shape=(Nz, Ny, Nx))
as_mask = voxels > 0
as_mask *= ~blood_mask

# Label the potential osteocytes
hole_id, num_holes = ndi.label(as_mask, output=np.uint64) # TODO out-of-core
if verbose > 0:
    print (f"Found {num_holes} potential osteocytes")

# Compute the volumes and sort out unrealistic osteocytes
volumes = np.zeros(num_holes+1, dtype=np.uint64)
bincount(hole_id, volumes)
#volumes = np.bincount(hole_id.ravel()) * voxel_volume # index 0 is background
volumes = volumes * voxel_volume
small_unknown = volumes < osteocyte_Vmin
large_unknown = volumes > osteocyte_Vmax
osteocyte_sized = (volumes >= osteocyte_Vmin) & (volumes <= osteocyte_Vmax)
if verbose > 0:
    print (f"Found {np.sum(small_unknown)} small and {np.sum(large_unknown)} large osteocytes")
    print (f"Found {np.sum(osteocyte_sized)} potential osteocytes")

# Compute ellipsoid fits
cms = np.zeros((num_holes+1, 3), dtype=np.float32)
center_of_masses(hole_id, cms)
ims = np.zeros((num_holes+1, 3, 3), dtype=np.float32)
inertia_matrices(hole_id, cms, ims)
principal_lambdas = np.linalg.eigvals(ims)
abc = 1 / np.sqrt(principal_lambdas)
a, b, c = abc.T
weirdly_long = (a / c) > 3
if verbose > 0:
    print (f"Found {np.sum(weirdly_long)} weirdly long osteocytes")


# Test that the osteocytes are not too different from the best ellipsoid
ellipsoid_errors = np.zeros(num_holes+1, dtype=np.uint64)
ellipsoid_volumes = (4/3) * np.pi * a * b * c
outside_ellipsoid(hole_id, cms, abc, ellipsoid_errors)
ellipsoid_error_threshold = 0.5
weirdly_shaped = (ellipsoid_errors / ellipsoid_volumes) > ellipsoid_error_threshold
if verbose > 0:
    print (f"Found {np.sum(weirdly_shaped)} weirdly shaped osteocytes")

# Final osteocyte segmentation
osteocyte_segments = np.argwhere(osteocyte_sized & (~weirdly_long) & (~weirdly_shaped)).flatten().astype(np.uint64)
osteocyte_mask = hole_id.copy()
if verbose > 0:
    print (f"Found {len(osteocyte_segments)} osteocytes")

where_in(osteocyte_mask, osteocyte_segments)

if verbose > 0:
    # Plot the debug images
    snz, sny, snx = max(10, Nz // 100), max(10, Ny // 100), max(10, Nx // 100)
    print(f"Saving osteocyt mask to {sample_path}")
    print(f'Plotting osteocyt mask to {plot_dir}/')
    hnz, hny, hnx = Nz//2, Ny//2, Nx//2
    plt.figure(figsize=(snx,sny)); plt.imshow(osteocyte_mask[hnz,:,:]); plt.savefig(f'{plot_dir}/{sample}_yx.png'); plt.clf()
    plt.figure(figsize=(snx,snz)); plt.imshow(osteocyte_mask[:,hny,:]); plt.savefig(f'{plot_dir}/{sample}_zx.png'); plt.clf()
    plt.figure(figsize=(sny,snz)); plt.imshow(osteocyte_mask[:,:,hnx]); plt.savefig(f'{plot_dir}/{sample}_zy.png'); plt.clf()

    yx = np.zeros((Ny, Nx, 3), dtype=np.uint8)
    yx[as_mask[hnz] > 0] = [255, 255, 0]
    yx[blood_mask[hnz] > 0] = [255, 0, 0]
    yx[osteocyte_mask[hnz] > 0] = [0, 255, 0]
    PIL.Image.fromarray(yx).save(f'{plot_dir}/{sample}_yx_overlay.png')

    zx = np.zeros((Nz, Nx, 3), dtype=np.uint8)
    zx[as_mask[:,hny] > 0] = [255, 255, 0]
    zx[blood_mask[:,hny] > 0] = [255, 0, 0]
    zx[osteocyte_mask[:,hny] > 0] = [0, 255, 0]
    PIL.Image.fromarray(zx).save(f'{plot_dir}/{sample}_zx_overlay.png')

    zy = np.zeros((Nz, Ny, 3), dtype=np.uint8)
    zy[as_mask[:,:,hnx] > 0] = [255, 255, 0]
    zy[blood_mask[:,:,hnx] > 0] = [255, 0, 0]
    zy[osteocyte_mask[:,:,hnx] > 0] = [0, 255, 0]
    PIL.Image.fromarray(zy).save(f'{plot_dir}/{sample}_zy_overlay.png')

# Save the mask
update_hdf5(sample_path,
            group_name='osteocyt',
            datasets={'mask': osteocyte_mask},
            attributes={
                'scale': scale,
                'voxel_size': voxel_size,
                'sample': sample,
                'name': 'osteocyt mask'
            })