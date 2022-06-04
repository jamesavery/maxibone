import sys
sys.path.append(sys.path[0]+"/../")
import cv2
import h5py
import pybind_kernels.histograms as histograms
import pybind_kernels.label as label
import numpy as np
from config.paths import binary_root, hdf5_root_fast as hdf5_root
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

debug = True

def load_block(sample, offset, block_size, field_names):
    Nfields = len(field_names)
    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    Nz, Ny, Nx = dm['voxels'].shape
    Nz -= np.sum(dm["volume_matching_shifts"][:])
    dm.close()
    block_size       = min(block_size, Nz-offset[0])
    voxels = np.zeros((block_size,Ny,Nx),    dtype=np.uint16)
    fields = np.zeros((Nfields,block_size//2,Ny//2,Nx//2), dtype=np.uint16)
    histograms.load_slice(voxels, f'{binary_root}/voxels/1x/{sample}.uint16', (offset[0], 0, 0), (Nz, Ny, Nx))
    voxels = voxels[:,:,:]
    for i in range(Nfields):
        fi = np.load(f"{binary_root}/fields/implant-{field_names[i]}/2x/{sample}.npy", mmap_mode='r')
        fields[i,:] = fi[offset[0]//2:offset[0]//2 + block_size//2,:Ny//2,:Nx//2]

    return voxels, fields

def load_probabilities(path, group, axes_names, field_names, c):
    with h5py.File(path, 'r') as f:
        grp = f[group]
        P_axes = [grp[f'{name}_bins_c{c}'][:,:] for name in axes_names]
        P_fields = [grp[f'field_bins_{name}_c{c}'][:,:] for name in field_names]
    return P_axes, P_fields

def load_value_ranges(path, group):
    with h5py.File(path, 'r') as f:
        f.require_group(group)
        a = f[group]['value_ranges']
        return list(a)

def nblocks(size, block_size):
    return (size // block_size) + (1 if size % block_size > 0 else 0)

if __name__ == '__main__':
    # TODO Don't hardcode
    sample = '770c_pag'
    subbins = 'bone_region2'
    block_size = 64
    z_offset = 2000
    blocks = 1 #nblocks(sz, block_size)
    probs_file = f'{hdf5_root}/processed/probabilities/{sample}-{subbins}.h5'
    group = 'otsu_seperation'

    (vmin, vmax), (fmin, fmax) = load_value_ranges(probs_file, group)
    vranges = np.array([vmin, vmax, fmin, fmax], np.float32)

    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    sz, sy, sx = dm['voxels'].shape
    fz, fy, fx = np.array((sz, sy, sx)) // 2
    dm.close()

    axes_names = ["x", "y", "z", "r"]
    field_names = ["gauss", "edt", "gauss+edt"]

    for c in {0}:
        P_axes, P_fields = load_probabilities(probs_file, group, axes_names, field_names, c)
        n_probs = len(P_axes) + len(P_fields)
        if debug:
            print ([P.min() for P in P_axes], [P.max() for P in P_axes], [P.min() for P in P_fields], [P.max() for P in P_fields])

        for i in tqdm(range(blocks), desc='Computing the probability distributions'):
            voxels = np.zeros((block_size, sy, sx), np.uint16)
            field = np.zeros((block_size//2, fy, fx), np.uint16)
            zstart, zstop = i*block_size + z_offset, min((i+1)*block_size + z_offset, sz)
            voxels, fields = load_block(sample, (zstart, 0, 0), block_size, field_names)
            fzstart, fzstop = i*(block_size//2), min((i+1)*(block_size//2), fz)
            ranges = np.array([0, block_size, 0, sy, 0, sx], np.uint64)
            result = np.zeros((block_size,sy,sx), dtype=np.uint16)

            label.material_prob(
                voxels, fields,
                P_axes, 0b1111,
                P_fields, 0b111,
                np.array([1. / n_probs] * n_probs), # Average of all of the probabilities
                result,
                (vmin, vmax), (fmin, fmax),
                (zstart, 0, 0), (zstop, sy, sx)
            )

            if debug:
                print (f'Segmentation has min {result.min()} and max {result.max()}')

            np.save(f'partials/c{c}_{i}', result)