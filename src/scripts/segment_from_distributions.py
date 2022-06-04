import sys
sys.path.append(sys.path[0]+"/../")
import cv2
import h5py
import pybind_kernels.histograms as histograms
import pybind_kernels.label as label
import numpy as np
from config.paths import binary_root, hdf5_root_fast as hdf5_root, commandline_args
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from helper_functions import block_info

debug = True

def load_block(sample, Ny, Nx, z_offset, block_size, field_names):
    Nfields = len(field_names)
    voxels = np.zeros((block_size,Ny,Nx),    dtype=np.uint16)
    fields = np.zeros((Nfields,block_size//2,Ny//2,Nx//2), dtype=np.uint16)
    histograms.load_slice(voxels, f'{binary_root}/voxels/1x/{sample}.uint16', (z_offset, 0, 0), (block_size, Ny, Nx))
    for i in range(Nfields):
        fi = np.load(f"{binary_root}/fields/implant-{field_names[i]}/2x/{sample}.npy", mmap_mode='r')
        fields[i,:] = fi[z_offset//2:z_offset//2 + block_size//2,:Ny//2,:Nx//2]

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
    sample, subbins, group, debug_output = commandline_args({'sample':'<required>', 'subbins': '<required>', 'group': 'otsu_seperation', 'debug_output': None})

    # Iterate over all subvolumes
    bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', block_size=0, n_blocks=0, z_offset=0)
    sz, sy, sx = bi['dimensions'][:3]
    fz, fy, fx = np.array((sz, sy, sx)) // 2
    axes_names = ["x", "y", "z", "r"]
    field_names = ["gauss", "edt", "gauss+edt"]

    for b in tqdm(range(bi['n_blocks']), desc='segmenting subvolumes'):
        probs_file = f'{hdf5_root}/processed/probabilities/{sample}-{subbins}{b}.h5'
        block_size = bi['subvolume_nzs'][b]
        zstart = bi['subvolume_starts'][b]
        zend = zstart + block_size
        fzstart, fzend = zstart // 2, zend // 2
        voxels, fields = load_block(sample, sy, sx, zstart, block_size, field_names)
        # These ranges shouldn't differ, but still let's be safe
        (vmin, vmax), (fmin, fmax) = load_value_ranges(probs_file, group)
        vranges = np.array([vmin, vmax, fmin, fmax], np.float32)

        for c in [0,1]:
            output_file = f'{hdf5_root}/processed/segmented/{sample}_{c}.bin'
            P_axes, P_fields = load_probabilities(probs_file, group, axes_names, field_names, c)
            n_probs = len(P_axes) + len(P_fields)
            result = np.zeros((bi['subvolume_nzs'][b],sy,sx), dtype=np.uint16)

            label.material_prob(
                voxels, fields,
                P_axes, 0b1111,
                P_fields, 0b111,
                np.array([1. / n_probs] * n_probs), # Average of all of the probabilities
                result,
                (vmin, vmax), (fmin, fmax),
                (zstart, 0, 0), (zend, sy, sx)
            )

            if debug:
                print (f'Segmentation has min {result.min()} and max {result.max()}')

            histograms.write_slice(result, zstart, output_file)