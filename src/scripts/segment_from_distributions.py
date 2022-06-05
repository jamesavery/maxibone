import os, sys, pathlib, h5py, numpy as np
sys.path.append(sys.path[0]+"/../")
import pybind_kernels.histograms as histograms
import pybind_kernels.label as label
from config.paths import binary_root, hdf5_root_fast as hdf5_root, commandline_args
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from helper_functions import block_info, load_block

debug = True


def load_probabilities(path, group, axes_names, field_names, c):
    with h5py.File(path, 'r') as prob_file:
        P_axes   = [prob_file[f'{group}/{name}/c{c}'][:,:] for name in axes_names]
        P_fields = [prob_file[f'{group}/{name}/c{c}'][:,:] for name in field_names]
    return P_axes, P_fields

def load_value_ranges(path, group):
    print(f"Reading value_ranges from {group} in {path}\n")    
    with h5py.File(path, 'r') as f:
#        f.require_group(group) # require_group betyder "opret gruppe hvis den ikke findes, overskriv hvis den allerede findes". Det er ikke et tjek.
        a = f[group]['value_ranges']
        return list(a)

def nblocks(size, block_size):
    return (size // block_size) + (1 if size % block_size > 0 else 0)

if __name__ == '__main__': 
    sample, block_start, n_blocks, region, group, debug_output = commandline_args({'sample':'<required>',
                                                                                   "block_start":0,
                                                                                   "n_blocks":0,
                                                                                   'region': 'bone_region',
                                                                                   'group': 'otsu_separation',
                                                                                   'debug_output': None})

    # Iterate over all subvolumes
    bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', block_size=0, n_blocks=n_blocks, z_offset=block_start)
    Nz, Ny, Nx = bi['dimensions'][:3]
    fz, fy, fx = np.array((Nz, Ny, Nx)) // 2
    axes_names =  []     # ["x", "y", "z", "r"] # For later
    field_names = ["edt"]# ["gauss", "edt", "gauss+edt"] 

    probs_file = f'{hdf5_root}/processed/probabilities/{sample}.h5'    
    for b in tqdm(range(block_start,block_start+bi['n_blocks']), desc='segmenting subvolumes'):
        group_name = f"{group}/{region}{b}/"
        block_size = bi['subvolume_nzs'][b]
        zstart = bi['subvolume_starts'][b]
        zend = zstart + block_size
        fzstart, fzend = zstart // 2, zend // 2
        mask_scale = 8
        voxels, fields = load_block(sample, zstart, block_size, region, mask_scale, field_names)
        # These ranges shouldn't differ, but still let's be safe
        (vmin, vmax), (fmin, fmax) = load_value_ranges(probs_file, group_name)
        vranges = np.array([vmin, vmax, fmin, fmax], np.float32)

        for c in [0,1]:
            output_dir = f'{binary_root}/segmented/1x/'
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            P_axes, P_fields = load_probabilities(probs_file, group_name, axes_names, field_names, c)
            n_probs = len(P_axes) + len(P_fields)
            result = np.zeros((bi['subvolume_nzs'][b],Ny,Nx), dtype=np.uint16)

            label.material_prob_justonefieldthx(voxels,fields[0],P_fields[0],result,
                                                (vmin,vmax),(fmin,fmax),
                                                (zstart,0,0), (zend,Ny,Nx));
            # label.material_prob(
            #     voxels, fields,
            #     P_axes, 0,#0b1111,
            #     P_fields, 0b01,#0b111,
            #     np.array([1. / n_probs] * n_probs), # Average of all of the probabilities
            #     result,
            #     (vmin, vmax), (fmin, fmax),
            #     (zstart, 0, 0), (zend, sy, sx)
            # )

            if debug:
                print (f'Segmentation has min {result.min()} and max {result.max()}')

            
#            histograms.write_slice(result, zstart, output_file)
