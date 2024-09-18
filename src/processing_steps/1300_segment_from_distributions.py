import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')
from config.paths import binary_root, hdf5_root as hdf5_root
import h5py
from lib.cpp.cpu.io import write_slice
from lib.cpp.gpu.label import material_prob_justonefieldthx
from lib.py.helpers import block_info, commandline_args, load_block, plot_middle_planes
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm

def load_probabilities(path, group, axes_names, field_names, m):
    if verbose >= 1: print(f"Reading probabilities from {group} in {path}\n")

    try:
        prob_file = h5py.File(path, 'r')
        P_axes    = [prob_file[f'{group}/{name}/P{m}'][:,:] for name in axes_names]
        P_fields  = [prob_file[f'{group}/field_bins_{name}/P{m}'][:,:] for name in field_names]
        prob_file.close()
    except Exception as e:
        print(f"Couldn't load {group}/{axes_names}|{field_names}/P{m} from {path}: {e}")
        sys.exit(-1)

    return P_axes, P_fields

def load_value_ranges(path, group):
    if verbose >= 1: print(f"Reading value_ranges from {group} in {path}\n")

    try:
        f = h5py.File(path, 'r')
    except Exception as e:
        print(f"Couldn't load {group}/value_ranges from {path}: {e}")
        sys.exit(-1)

    return f[group]['value_ranges'][:].astype(int)


def nblocks(size, block_size):
    return (size // block_size) + (1 if size % block_size > 0 else 0)

if __name__ == '__main__':
    sample, scale, block_start, block_size, region_mask, group, mask_scale, scheme, field_scale, verbose = commandline_args({
        'sample' : '<required>',
        'scale' : 1,
        "block_start" : 0,
        "block_size" : 0,
        'region_mask' :  'bone_region',
        'group' :  'otsu_separation',
        'mask_scale' :  8,
        'scheme' : "gauss+edt", #MIDLERTIDIG
        'field_scale' : 2,
        'verbose' : 1
    })

    # TODO scale may not trickle down correctly
    # Iterate over all subvolumes
    bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', scale, block_size=block_size, n_blocks=0, z_offset=block_start)
    Nz, Ny, Nx = bi['dimensions'][:3]
    axes_names =  [] # ["x", "y", "z", "r"] # For later when we also use axes.
    field_names = [scheme] # TODO: We are currently only using one field.

    probs_file = f'{hdf5_root}/processed/probabilities/{sample}.h5'
    for b in tqdm(range(block_start,block_start+bi['n_blocks']), desc='segmenting subvolumes'):
        group_name = f"{group}/{region_mask}/"

        block_size = bi['block_size']
        zstart = b*block_size
        zend = min(zstart + block_size, Nz)

        voxels, fields = load_block(sample, 1, zstart, block_size, region_mask, mask_scale, field_names, field_scale)
        (vmin, vmax), (fmin, fmax) = load_value_ranges(probs_file, group_name)

        this_z = zend - zstart
        zmid = this_z // 2
        if verbose >= 1:
            plot_dir = f'{hdf5_root}/processed/segmentation/{sample}'
            pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
            combined_yx = np.zeros((Ny,Nx,3), dtype=np.uint8)
            combined_zy = np.zeros((this_z,Ny,3), dtype=np.uint8)
            combined_zx = np.zeros((this_z,Nx,3), dtype=np.uint8)

            plot_middle_planes(voxels, plot_dir, f'{b}_voxels')
            plot_middle_planes(fields[0], plot_dir, f'{b}_fields_{scheme}')

        for m in [0,1]:
            output_dir  = f'{binary_root}/segmented/{scheme}/P{m}/1x/'
            output_file = f"{output_dir}/{sample}.uint16";
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            P_axes, P_fields = load_probabilities(probs_file, group_name, axes_names, field_names, m)
            n_probs = len(P_axes) + len(P_fields)
            result = np.zeros((zend-zstart,Ny,Nx), dtype=np.uint16)

            material_prob_justonefieldthx(voxels,fields[0],P_fields[0],result,
                                                (vmin,vmax),(fmin,fmax),
                                                (zstart,0,0), (zend,Ny,Nx));

            if verbose >= 1:
                red = [255,0,0]
                yellow = [255,255,0]
                combined_yx[result[zmid,:,:]  > 0] = red if m == 0 else yellow
                combined_zx[result[:,Ny//2,:] > 0] = red if m == 0 else yellow
                combined_zy[result[:,:,Nx//2] > 0] = red if m == 0 else yellow

                plot_middle_planes(result, plot_dir, f'{b}_{scheme}_P{m}')

            if verbose >= 2: print (f'Segmentation has min {result.min()} and max {result.max()}')

            if verbose >= 1: print(f"Writing results from block {b}")
            write_slice(result, output_file, (zstart,0,0), result.shape)

        if verbose >= 1:
            # Draw two plots in one, one above and one below
            fig = plt.figure(figsize=(34,34*2))
            fig.subplots_adjust(wspace=0, hspace=0)
            plt.subplot(2,1,1)
            plt.imshow(combined_yx, interpolation='none')
            plt.subplot(2,1,2)
            plt.imshow(voxels[zmid,:,:], interpolation='none')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{b}_{scheme}_combined_yx.pdf', bbox_inches='tight')
            fig.clear()
            plt.clf()
            plt.cla()

            fig = plt.figure(figsize=(34,10))
            fig.subplots_adjust(wspace=0, hspace=0)
            plt.subplot(2,1,1)
            plt.imshow(combined_zx, interpolation='none')
            plt.subplot(2,1,2)
            plt.imshow(voxels[:,Ny//2,:], interpolation='none')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{b}_{scheme}_combined_zx.pdf', bbox_inches='tight')
            fig.clear()
            plt.clf()
            plt.cla()

            fig = plt.figure(figsize=(34,10))
            fig.subplots_adjust(wspace=0, hspace=0)
            plt.subplot(2,1,1)
            plt.imshow(combined_zy, interpolation='none')
            plt.subplot(2,1,2)
            plt.imshow(voxels[:,:,Nx//2], interpolation='none')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{b}_{scheme}_combined_zy.pdf', bbox_inches='tight')
            fig.clear()
            plt.clf()
            plt.cla()
