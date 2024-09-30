#! /usr/bin/python3
'''
This script segments the subvolumes using the probabilities computed in the previous step.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')

from config.paths import binary_root, hdf5_root as hdf5_root
import h5py
from lib.cpp.cpu.io import write_slice
from lib.cpp.gpu.label import material_prob_justonefieldthx
from lib.py.commandline_args import add_volume, default_parser
from lib.py.helpers import chunk_info, load_chunk, plot_middle_planes
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm

def load_probabilities(path, group, axes_names, field_names, m, verbose):
    '''
    Load the probabilities from the HDF5 file.

    Parameters
    ----------
    `path` : str
        The path to the HDF5 file.
    `group` : str
        The group in the HDF5 file.
    `axes_names` : list[str]
        The names of the axes.
    `field_names` : list[str]
        The names of the fields.
    `m` : int
        The index of the probability to load. 0 is currently soft tissue, 1 is bone.
    `verbose` : int
        The verbosity level of the script.

    Returns
    -------
    `P_axes` : list[numpy.array[float32]]
        The probabilities of the axes.
    `P_fields` : list[numpy.array[float32]]
        The probabilities of the fields.
    '''

    if verbose >= 2: print(f"Reading probabilities from {group} in {path}\n")

    try:
        prob_file = h5py.File(path, 'r')
        P_axes    = [prob_file[f'{group}/{name}/P{m}'][:,:] for name in axes_names]
        P_fields  = [prob_file[f'{group}/field_bins_{name}/P{m}'][:,:] for name in field_names]
        prob_file.close()
    except Exception as e:
        print(f"Couldn't load {group}/{axes_names}|{field_names}/P{m} from {path}: {e}")
        sys.exit(-1)

    return P_axes, P_fields

def load_value_ranges(path, group, verbose):
    '''
    Load the value ranges from the HDF5 file.

    Parameters
    ----------
    `path` : str
        The path to the HDF5 file.
    `group` : str
        The group in the HDF5 file.
    `verbose` : int
        The verbosity level of the script.

    Returns
    -------
    `value_ranges` : numpy.array[int]
        The value ranges.
    '''

    if verbose >= 2: print(f"Reading value_ranges from {group} in {path}\n")

    try:
        f = h5py.File(path, 'r')
    except Exception as e:
        print(f"Couldn't load {group}/value_ranges from {path}: {e}")
        sys.exit(-1)

    return f[group]['value_ranges'][:].astype(int)


def nchunks(size, chunk_size):
    '''
    Calculate the number of chunks needed to cover a `size`.

    Parameters
    ----------
    `size` : int
        The size to cover.
    `chunk_size` : int
        The size of the chunks.

    Returns
    -------
    `n_chunks` : int
        The number of chunks needed
    '''

    return (size + chunk_size - 1) // chunk_size

if __name__ == '__main__':
    argparser = default_parser(__doc__)
    argparser = add_volume(argparser, 'field', 2)
    argparser = add_volume(argparser, 'mask', 8)
    argparser.add_argument('--chunk-start', action='store', type=int, default=0, nargs='?',
        help='The starting chunk to process. Default is 0.')
    argparser.add_argument('--group', action='store', type=str, default='otsu_separation',
        help='The group in the HDF5 file to load the probabilities from. Default is "otsu_separation".')
    args = argparser.parse_args()

    # TODO scale may not trickle down correctly
    # Iterate over all subvolumes
    bi = chunk_info(f'{hdf5_root}/hdf5-byte/msb/{args.sample}.h5', args.sample_scale, chunk_size=args.chunk_size, n_chunks=0, z_offset=args.chunk_start, verbose=args.verbose)
    Nz, Ny, Nx = bi['dimensions'][:3]
    axes_names =  [] # ["x", "y", "z", "r"] # For later when we also use axes.
    field_names = [args.field] # TODO: We are currently only using one field.

    probs_file = f'{hdf5_root}/processed/probabilities/{args.sample}.h5'
    block_rng = range(args.chunk_start, args.chunk_start+bi['n_chunks'])
    block_iter = tqdm(block_rng, desc='segmenting subvolumes') if args.verbose >= 1 else block_rng
    for b in block_iter:
        group_name = f"{args.group}/{args.mask}/"

        chunk_size = bi['chunk_size']
        zstart = b*chunk_size
        zend = min(zstart + chunk_size, Nz)

        voxels, fields = load_chunk(args.sample, 1, zstart, chunk_size, args.mask, args.mask_scale, field_names, args.field_scale, verbose=args.verbose)
        (vmin, vmax), (fmin, fmax) = load_value_ranges(probs_file, group_name, args.verbose)

        this_z = zend - zstart
        zmid = this_z // 2
        if args.verbose >= 1:
            plot_dir = f'{hdf5_root}/processed/segmentation/{args.sample}'
            pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
            combined_yx = np.zeros((Ny,Nx,3), dtype=np.uint8)
            combined_zy = np.zeros((this_z,Ny,3), dtype=np.uint8)
            combined_zx = np.zeros((this_z,Nx,3), dtype=np.uint8)

            plot_middle_planes(voxels, plot_dir, f'{b}_voxels', verbose=args.verbose)
            plot_middle_planes(fields[0], plot_dir, f'{b}_fields_{args.field}', verbose=args.verbose)

        for m in [0,1]:
            output_dir  = f'{binary_root}/segmented/{args.field}/P{m}/1x/'
            output_file = f"{output_dir}/{args.sample}.uint16";
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            P_axes, P_fields = load_probabilities(probs_file, group_name, axes_names, field_names, m, args.verbose)
            n_probs = len(P_axes) + len(P_fields)
            result = np.zeros((zend-zstart,Ny,Nx), dtype=np.uint16)

            material_prob_justonefieldthx(voxels,fields[0], P_fields[0], result,
                                                (vmin,vmax), (fmin,fmax),
                                                (zstart,0,0), (zend,Ny,Nx),
                                                args.verbose)

            if args.verbose >= 1:
                red = [255,0,0]
                yellow = [255,255,0]
                combined_yx[result[zmid,:,:]  > 0] = red if m == 0 else yellow
                combined_zx[result[:,Ny//2,:] > 0] = red if m == 0 else yellow
                combined_zy[result[:,:,Nx//2] > 0] = red if m == 0 else yellow

                plot_middle_planes(result, plot_dir, f'{b}_{args.field}_P{m}', verbose=args.verbose)

            if args.verbose >= 2: print (f'Segmentation has min {result.min()} and max {result.max()}')

            if args.verbose >= 2: print(f"Writing results from block {b}")
            write_slice(result, output_file, (zstart,0,0), result.shape)

        if args.verbose >= 1:
            # Draw two plots in one, one above and one below
            fig = plt.figure(figsize=(34,34*2))
            fig.subplots_adjust(wspace=0, hspace=0)
            plt.subplot(2,1,1)
            plt.imshow(combined_yx, interpolation='none')
            plt.subplot(2,1,2)
            plt.imshow(voxels[zmid,:,:], interpolation='none')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{b}_{args.field}_combined_yx.pdf', bbox_inches='tight')
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
            plt.savefig(f'{plot_dir}/{b}_{args.field}_combined_zx.pdf', bbox_inches='tight')
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
            plt.savefig(f'{plot_dir}/{b}_{args.field}_combined_zy.pdf', bbox_inches='tight')
            fig.clear()
            plt.clf()
            plt.cla()
