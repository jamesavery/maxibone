#! /usr/bin/python3
'''
Script for matching volumes from the top and bottom of a multi-scan tomogram.
'''
import sys
sys.path.append(sys.path[0]+"/../")

from config.paths import hdf5_root
import h5py
import jax
import jax.numpy as jp
from lib.py.commandline_args import default_parser
import numpy as np
import os.path
import pathlib
from PIL import Image
import tqdm

# TODO:
# 1] shadow effects could perhaps be removed by histogram matching of final adjacent regions
#    - should probably be removed by kernel-operation around outer edge, before matching...

def match_region(voxels_top, voxels_bot, overlap, max_shift, verbose):
    '''
    Find shift that minimizes squared differences with `overlap <= shift <= max_shift`.

    Parameters
    ----------
    `voxels_top` : jp.array[Any]
        The top region to match.
    `voxels_bot` : jp.array[Any]
        The bottom region to match.
    `overlap` : int
        The overlap between the regions.
    `max_shift` : int
        The maximum shift to consider.
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `result` : Tuple[int, float]
        The shift that minimizes the squared differences and the corresponding error.
    '''

    # Shifts smaller than the overlap overlap with shift
    slice_size = voxels_top.shape[1] * voxels_top.shape[2] # Normalize by number of voxels (to make sums_lt and sums_ge comparable)
    rng_lt = range(2, overlap)
    iter_lt = tqdm.tqdm(rng_lt, f"Matchings shifted 0 to {overlap}") if verbose >= 1 else rng_lt
    sums_lt = jp.array( [ jp.sum(((voxels_top[-shift:] - voxels_bot[0:shift]) / (shift * slice_size))**2)
                          for shift in iter_lt] )

    # Shifts larger than the overlap overlap with overlap
    rng_ge = range(0, max_shift-overlap)
    iter_ge = tqdm.tqdm(rng_ge, f"Matchings shifted {overlap} to {max_shift}") if verbose >= 1 else rng_ge
    sums_ge = jp.array( [ jp.sum(((voxels_top[-overlap:] - voxels_bot[shift:shift+overlap]) / (overlap * slice_size))**2)
                          for shift in iter_ge] )

    if verbose >= 1:
        print("sums_lt=", sums_lt)
        print("sums_ge=", sums_ge)

    sums = jp.concatenate([sums_lt, sums_ge])
    result = jp.argmin( sums ), jp.sqrt(sums.min())

    del sums
    del sums_lt
    del sums_ge

    return result

def match_all_regions(voxels, sample, crossings, overlap, max_shift, verbose):
    '''
    Match all regions in a volume.

    Parameters
    ----------
    `voxels` : np.array[Any]
        The volume to match.
    `sample` : str
        The sample name. Used for naming output images. Isn't used if verbose < 2.
    `crossings` : np.array[int]
        The crossings between the regions.
    `overlap` : int
        The overlap between the regions.
    `max_shift` : int
        The maximum shift to consider.
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `shifts, errors` : Tuple[np.array[int], np.array[float]]
        The shifts that minimize the squared differences and the corresponding errors.
    '''

    shifts = np.zeros(len(crossings), dtype=np.int32)
    errors = np.zeros(len(crossings), dtype=np.float32)
    match_region_jit = jax.jit(match_region, static_argnums=(2,3,4))

    if verbose >= 1: print(f"Crossings at z-indices: {crossings}")
    for i in range(len(crossings)):
        crossing = crossings[i]
        if verbose >= 1: print(f"Processing crossing at z={crossing}:")

        if verbose >= 1: print(f"Reading top region:    voxels[{crossing-max_shift}:{crossing}]")
        top_voxels = jp.array(voxels[crossing-max_shift:crossing]).astype(jp.float32)

        if verbose >= 1: print(f"Reading bottom region: voxels[{crossing}:{crossing+max_shift}]")
        bot_voxels = jp.array(voxels[crossing:crossing+max_shift]).astype(jp.float32)
        if verbose >= 1: print(f"Matching regions (Shapes: {bot_voxels.shape} {top_voxels.shape})")

        shift, error = match_region_jit(top_voxels, bot_voxels, overlap, max_shift, verbose)
        shifts[i] = shift
        errors[i] = error
        if verbose >= 1: print(f"Optimal shift is {shift} with error {error} per voxel")

        if verbose >= 2:
            image_dir = f"{volume_matched_dir}/verification"
            pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)
            if verbose >= 1: print(f"Writing images of matched slices to {image_dir} to check correctness.")

            merged_zy_slice = np.concatenate([top_voxels[:,:,Nx//2], bot_voxels[shift:,:,Nx//2]])
            Image.fromarray(merged_zy_slice.astype(np.uint8)).save(f"{image_dir}/{sample}-{i}cross-zy.png")
            image = np.zeros((2*max_shift, 1980), dtype=np.uint8)
            image[:max_shift,:1980//2] = top_voxels[:,Ny//2-1980//4:Ny//2+1980//4,Nx//2].astype(np.uint8)
            shift = max(1, shift)
            image[max_shift-shift:-shift,1980//2:] = bot_voxels[:,Ny//2-1980//4:Ny//2+1980//4,Nx//2].astype(np.uint8)
            Image.fromarray(image).save(f"{image_dir}/{sample}-{i}bottop-zx.png")

        del top_voxels
        del bot_voxels

    return shifts, errors

def write_matched(voxels_in, voxels_out, crossings, shifts, verbose):
    '''
    Copy through the volume matched volume from the original.
    - general interface that works equally well for anything indexed like numpy arrays - including cupy, HDF5 and netCDF.

    Parameters
    ----------
    `voxels_in` : np.array[Any]
        The original volume.
    `voxels_out` : np.array[Any]
        The volume to write to.
    `crossings` : np.array[int]
        The crossings between the regions.
    `shifts` : np.array[int]
        The shifts that minimize the squared differences.
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `None`
    '''

    voxels_out[:crossings[0]] = voxels_in[:crossings[0]]
    cum_shifts = [0]+list(np.cumsum(shifts))
    crossings  = list(crossings) + [voxels_in.shape[0]]

    if verbose >= 1: print(f"Cumulative shifts: {cum_shifts}")
    if verbose >= 1: print(f"Duplicating subvolume 0: 0:{crossings[0]}")
    voxels_out[:crossings[0]] = voxels_in[:crossings[0]]

    for i in range(len(crossings)-1):
        if verbose >= 1: print(f"Duplicating unmatched part of subvolume {i+1}: voxels_out[{crossings[i]-cum_shifts[i]}:{crossings[i+1]-cum_shifts[i]-shifts[i]}] = voxels_in[{crossings[i]+shifts[i]}:{crossings[i+1]}]")
        voxels_out[crossings[i]-cum_shifts[i]:crossings[i+1]-cum_shifts[i]-shifts[i]] = voxels_in[crossings[i]+shifts[i]:crossings[i+1]]

def write_matched_hdf5(h5_filename_in, h5_filename_out, crossings, shifts, compression='lzf', verbose=0):
    '''
    Create and populate a volume matched HDF5-file from the original.

    Parameters
    ----------
    `h5_filename_in` : str
        The input HDF5 file.
    `h5_filename_out` : str
        The output HDF5 file.
    `crossings` : np.array[int]
        The crossings between the regions.
    `shifts` : np.array[int]
        The shifts that minimize the squared differences.
    `compression` : str
        The compression to use.
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `None`
    '''

    h5in  = h5py.File(h5_filename_in, "r")
    h5out = h5py.File(h5_filename_out,"w")
    voxels_in = h5in['voxels']

    (Nz,Ny,Nx) = voxels_in.shape
    matched_Nz = Nz - np.sum(shifts)
    voxels_out = h5out.create_dataset("voxels", (matched_Nz,Ny,Nx), dtype=voxels_in.dtype, compression=compression)
    write_matched(voxels_in, voxels_out, crossings, shifts, verbose)

    # TODO: Duplicate all metadata into new HDF5 (Nice to have, but can always be found in original HDF5)

if __name__ == "__main__":
    argparser = default_parser(description=__doc__)
    argparser.add_argument('overlap', action='store', type=int, default=10, nargs='?',
        help='The overlap between the regions. Default is 10.')
    argparser.add_argument('max_shift', action='store', type=int, default=150, nargs='?',
        help='The maximum shift to consider. Default is 150.')
    argparser.add_argument('--generate_h5', action='store_true',
        help='Toggles generating the HDF5 file with the matched volumes.')
    args = argparser.parse_args()

    volume_matched_dir = f"{hdf5_root}/processed/volume_matched"
    input_h5name  = f"{hdf5_root}/hdf5-byte/msb/{args.sample}.h5"
    output_h5name = f"{volume_matched_dir}/1x/{args.sample}.h5"

    outdir = os.path.dirname(output_h5name)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    h5file = h5py.File(input_h5name, "r+")
    voxels = h5file['voxels']
    subvolume_dimensions = h5file['subvolume_dimensions'][:]
    (Nz,Ny,Nx) = h5file['voxels'].shape

    crossings = np.cumsum(subvolume_dimensions[:-1,0]).astype(int)
    if args.verbose >= 1: print(f"Matching all regions for sample {args.sample} at crossings {crossings}.")
    shifts, errors = match_all_regions(voxels, args.sample, crossings, args.overlap, args.max_shift, verbose=args.verbose)

    np.save(f"{volume_matched_dir}/{args.sample}-shifts.npy", shifts)

    if ("volume_matching_shifts" not in h5file):
        h5file.create_dataset("volume_matching_shifts", data=shifts)
    else:
        h5file["volume_matching_shifts"][...] = shifts

    h5file.close()

    if (args.generate_h5):
        if args.verbose >= 1: print(f"Copying over volume from {input_h5name} shifted by {shifts} to {output_h5name}")
        write_matched_hdf5(input_h5name, output_h5name, crossings, shifts, verbose=args.verbose)
