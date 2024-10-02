#! /usr/bin/python3
'''
Converts a 16-bit HDF5 file to a 16-bit binary file. The binary file can be read faster than the HDF5 file, if it is stored on a fast disk.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.paths import hdf5_root, binary_root
import h5py
from lib.cpp.cpu_seq.io import write_slice
from lib.py.commandline_args import default_parser
from lib.py.helpers import update_hdf5
import numpy as np
from tqdm import tqdm

slice_all = slice(None)

def h5tobin(sample, region=(slice_all,slice_all,slice_all), verbose=0):
    '''
    Convert a 16-bit HDF5 file to a 16-bit binary file.
    Read/write a full subvolume at a time.
    For each subvolume, correct for subvolume_matching_shifts.

    Parameters
    ----------
    `sample` : str
        The sample name.
    `region` : Tuple[slice, slice, slice]
        The region to extract from the HDF5 file. Default is to extract the full volume.

    Returns
    -------
    None
    '''

    msb_file   = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    lsb_file   = h5py.File(f'{hdf5_root}/hdf5-byte/lsb/{sample}.h5', 'r')
    dmsb, dlsb = msb_file['voxels'], lsb_file['voxels']
    Nz, Ny, Nx = dmsb.shape

    pathlib.Path(f"{binary_root}/voxels/1x/").mkdir(parents=True, exist_ok=True)
    outfile = f'{binary_root}/voxels/1x/{sample}.uint16'

    subvolume_dims = msb_file['subvolume_dimensions'][:]
    vm_shifts      = msb_file['volume_matching_shifts'][:]
    Nvols          = len(subvolume_dims)
    Nzs            = subvolume_dims[:,0]

    # TODO Rather than just choosing one of the volumes as the reference in the overlapping regions, make a smooth weighted average based on how "close" each plane is to each of the volumes.

    # The ith subvolume in input  starts at sum(Nzs[:(i-1)]) and ends at sum(Nzs[:i])
    # The ith subvolume in output starts at sum(Nzs[:(i-1)]) - sum(vm_shifts[:i])
    input_zstarts  = np.concatenate([[0], np.cumsum(Nzs[:-1])]).astype(int)
    input_zends    = (np.cumsum(Nzs) - np.concatenate([vm_shifts,[0]])).astype(int)
    output_zstarts = np.concatenate([[0], np.cumsum(Nzs[:-1]) - np.cumsum(vm_shifts)]).astype(int)
    output_zends   = np.concatenate([output_zstarts[1:], [output_zstarts[-1]+Nzs[-1]]]).astype(int)

    if verbose >= 1:
        print(f'HDF5 voxel data:')
        print(f'subvolume_dims =\n{subvolume_dims}')
        print(f'Nzs = {Nzs}')
        print(f'vm_shifts = {vm_shifts}')
        print(f'input_zstarts  = {input_zstarts}')
        print(f'input_zends    = {input_zends}')
        print(f'output_zstarts = {output_zstarts}')
        print(f'output_zends   = {output_zends}')

    assert((input_zends - input_zstarts == output_zends - output_zstarts).all())
    if verbose >= 1: print(f'Shape to extract:\n{region}')

    nzs = input_zends - input_zstarts # Actual number of z-slices per subvolume after vm-correction
    if verbose >= 1: print(f"Volume matched subvolume nzs = {nzs}")

    # TODO: z_range is ignored
    # TODO: Store metadata about region range in json
    # TODO: Come up with appropriate "file format" scheme
    # TODO: append_file should be in io pybind module, not histograms
    # TODO: command-line specified output dtype
    # TODO: cross-section thumbnails

    z_range, y_range, x_range = region

    if verbose >= 1: print (f'Output file is {outfile}')

    total = 0 # Will be set in the first iteration
    written = 0
    nvol_rng = range(Nvols)
    nvol_iter = tqdm(nvol_rng, desc=f'Loading {sample} from HDF5 and writing binary') if verbose >= 1 else nvol_rng
    for i in nvol_iter:
        subvolume_msb = dmsb[input_zstarts[i]:input_zends[i],y_range,x_range].astype(np.uint16)
        subvolume_lsb = dlsb[input_zstarts[i]:input_zends[i],y_range,x_range].astype(np.uint16)
        combined = (subvolume_msb << 8) | subvolume_lsb

        if i == 0:
            total = output_zends[-1] * combined.shape[1] * combined.shape[2]

        del subvolume_msb
        del subvolume_lsb

        write_slice(combined, outfile, (np.uint64(output_zstarts[i]), 0, 0), combined.shape)
        written += np.prod(combined.shape)

        del combined

    msb_file.close()
    lsb_file.close()

    if verbose >= 1:
        if written == total:
            print(f"Written {written} out of {total} voxels to {outfile}")
        else:
            print(f"ERROR: Written {written} out of {total} voxels to {outfile}, missing {total-written} voxels!")

    update_hdf5(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5',
                group_name="volume_matched",
                datasets={"shape": np.array([np.sum(nzs), Ny, Nx]),
                          "subvolume_starts": output_zstarts,
                          "subvolume_ends": output_zends})

if __name__ == "__main__":
    argparser = default_parser(description=__doc__)
    argparser.add_argument('y_cutoff', action='store', type=int, default=0, nargs='?',
        help='The y-coordinate to cut the volume at. Default is 0.')
    args = argparser.parse_args()

    region = (slice_all, slice(args.y_cutoff, None), slice_all)
    h5tobin(args.sample, region, args.verbose)
