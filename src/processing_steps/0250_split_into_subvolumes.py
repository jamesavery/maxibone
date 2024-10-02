#! /usr/bin/python3
'''
Split a large HDF5 file into smaller subvolumes.
Processing subvolumes separately can be improve segmentation performance, as there is less variance internally in the subvolumes.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.paths import hdf5_root
import h5py
from lib.py.commandline_args import default_parser
import tqdm

if __name__ == "__main__":
    args = default_parser(description=__doc__).parse_args()

    # Define the paths
    base_dir = f"{hdf5_root}/hdf5-byte"
    input_h5name  = f"{hdf5_root}/hdf5-byte/msb/{args.sample}.h5"

    # Read the subvolume dimensions
    h5file = h5py.File(input_h5name, "r+")
    subvolume_dimensions = h5file['subvolume_dimensions'][:]
    _, Ny, Nx = h5file['voxels'].shape
    h5file.close()

    for byts in ['msb', 'lsb']:
        # Open the input HDF5 file
        input_h5_name = f"{base_dir}/{byts}/{args.sample}.h5"
        input_h5 = h5py.File(input_h5_name, "r")
        voxels = input_h5['voxels']

        subvolume_iter = tqdm.tqdm(range(subvolume_dimensions.shape[0]), desc=f"Splitting {byts}") if args.verbose >= 1 else range(subvolume_dimensions.shape[0])
        for i in subvolume_iter:
            # Get the dimensions for this subvolume
            Nz, _, _ = subvolume_dimensions[i]
            z_offset = sum(subvolume_dimensions[:i,0])
            output_h5_name = f"{base_dir}/{byts}/{args.sample}_sub{i}.h5"
            output_h5 = h5py.File(output_h5_name, "w")

            # Copy the metadata
            meta = output_h5.create_group("metadata")
            for k, v in input_h5['metadata'].attrs.items():
                meta.attrs[k] = v
            subv = meta.create_group('subvolume0')
            for k, v in input_h5['metadata'][f'subvolume{i}'].items():
                subv.attrs[k] = v
            output_h5.create_dataset("global_range", data=input_h5['global_range'])

            # Overwrite the subvolume information
            output_h5.create_dataset("subvolume_dimensions", data=subvolume_dimensions[i:i+1])
            output_h5.create_dataset("subvolume_range", data=input_h5['subvolume_range'][i:i+1])
            vol_matched = output_h5.create_group('volume_matched')
            vol_matched.create_dataset('shape', data=[Nz, Ny, Nx])
            vol_matched.create_dataset('subvolume_ends', data=[Nz])
            vol_matched.create_dataset('subvolume_starts', data=[0])
            output_h5.create_dataset('volume_matching_shifts', data=[], dtype='int32')

            # Create the new dataset
            output_voxels = output_h5.create_dataset("voxels", (Nz, Ny, Nx), dtype='uint8')
            output_voxels[...] = voxels[z_offset:z_offset+Nz]
            output_voxels.attrs['voxelsize'] = voxels.attrs['voxelsize']

            output_h5.close()
        input_h5.close()