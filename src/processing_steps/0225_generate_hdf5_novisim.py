#! /usr/bin/python3
'''
Script for generating HDF5 files from the raw novisim data.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')

from config.paths import hdf5_root
import h5py
from lib.py.commandline_args import default_parser
from lib.py.helpers import generate_cylinder_mask, plot_middle_planes
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

if __name__ == "__main__":
    parser = default_parser(description=__doc__)
    parser.add_argument('--shape', action='store', default=['818', '864', '864'], nargs=3, metavar=('nz', 'ny', 'nx'), type=int,
        help='The shape of the raw data. Default is 818 864 864.')
    parser.add_argument('voxelsize', action='store', type=float, default=5, nargs='?',
        help='The size of the voxels in micrometers. Default is 5.')
    args = parser.parse_args()

    nz, ny, nx = args.shape

    file_path = f'{hdf5_root}/raw/rec_{nx}x{ny}x{nz}_{args.sample}.raw'
    image_output_dir = f'{hdf5_root}/processed/novisim'

    os.makedirs(image_output_dir, exist_ok=True)

    tomo = np.fromfile(file_path, dtype=np.float32).reshape(nz, ny, nx)
    # Rotate 90 degrees around z axis
    tomo = np.rot90(tomo, -1, (1, 2))
    yshift = ny - 1435
    tomo = np.roll(tomo, yshift, axis=1)

    vmin, vmax = tomo.min(), tomo.max()

    # Remove circle on yx planes - i.e. cylinder mask
    tomo *= generate_cylinder_mask(tomo.shape[1])

    # Plot each plane
    if args.verbose >= 1:
        plot_middle_planes(tomo, image_output_dir, args.sample)

    # Scale to 16-bit
    tomo_norm = (tomo - tomo.min()) / (tomo.max() - tomo.min())
    tomo_norm *= 2**16 - 1
    tomo_norm = tomo_norm.astype(np.uint16)

    # Plot the histogram
    if args.verbose >= 1:
        # Compute histogram of tomo_norm
        hist, bins = np.histogram(tomo_norm, bins=1000, range=(0, 2**16))
        hist[0] = 0

        # Smooth with a gaussian
        smoothed = hist
        smoothed = gaussian_filter1d(hist, 10)

        # Find the peaks
        plt.plot(np.arange(len(smoothed)) * (2**16/1000), smoothed)
        peaks, meta = find_peaks(smoothed, height=1000)
        scaled_peaks = peaks * (2**16/1000)
        scaled_peaks = scaled_peaks.astype(int)
        print (scaled_peaks)

        # Plot vertical lines where the peaks are
        plt.vlines(scaled_peaks, 0, smoothed[peaks].astype(int), color='r')

        # Find the valley between the peaks
        valley = np.argmin(smoothed[peaks[0]:peaks[1]]) + peaks[0]
        scaled_valley = valley * (2**16/1000)
        scaled_valley = scaled_valley.astype(int)
        plt.vlines([scaled_valley], 0, smoothed[valley].astype(int), color='g')
        print (scaled_valley)
        plt.savefig(f"{image_output_dir}/{args.sample}_histogram.png", bbox_inches='tight')
        plt.clf()

    # Print metadata keys and attributes
    if args.verbose >= 1:
        with h5py.File(f'{hdf5_root}/hdf5-byte/msb/770c_pag.h5', 'r') as meta:
            print (meta.keys())
            print (meta['subvolume_range'])
            print (meta['metadata'].keys())
            print (meta['metadata']['subvolume0'].attrs.keys())

    # Split into least and most significant bytes
    tomo_lsb = tomo_norm & 0xff
    tomo_msb = tomo_norm >> 8
    tomo_lsb = tomo_lsb.astype(np.uint8)
    tomo_msb = tomo_msb.astype(np.uint8)

    # Create HDF5 files
    h5_lsb = h5py.File(f'{hdf5_root}/hdf5-byte/lsb/novisim.h5', 'w')
    h5_msb = h5py.File(f'{hdf5_root}/hdf5-byte/msb/novisim.h5', 'w')

    for h5, tomo in [(h5_lsb, tomo_lsb), (h5_msb, tomo_msb)]:
        h5.create_dataset('subvolume_dimensions', (1,3), data=[[nz, ny, nx]], dtype=np.uint16)
        h5.create_dataset('subvolume_range', (1,2,), dtype=np.float32, data=np.array([0, 65535]))
        h5_tomo = h5.create_dataset('voxels', (nz, ny, nx), dtype=np.uint8, data=tomo)
        h5_tomo.dims[0].label = 'z'
        h5_tomo.dims[1].label = 'y'
        h5_tomo.dims[2].label = 'x'
        h5_tomo.attrs['voxelsize'] = args.voxelsize
        h5.create_dataset('volume_matching_shifts', data=[], dtype=np.int32)
        h5.create_dataset

    h5_lsb.close()
    h5_msb.close()