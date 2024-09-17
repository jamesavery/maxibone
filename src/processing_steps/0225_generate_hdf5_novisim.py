import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(sys.path[0]+"/../")
from config.paths import hdf5_root
import h5py
from lib.py.helpers import commandline_args, generate_cylinder_mask
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sample, nz, ny, nx, verbose = commandline_args({"sample" : "2000_projections", "nz" : 818, "ny" : 864, "nx" : 864, "verbose" : 1})

file_path = f'{hdf5_root}/raw/rec_{nx}x{ny}x{nz}_{sample}.raw'
image_output_dir = f'{hdf5_root}/processed/novisim'

os.makedirs(image_output_dir, exist_ok=True)

tomo = np.fromfile(file_path, dtype=np.float32).reshape(nz, ny, nx)
vmin, vmax = tomo.min(), tomo.max()

# Remove circle on yx planes - i.e. cylinder mask
r = tomo.shape[1] // 2
for y in range(tomo.shape[2]):
    for x in range(tomo.shape[1]):
        if (x - r) ** 2 + (y - r) ** 2 >= r ** 2:
            tomo[:, y, x] = vmin

# Plot each plane
if verbose >= 1:
    names = ['yx', 'zx', 'zy']
    planes = [tomo[nz//2,:,:], tomo[:,ny//2,:], tomo[:,:,nx//2]]

    for name, plane in zip(names, planes):
        plt.figure(figsize=(10,10))
        plt.imshow(plane)
        plt.colorbar()
        plt.savefig(f"{image_output_dir}/{sample}_{name}.png", bbox_inches='tight')
        plt.close()

# Scale to 16-bit
tomo_norm = (tomo - tomo.min()) / (tomo.max() - tomo.min())
tomo_norm *= 2**16 - 1
tomo_norm = tomo_norm.astype(np.uint16)

# Plot the histogram
if verbose >= 1:
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
    plt.savefig(f"{image_output_dir}/{sample}_histogram.png", bbox_inches='tight')
    plt.clf()

# Print metadata keys and attributes
if verbose >= 1:
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
    h5_tomo.attrs['voxelsize'] = 5 #1.85 * 8 # TODO is this right?
    h5.create_dataset('volume_matching_shifts', data=[], dtype=np.int32)
    h5.create_dataset

h5_lsb.close()
h5_msb.close()