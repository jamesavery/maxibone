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

def apply_otsu(bins, c):
    P = np.zeros(bins.shape, dtype=np.float32)
    threshes = np.empty(bins.shape[0], dtype=np.uint64)
    histograms.otsu(bins, threshes, 1)
    for i, row in enumerate(bins):
        ma = max(1,np.float32(row.sum()))
        if c:
            P[i,:threshes[i]] = row[:threshes[i]].astype(np.float32) / ma
        else:
            P[i,threshes[i]:] = row[threshes[i]:].astype(np.float32) / ma
    return P

def load_probabilities(labeled, axes_names, field_names, c):
    P_axes = [apply_otsu(labeled[f'{name}_bins'], c) for name in axes_names]
    P_fields = []
    for name in field_names:
        idx = list(labeled['field_names']).index(name)
        bins = labeled['field_bins'][idx]
        P_fields.append(apply_otsu(bins, c))
    
    if debug:
        for i, name in enumerate(axes_names):
            plt.imshow(P_axes[i])
            plt.savefig(f'partials/c{c}_P_{name}.png')
            plt.clf()
        for i, name in enumerate(field_names):
            plt.imshow(P_fields[i])
            plt.savefig(f'partials/c{c}_P_{name}.png')
            plt.clf()
    
    return P_axes, P_fields

def nblocks(size, block_size):
    return (size // block_size) + (1 if size % block_size > 0 else 0)

if __name__ == '__main__':
    sample = '770c_pag'

    # Load the histograms
    labeled = np.load(f'{hdf5_root}/processed/histograms/{sample}/bins-bone_region2.npz')

    # TODO is not implemented in all
    #y_cutoff = 1300 # 770c_pag
    implant_threshold_u16 = 32000
    block_size = 64
    (vmin, vmax), (fmin, fmax) = labeled['value_ranges']
    #vmin, vmax = 0, implant_threshold_u16 #histograms.masked_minmax(voxels)
    #fmin, fmax = 0, 65535.000000 # TODO don't hardcode.
    vranges = np.array([vmin, vmax, fmin, fmax], np.float32)

    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    sz, sy, sx = dm['voxels'].shape
    fz, fy, fx = np.array((sz, sy, sx)) // 2
    dm.close()
    blocks = 1#nblocks(sz, block_size)

    axes_names = ["x", "y", "z", "r"]
    field_names = ["gauss", "edt", "gauss+edt"]

    for c in {False}:
        P_axes, P_fields = load_probabilities(labeled, axes_names, field_names, c)
        print ([P.min() for P in P_axes], [P.max() for P in P_axes], [P.min() for P in P_fields], [P.max() for P in P_fields])
        #Pxs, Pys, Pzs, Prs, Pfield = load_probabilities(labeled, c)

        for i in tqdm(range(blocks), desc='Computing the probability distributions'):
            voxels = np.zeros((block_size, sy, sx), np.uint16)
            field = np.zeros((block_size//2, fy, fx), np.uint16)
            zstart, zstop = i*block_size + 2000, min((i+1)*block_size + 2000, sz)
            voxels, fields = load_block(sample, (zstart, 0, 0), block_size, field_names)
            fzstart, fzstop = i*(block_size//2), min((i+1)*(block_size//2), fz)
            ranges = np.array([0, block_size, 0, sy, 0, sx], np.uint64)
            result = np.zeros((block_size,sy,sx), dtype=np.uint16)

            label.material_prob(
                voxels, fields,
                P_axes, 0b1111,
                P_fields, 0b111,
                result,
                (vmin, vmax), (fmin, fmax),
                (zstart, 0, 0), (zstop, sy, sx)
            )

            print (f'Segmentation has min {result.min()} and max {result.max()}')

            np.save(f'partials/c{c}_{i}', result)