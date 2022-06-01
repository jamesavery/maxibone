import sys
sys.path.append(sys.path[0]+"/../")
import cv2
import h5py
import pybind_kernels.histograms as histograms
import pybind_kernels.label as label
import numpy as np
import matplotlib.pyplot as plt
from config.paths import binary_root, hdf5_root_fast as hdf5_root
from tqdm import tqdm

def load_block(sample, offset, block_size, field_names):
    Nfields = len(field_names)
    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    Nz, Ny, Nx = dm['voxels'].shape
    Nz -= np.sum(dm["volume_matching_shifts"][:])
    dm.close()
    block_size       = min(block_size, Nz-offset[0])
    voxels = np.zeros((block_size,Ny,Nx),    dtype=np.uint16)
    fields = np.zeros((Nfields,block_size//2,(Ny//2)-(offset[1]//2),Nx//2), dtype=np.uint16)
    histograms.load_slice(voxels, f'{binary_root}/voxels/1x/{sample}.uint16', (offset[0], 0, 0), (Nz, Ny, Nx))
    voxels = voxels[:,y_cutoff:,:]
    for i in range(Nfields):
        fi = np.load(f"{binary_root}/fields/implant-{field_names[i]}/2x/{sample}.npy", mmap_mode='r')
        fields[i,:] = fi[offset[0]//2:offset[0]//2 + block_size//2,offset[1]//2:Ny//2,:Nx//2]

    return voxels, fields

def load_probabilities(labeled, axes_names, field_names, c):
    P_axes = []
    P_fields = []
    for name, prob in labeled.items():
        if name.endswith('_bins'):
            P = prob.copy()
            P[P != c] = 0
            P[P == c] = 1
            P = P.astype(np.float32)
            P = cv2.GaussianBlur(P, (101,101), 0, 0)
            if P.max() > 0:
                P *= 1.0/P.max()
                #P[prob != 0] = 1.0
            if name[:-5] in axes_names:
                P_axes.append(P)
            else:
                if len(P.shape) == 3:
                    assert P.shape[0] == len(field_names)
                    P_fields += [Pf for Pf in P]
                elif len(P.shape) == 2:
                    P_fields.append(P)
                else:
                    print ('Fields is not correct shape')
                    exit(-1)
    return P_axes, P_fields

def nblocks(size, block_size):
    return (size // block_size) + (1 if size % block_size > 0 else 0)

if __name__ == '__main__':
    sample = '770c_pag'

    # Load the histograms
    labeled = np.load(f'{hdf5_root}/processed/histograms/{sample}/bins_labeled.npz')
    materials = set()
    for value in labeled.values():
        materials |= set(value.flatten())
    materials -= {0}
    print(materials)

    # TODO is not implemented in all
    y_cutoff = 1300 # 770c_pag
    implant_threshold_u16 = 32000
    block_size = 64
    vmin, vmax = 0, implant_threshold_u16 #histograms.masked_minmax(voxels)
    fmin, fmax = 0, 65535.000000 # TODO don't hardcode.
    vranges = np.array([vmin, vmax, fmin, fmax], np.float32)

    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    sz, sy, sx = dm['voxels'].shape
    sy -= y_cutoff
    fz, fy, fx = np.array((sz, sy, sx)) // 2
    dm.close()
    blocks = 1#nblocks(sz, block_size)

    axes_names = ["x", "y", "z", "r"]
    field_names = ["gauss"]#, "edt", "gauss+edt"]

    for c in {3}:
        P_axes, P_fields = load_probabilities(labeled, axes_names, field_names, c)
        #Pxs, Pys, Pzs, Prs, Pfield = load_probabilities(labeled, c)

        for i in tqdm(range(blocks), desc='Computing the probability distributions'):
            voxels = np.zeros((block_size, sy, sx), np.uint16)
            field = np.zeros((block_size//2, fy, fx), np.uint16)
            zstart, zstop = i*block_size + 1024, min((i+1)*block_size + 1024, sz)
            voxels, fields = load_block(sample, (zstart, y_cutoff, 0), block_size, field_names)
            fzstart, fzstop = i*(block_size//2), min((i+1)*(block_size//2), fz)
            ranges = np.array([0, block_size, 0, sy, 0, sx], np.uint64)
            result = np.zeros((block_size,sy,sx), dtype=np.uint8)

            label.material_prob(
                voxels, fields,
                P_axes, 1,#0b1111,
                P_fields, 0,#0b1,
                result,
                (vmin, vmax), (fmin, fmax),
                (zstart, 0, 0), (zstop, sy, sx)
            )

            np.save(f'partials/c{c}_{i}', result)