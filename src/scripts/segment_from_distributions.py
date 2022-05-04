import cv2
import h5py
import histograms
import label
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data(experiment):
    dm = h5py.File(f'{h5root}/msb/{experiment}.h5', 'r')
    dl = h5py.File(f'{h5root}/lsb/{experiment}.h5', 'r')
    result = np.ndarray(dm['voxels'].shape, dtype=np.uint16)
    block_size = 1024
    blocks = int(np.ceil(dm['voxels'].shape[0] / block_size))
    for i in range(blocks):
        start, stop = i*block_size, min((i+1)*block_size, dm['voxels'].shape[0]-1)
        result[start:stop] = (dm['voxels'][start:stop].astype(np.uint16) << 8) | dl['voxels'][start:stop].astype(np.uint16)
    dm.close()
    dl.close()
    return result

def load_probabilities(labeled, c):
    Ps = []
    for prob in labeled.values():
        P = prob.copy()
        P[P != c] = 0
        P[P == c] = 1
        P = P.astype(np.float32)
        P = cv2.GaussianBlur(P, (101,101), 0, 0)
        if P.max() > 0:
            P *= 1.0/P.max()
            P[prob != 0] = 1.0
        Ps.append(P)

    return Ps

def nblocks(size, block_size):
    return (size // block_size) + (1 if size % block_size > 0 else 0)

if __name__ == '__main__':
    h5root = '/mnt/shared/MAXIBONE/Goats/tomograms'
    sample = '770c_pag'

    # Load the histograms
    labeled = np.load(f'{h5root}/processed/histograms/{sample}/bins_relabeled.npz')
    materials = set()
    for value in labeled.values():
        materials |= set(value.flatten())
    materials -= {0}
    print(materials)

    dm = h5py.File(f'{h5root}/hdf5-byte/msb/{sample}.h5', 'r')
    dl = h5py.File(f'{h5root}/hdf5-byte/lsb/{sample}.h5', 'r')
    fi = h5py.File(f'{h5root}/processed/implant-edt/2x/{sample}.h5', 'r')

    y_cutoff = 1300 # 770c_pag
    implant_threshold_u16 = 32000
    block_size = 256
    vmin, vmax = 0, implant_threshold_u16 #histograms.masked_minmax(voxels)
    fmin, fmax = 4.000000, 65535.000000 # TODO don't hardcode.
    vranges = np.array([vmin, vmax, fmin, fmax], np.float32)

    sz, sy, sx = dm['voxels'].shape
    sy -= y_cutoff
    fz, fy, fx = fi['voxels'].shape
    fy -= y_cutoff // 2
    blocks = nblocks(sz, block_size)

    for c in {2}:
        Pxs, Pys, Pzs, Prs, Pfield = load_probabilities(labeled, c)

        for i in tqdm(range(blocks), desc='Computing the probability distributions'):
            voxels = np.zeros((block_size, sy, sx), np.uint16)
            field = np.zeros((block_size//2, fy, fx), np.uint16)
            zstart, zstop = i*block_size, min((i+1)*block_size, sz)
            fzstart, fzstop = i*(block_size//2), min((i+1)*(block_size//2), fz)
            voxels[:,:,:] = \
                    (dm['voxels'][zstart:zstop,y_cutoff:,:].astype(np.uint16) << 8) | \
                    (dl['voxels'][zstart:zstop,y_cutoff:,:].astype(np.uint16))
            field = np.zeros(np.array(voxels.shape)//2, dtype=np.uint16)
            if fzstop > fzstart:
                print (f'fzstart:fzstop = {fzstart}:{fzstop}, field.shape = {field.shape}, fi.shape = {fi["voxels"].shape}')

                field[:(fzstop-fzstart),:fy,:fx] = fi['voxels'][fzstart:fzstop,y_cutoff//2:,:].astype(np.uint16)
            ranges = np.array([0, block_size, 0, sy, 0, sx], np.uint64)
            result = np.zeros((block_size,sy,sx), dtype=np.uint8)

            label.material_prob(
                voxels,
                Pxs, Pys, Pzs, Prs, Pfield,
                field,
                result,
                (vmin, vmax), (fmin, fmax),
                (zstart, 0, 0), (zstop, sy, sx)
            )

            np.save(f'partials/c{c}_{i}', result)