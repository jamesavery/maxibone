import cv2
import h5py
import histograms
import label
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    h5root = '/mnt/shared/MAXIBONE/Goats/tomograms/hdf5-byte/'

    # Load the histograms
    labeled = np.load('/mnt/shared/MAXIBONE/Goats/tomograms/processed/histograms/770c_pag/bins_relabeled.npz')
    materials = set()
    for value in labeled.values():
        materials |= set(value.flatten())
    materials -= {0}
    print(materials)

    voxels = load_data('770c_pag')
    with h5py.File('/mnt/shared/MAXIBONE/Goats/tomograms/processed/implant-edt/2x/770c_pag.h5', 'r') as field_h5:
        field = field_h5['voxels'][:]
    vmin, vmax = histograms.masked_minmax(voxels)
    fmin, fmax = 4.000000, 65535.000000 # TODO don't hardcode.
    vranges = np.array([vmin, vmax, fmin, fmax], np.float32)

    for c in {5}:
        Pxs, Pys, Pzs, Prs, Pfield = load_probabilities(labeled, c)

        for z in range((voxels.shape[0] // 1024)-1):
            for y in range((voxels.shape[1] // 1024)-1):
                for x in range((voxels.shape[2] // 1024)-1):
                    ranges = np.array([z*1024, (z+1)*1024, y*1024, (y+1)*1024, x*1024, (x+1)*1024], np.uint64)
                    result = np.zeros((1024,1024,1024), dtype=np.uint8)

                    label.material_prob(
                        voxels,
                        Pxs, Pys, Pzs, Prs, Pfield,
                        field,
                        result,
                        vranges,
                        ranges
                    )

                    np.save(f'partials/c{c}_{z}_{y}_{x}', result)