import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi, tqdm, matplotlib.pyplot as plt
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, hdf5_root_fast, binary_root, commandline_args
from pybind_kernels.geometry import center_of_mass, inertia_matrix, integrate_axes, sample_plane
from pybind_kernels.histograms import load_slice
from io_modules.io import update_hdf5
from scipy import ndimage as ndi

sample, c, chunk_size = commandline_args({"sample":"<required>", "material":0, "chunk_size":256})

scales = [32, 16, 8, 4, 2]

meta = f'{hdf5_root}/hdf5-byte/msb/{sample}.h5'
h5meta = h5py.File(meta, 'r')
Nz, Ny, Nx = h5meta['voxels'].shape

for scale in tqdm.tqdm(scales, desc= 'Computing connected components'):
    data = f'{binary_root}/segmented/c{c}/{scale}x/{sample}.uint16'
    output_dir = f'{hdf5_root_fast}/masks/{scale}x'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    nz, ny, nx = Nz // scale, Ny // scale, Nx // scale
    buffer = np.empty((nz, ny, nx), dtype=np.uint16)
    load_slice(buffer, data, (0,0,0), (nz, ny, nx))
    label, n_features = ndi.label(buffer)
    counts = np.bincount(label[label>0], minlength=n_features+1)
    largest_cc = np.argmax(counts)
    mask = (label == largest_cc)
    voxel_size = h5meta['voxels'].attrs['voxelsize'] * scale
    update_hdf5(f"{output_dir}/{sample}.h5",
                group_name=f"c{c}",
                datasets={'mask':mask},
                attributes={'scale':scale,'voxel_size':voxel_size,
                            'sample':sample, 'name':"implant_mask"})

h5meta.close()