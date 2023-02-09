import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi, tqdm, matplotlib.pyplot as plt
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, hdf5_root_fast, binary_root, commandline_args
from pybind_kernels.histograms import load_slice
from scipy import ndimage as ndi
from helper_functions import *

sample, m, scheme, chunk_size = commandline_args({"sample":"<required>", "material":0, "scheme":"edt","chunk_size":256})

scales = [32, 16, 8, 4, 2]

bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5')
Nz, Ny, Nx, _ = bi["dimensions"]

for scale in tqdm.tqdm(scales, desc= 'Computing connected components'):
    data = f'{binary_root}/segmented/{scheme}/P{m}/{scale}x/{sample}.uint16'
    output_dir = f'{hdf5_root_fast}/masks/{scale}x'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    nz, ny, nx = Nz // scale, Ny // scale, Nx // scale
    voxel_size = bi["voxel_size"]*scale
    
    voxels = np.empty((nz, ny, nx), dtype=np.uint16)    
    load_slice(voxels, data, (0,0,0), (nz, ny, nx))
    
#    hist,bins = np.histogram(voxels.flatten(), 256)
#    plt.plot(bins[11:],hist[10:]); plt.show()

#    mask = voxels > 32000
    mask = voxels > 40000    
    
    label, n_features = ndi.label(mask)
    counts = np.bincount(label[label>0], minlength=n_features+1)
    largest_cc = np.argmax(counts)
    mask = (label == largest_cc)

    update_hdf5(f"{output_dir}/{sample}.h5",
                group_name=f"blood",
                datasets={'mask':mask},
                attributes={'scale':scale,'voxel_size':voxel_size,
                            'sample':sample, 'name':"blood_mask"})


