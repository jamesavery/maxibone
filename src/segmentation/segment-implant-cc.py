import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root, commandline_args
from pybind_kernels.geometry import center_of_mass, inertia_matrix, integrate_axes, sample_plane
from pybind_kernels.histograms import load_slice
import matplotlib.pyplot as plt

NA = np.newaxis

sample, scale, chunk_size = commandline_args({"sample":"<required>","scale":8, "chunk_size":256})

# Load metadata. TODO: Clean up, make automatic function.
meta_filename = f"{hdf5_root}/hdf5-byte/msb/{sample}.h5"
h5meta     = h5py.File(meta_filename,'r')
vm_shifts  = h5meta['volume_matching_shifts']
full_Nz, Ny, Nx = h5meta['voxels'].shape    # Full image resolution
Nz         = full_Nz - np.sum(vm_shifts)    # Full volume matched image resolution
nz,ny,nx   = np.array([Nz,Ny,Nx])//scale    # Volume matched image resolution at chosen scale

voxelsize   = h5meta['voxels'].attrs['voxelsize'] * scale
global_vmin = np.min(h5meta['subvolume_range'][:,0])
global_vmax = np.max(h5meta['subvolume_range'][:,1])
values      = np.linspace(global_vmin,global_vmax,2**16)
implant_threshold_u16 = np.argmin(np.abs(values-implant_threshold))
print(f"Implant threshold {implant_threshold} -> {implant_threshold_u16} as uint16")
h5meta.close()

output_dir = f"{binary_root}/masks/implant/{scale}x"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

noisy_implant = np.empty((nz,ny,nx),dtype=bool)
voxel_chunk   = np.empty((chunk_size,ny,nx),dtype=np.uint16)

for z in tqdm.tqdm(range(0,nz,chunk_size),"Loading and thresholding voxels"):
    chunk_length = min(chunk_size,nz-z)
    load_slice(voxel_chunk, f"{binary_root}/voxels/{scale}x/{sample}.uint16",
               (chunk_length,0,0), (nz,ny,nx))
    noisy_implant[z:z+chunk_length] = voxel_chunk[:chunk_length]
    
                                                  
print(f"Computing connected components")
label, n_features = ndi.label(noisy_implant)
print(f"Counting component volumes")
bincnts           = np.bincount(label[label>0],minlength=n_features+1)

print(f"Writing largest connected component to {output_dir}/{sample}.npz")
largest_cc_ix     = np.argmax(bincnts)
implant_mask=(label==largest_cc_ix)
np.savez_compressed(f"{output_dir}/{sample}.npz",implant_mask=implant_mask)

