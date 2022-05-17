import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root, commandline_args
from pybind_kernels.geometry import center_of_mass, inertia_matrix, integrate_axes, sample_plane
from pybind_kernels.histograms import load_slice
import matplotlib.pyplot as plt

NA = np.newaxis

sample, scale, chunk_size, pa_scale = commandline_args({"sample":"<required>","scale":1,'chunk_size':256,"pa_scale":8})

# Load metadata. TODO: Clean up, make automatic function.
meta_filename = f"{hdf5_root}/hdf5-byte/msb/{sample}.h5"
h5meta     = h5py.File(meta_filename,'r')
vm_shifts  = h5meta['volume_matching_shifts']
full_Nz, Ny, Nx = h5meta['voxels'].shape    # Full image resolution
Nz         = full_Nz - np.sum(vm_shifts)
nz,ny,nx   = np.array([Nz,Ny,Nx])//pa_scale   # Coarse image resolution for PA-analysis

voxelsize   = h5meta['voxels'].attrs['voxelsize'] * scale
global_vmin = np.min(h5meta['subvolume_range'][:,0])
global_vmax = np.max(h5meta['subvolume_range'][:,1])
values      = np.linspace(global_vmin,global_vmax,2**16)
h5meta.close()

output_dir = f"{binary_root}/masks/implant/{scale}x"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

pa_voxels = np.fromfile(f"{binary_root}/voxels/{pa_scale}x/{sample}.uint16",count=nz*ny*nx*2, offset=0, dtype=np.uint16).reshape(nz,ny,nx)
implant_threshold_u16 = np.argmin(np.abs(values-implant_threshold))

print(f"Implant threshold {implant_threshold} -> {implant_threshold_u16} as uint16")
pa_noisy_implant = pa_voxels > implant_threshold_u16

pa_label, n_features = ndi.label(pa_noisy_implant)
pa_bincnts    = np.bincount(pa_label[pa_label>0],minlength=n_features+1)
largest_cc_ix = np.argmin(pa_bincnts)

implant_mask=(pa_label==largest_cc_ix)
np.savez_compressed(f"{output_dir}/{sample}.npz",implant_mask=implant_mask)
# for z0 in range(0,Nz,chunk_size):
#     z1 = min(Nz,z0+chunk_size)
#     print(f"Reading and thresholding chunk {z0}:{z1} of {voxels_in.shape} {voxels_in.dtype}.")
#     implant_chunk       = voxels_in[z0:z1] >= byte_implant_threshold
#     print(f"Max inddata: {voxels_in[z0:z1].max()}; Number of matching voxels: {np.sum(implant_chunk)}")
#     if(sphere_diameter>1):
#         print(f"Binary opening with {sphere_diameter*voxelsize} micrometer sphere ({sphere_diameter} voxel radius).")
#         implant_chunk[sphere_diameter//2:-sphere_diameter//2] = ndi.binary_opening(implant_chunk,sph5)[sphere_diameter//2:-sphere_diameter//2]
#     print("Writing chunk")
#     voxels_out[z0:z1]  = implant_chunk
    
# h5in.close()
# h5out.close()
