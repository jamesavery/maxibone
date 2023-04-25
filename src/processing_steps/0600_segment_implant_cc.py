import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi, tqdm, matplotlib.pyplot as plt
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root
from lib.py.helpers import commandline_args, update_hdf5_mask
from lib.cpp.cpu.io import load_slice

NA = np.newaxis

sample, scale, chunk_size, verbose = commandline_args({"sample" : "<required>",
                                                       "scale" : 8,
                                                       "chunk_size" : 256,
                                                       "verbose" : 1})

# Load metadata. TODO: Clean up, make automatic function.
meta_filename = f"{hdf5_root}/hdf5-byte/msb/{sample}.h5"
h5meta     = h5py.File(meta_filename,'r')
vm_shifts  = h5meta['volume_matching_shifts'][:]
full_Nz, Ny, Nx = h5meta['voxels'].shape    # Full image resolution
Nz         = full_Nz - np.sum(vm_shifts)    # Full volume matched image resolution
nz,ny,nx   = np.array([Nz,Ny,Nx])//scale    # Volume matched image resolution at chosen scale

voxel_size   = h5meta['voxels'].attrs['voxelsize'] * scale
global_vmin = np.min(h5meta['subvolume_range'][:,0])
global_vmax = np.max(h5meta['subvolume_range'][:,1])
values      = np.linspace(global_vmin,global_vmax,2**16)
implant_threshold_u16 = np.argmin(np.abs(values-implant_threshold))

if verbose >= 1: print(f"Reading metadata from {meta_filename}.\n"+
      f"volume_matching_shifts = {vm_shifts}\n"+
      f"full_Nz,Ny,Nx = {full_Nz,Ny,Nx}\n"+
      f"Nz            = {Nz}\n"+
      f"nz,ny,nx      = {nz,ny,nx}\n"+
      f"voxel_size    = {voxel_size}\n"+
      f"vmin,vmax     = {global_vmin,global_vmax}\n"+
      f"Implant threshold {implant_threshold} -> {implant_threshold_u16} as uint16")
h5meta.close()

noisy_implant = np.empty((nz,ny,nx),dtype=bool)
voxel_chunk   = np.empty((chunk_size,ny,nx),dtype=np.uint16)

for z in tqdm.tqdm(range(0,nz,chunk_size),"Loading and thresholding voxels"):
    chunk_length = min(chunk_size,nz-z)
    load_slice(voxel_chunk, f"{binary_root}/voxels/{scale}x/{sample}.uint16",
               (z,0,0), (nz,ny,nx))
    noisy_implant[z:z+chunk_length] = (voxel_chunk[:chunk_length] > implant_threshold_u16)

if verbose >= 1: print(f"Computing connected components")

label, n_features = ndi.label(noisy_implant)
if verbose >= 1: print(f"Counting component volumes")
bincnts           = np.bincount(label[label>0],minlength=n_features+1)

largest_cc_ix     = np.argmax(bincnts)
implant_mask=(label==largest_cc_ix)

output_dir = f"{hdf5_root}/masks/{scale}x/"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
if verbose >= 1: print(f"Writing largest connected component to {output_dir}/{sample}.h5")

update_hdf5_mask(f"{output_dir}/{sample}.h5",
                 group_name="implant",
                 datasets={'mask':implant_mask},
                 attributes={'scale':scale,'voxel_size':voxel_size,
                             'sample':sample, 'name':"implant_mask"})


# np.savez_compressed(f"{output_dir}/{sample}",mask=mask, scale=scale,voxel_size=voxel_size,
#                     sample=sample, name="implant_mask")
