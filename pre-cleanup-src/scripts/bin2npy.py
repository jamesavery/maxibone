import os, sys, h5py, numpy as np, tqdm
sys.path.append(sys.path[0]+"/../")
import pybind_kernels.histograms as binary_io # TODO: Move binary io stuff away from histograms
from config.paths import *
from helper_functions import block_info

sample, image, scale, dtype, block_size = commandline_args({"sample":"<required>",
                                                            "image":"voxels",
                                                            "scale":1,
                                                            "dtype":"uint16",
                                                            "block_size": 100
})


meta_filename = f"{hdf5_root}/hdf5-byte/msb/{sample}.h5"
bin_filename  = f"{binary_root}/{image}/{scale}x/{sample}.{dtype}"
npy_filename  = f"{binary_root}/{image}/{scale}x/{sample}.npy"

bi = block_info(meta_filename)

(Nz,Ny,Nx,Nr) = bi['dimensions']
(Nz,Ny,Nx)    = np.array((Nz,Ny,Nx))//scale

npy_file = np.memmap(npy_filename,dtype=np.dtype(dtype),mode="w+", shape=(Nz,Ny,Nx))

print(f"Convertion from {bin_filename} to {npy_filename}")
for z in tqdm.tqdm(range(0,Nz,block_size)):
    nz = min(block_size, Nz-z)
    buffer   = np.empty((nz,Ny,Nx),dtype=np.dtype(dtype))
    binary_io.load_slice(buffer, bin_filename, (z, 0, 0), (Nz, Ny, Nx)) 

    npy_file[z:z+nz] = buffer






