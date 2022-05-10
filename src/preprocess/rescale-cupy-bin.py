import h5py, sys
import numpy as np
import cupy  as cp
import os.path as path
from resample import downsample2x, downsample3x
from config.paths import commandline_args

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()


if __name__ == "__main__":
    sample, image, compression, chunk_size = commandline_args({"sample":"<required>", "image": "voxels",
                                                               "chunk_size":32*10, "dtype":"uint16"})
    
    scales = [2,4,8,16,32];     # Can do 6, 9, 12, 24, 27, etc. as well, but we currently don't. See old rescaly-cupy.py
    
    meta_h5    = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    Nz, Ny, Nx = meta_h5['voxels'].shape
    meta_h5.close()    

    input_bin = f"{binary_root}/{image}/1x/{sample}.{dtype}"

    print("Downscaling from 1x to 2x")
    if(chunk_size % 32):
        print("Chunk size {chunk_size} is invalid: must be divisible by 32.")
        sys.exit(-1)
        print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")

    voxels2x = cp.empty((Nz//2,Ny//2,Nx//2),dtype=np.dtype(dtype))

    print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")
    
    for z in range(0,Nz,chunk_size):
        zend  = min(z+chunk_size, Nz)
        chunk_bytes = (zend-z) * Ny * Nx # +1?
        print(f"Reading {(zend-z,Ny,Nx)}={(zend-z)*Ny*Nx} {voxels1x.dtype} from file to GPU")
        voxels1x_chunk = cp.fromfile(input_bin, dtype=np.dtype(dtype), count=chunk_bytes, offset=z*Ny*Nx)
        print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")
        print(f"...Downsampling chunk {z}:{zend}.")
        voxels2x[z//2:zend//2] = downsample2x(voxels1x_chunk)
        del voxels1x_chunk
        print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")              # 0
        
    print("Downscaling to all smaller scales: {scales[2:]}")    
    print(f"Allocating {(Nz//2,Ny//2,Nx//2)}={Nz//2*Ny//2*Nx//2} {dtype} for voxels2x on GPU")
    
    voxels4x  = downsample2x(voxels2x)
    voxels8x  = downsample2x(voxels4x)
    voxels12x = downsample3x(voxels4x)
    voxels16x = downsample2x(voxels8x)
    voxels24x = downsample3x(voxels8x)    
    voxels32x = downsample2x(voxels16x)

    voxels = [voxels2x,voxels4x,voxels8x,voxels16x,voxels32x];

    for i in range(len(scales)):
        output_bin = f"{binary_root}/{image}/{scales[i]}x/{sample}.{dtype}"
        print(f"Writing out scale {scale}x to {output_bin}")
        voxels[i].tofile(output_bin)
