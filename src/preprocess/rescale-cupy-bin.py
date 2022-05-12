import h5py, sys, pathlib
sys.path.append(sys.path[0]+"/../")
import numpy as np
import cupy  as cp
#import numpy as cp
from resample import downsample2x, downsample3x
from config.paths import commandline_args, hdf5_root, binary_root

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()


if __name__ == "__main__":
    sample, image, chunk_size, dtype = commandline_args({"sample":"<required>", "image": "voxels",
                                                               "chunk_size":32*2, "dtype":"uint16"})

    scales = [2,4,8,16,32];     # Can do 6, 9, 12, 24, 27, etc. as well, but we currently don't. See old rescaly-cupy.py
    T = np.dtype(dtype)
    
    meta_h5    = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    Nz, Ny, Nx = meta_h5['voxels'].shape
    shifts     = meta_h5['volume_matching_shifts'][:] # TODO: Do this in a neater way
    Nz        -= np.sum(shifts)
    meta_h5.close()    

    input_bin = f"{binary_root}/{image}/1x/{sample}.{dtype}"

    print("Downscaling from 1x to 2x")
    if(chunk_size % 32):
        print("Chunk size {chunk_size} is invalid: must be divisible by 32.")
        sys.exit(-1)
#        print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")

#    print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")

    # TODO: Just iterate now we do powers of two
    voxels2x  = np.empty((Nz//2,Ny//2,Nx//2),dtype=T)
    voxels4x  = np.empty((Nz//4,Ny//4,Nx//4),dtype=T)
    voxels8x  = np.empty((Nz//8,Ny//8,Nx//8),dtype=T)
    voxels16x = np.empty((Nz//16,Ny//16,Nx//16),dtype=T)
    voxels32x = np.empty((Nz//32,Ny//32,Nx//32),dtype=T)            
    voxels    = [voxels2x,voxels4x,voxels8x,voxels16x,voxels32x];    
    
    for z in range(0,Nz,chunk_size):
        zend  = min(z+chunk_size, Nz)
        chunk_items = (zend-z) * Ny * Nx 
        print(f"Reading {(zend-z,Ny,Nx)}={(zend-z)*Ny*Nx} {dtype} from file to GPU")
        # NB: count is in items, offset is in bytes! Jesus Christ.
        voxels1x_chunk = cp.fromfile(input_bin, dtype=T, count=chunk_items, offset=z*Ny*Nx*T.itemsize).reshape(zend-z,Ny,Nx) 
#        print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")
        print(f"...Downsampling chunk {z}:{zend}.")
        voxels2x_chunk = downsample2x(voxels1x_chunk)
        del voxels1x_chunk
        voxels4x_chunk  = downsample2x(voxels2x_chunk)
        voxels8x_chunk  = downsample2x(voxels4x_chunk)
        voxels16x_chunk = downsample2x(voxels8x_chunk)
        voxels32x_chunk = downsample2x(voxels16x_chunk)

        ## if cupy
        voxels2x[z//2:zend//2]  = voxels2x_chunk.get()
        voxels4x[z//4:zend//4]  = voxels4x_chunk.get()
        voxels8x[z//8:zend//8]  = voxels8x_chunk.get()
        voxels16x[z//16:zend//16] = voxels16x_chunk.get()
        voxels32x[z//32:zend//32] = voxels32x_chunk.get()
        ## else
        # voxels2x[z//2:zend//2]  = voxels2x_chunk
        # voxels4x[z//4:zend//4]  = voxels4x_chunk
        # voxels8x[z//8:zend//8]  = voxels8x_chunk
        # voxels16x[z//16:zend//16] = voxels16x_chunk
        # voxels32x[z//32:zend//32] = voxels32x_chunk

#        print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")              # 0
        
    print(f"Downscaling to all smaller scales: {scales[2:]}")    
    print(f"Allocating {(Nz//2,Ny//2,Nx//2)}={Nz//2*Ny//2*Nx//2} {dtype} for voxels2x on GPU")
    


    for i in range(len(scales)):
        output_dir = f"{binary_root}/{image}/{scales[i]}x/"
        pathlib.Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)            
        print(f"Writing out scale {scales[i]}x to {output_dir}/{sample}.uint16")
        voxels[i].tofile(f"{output_dir}/{sample}.uint16")
