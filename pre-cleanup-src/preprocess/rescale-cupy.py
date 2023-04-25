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
    input_h5, output_dir, compression, chunk_size = commandline_args({"input_h5":"<required>", "output_dir": "<required>",
                                                                      "compression":"lzf","chunk_size":6*10})
      
    scales = [2,3,4,6,8,9,12,16,24,32];

#    downsample2x_jit = jax.jit(downsample2x)
#    downsample3x_jit = jax.jit(downsample3x)    
    downsample2x_jit = downsample2x
    downsample3x_jit = downsample3x
    
    print(f"Opening {input_h5}")
    f = h5py.File(input_h5,'r')
    print(f"Reading {input_h5}")
    voxels1x = f['voxels']
    (Nz,Ny,Nx) = voxels1x.shape
    print("Downscaling 2x and 3x")
    if(chunk_size % 6):
        print("Chunk size {chunk_size} is invalid: must be divisible by 6 (both 2 and 3).")
        sys.exit(-1)

        print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")              # 0



    voxels2x = np.empty((Nz//2,Ny//2,Nx//2),dtype=voxels1x.dtype)
    voxels3x = np.empty((Nz//3,Ny//3,Nx//3),dtype=voxels1x.dtype)

    print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")              # 0
    
    for z in range(0,Nz,chunk_size):
        zend  = min(z+chunk_size, Nz)
        print(f"Reading {(zend-z,Ny,Nx)}={(zend-z)*Ny*Nx} {voxels1x.dtype} from file to GPU")   
        chunk = cp.array(voxels1x[z:zend])
        print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")              # 0
        
        print(f"...Downsampling chunk {z}:{zend}.")
        voxels2x[z//2:zend//2] = downsample2x_jit(chunk).get()
        voxels3x[z//3:zend//3] = downsample3x_jit(chunk).get()
        del chunk
        print(f"Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")              # 0
        
    f.close()

    print(f"Allocating {(Nz//3,Ny//3,Nx//3)}={Nz//3*Ny//3*Nx//3} {voxels1x.dtype} for voxels3x on GPU")
    voxels3x = cp.array(voxels3x)
    
    print("Downscaling to all smaller scales: {scales[2:]}")    
    voxels6x = downsample2x_jit(voxels3x)
    voxels9x = downsample3x_jit(voxels3x)

    print("Releasing voxels3x")
    print(f"Before: Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")              # 0
    voxels3x = voxels3x.get()
    print(f"After: Used GPU memory: {mempool.used_bytes()//1000000}MB out of {mempool.total_bytes()/1000000}MB. {pinned_mempool.n_free_blocks()} free pinned blocks.")              # 0
    
    print(f"Allocating {(Nz//2,Ny//2,Nx//2)}={Nz//2*Ny//2*Nx//2} {voxels1x.dtype} for voxels2x on GPU")
    voxels2x = cp.array(voxels2x)
    
    
    voxels4x = downsample2x_jit(voxels2x)
    voxels8x = downsample2x_jit(voxels4x)
    voxels12x = downsample3x_jit(voxels4x)
    voxels16x = downsample2x_jit(voxels8x)
    voxels24x = downsample3x_jit(voxels8x)    
    voxels32x = downsample2x_jit(voxels16x)

    basename  = path.basename(input_h5);

    voxels = [voxels2x,voxels3x,voxels4x,voxels6x,voxels8x,voxels9x,voxels12x,voxels16x,voxels24x,voxels32x];

    for i in range(len(scales)):
        output_h5 = f"{output_dir}/{scale}x/{basename}.h5"
        print(f"Writing out scale {scale}x to {output_h5}")
        f = h5py.File(output_h5,'w')
        f.create_dataset("voxels",data=voxels[i].get().astype(np.uint8),compression=compression);
        # TODO: Copy over metadata
        f.close()
    
