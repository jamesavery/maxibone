import h5py, sys, jax, os.path, pathlib
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy  as jp
from resample import downsample2x, downsample3x
from config.paths import commandline_args


if __name__ == "__main__":
    input_h5, dataset_name, output_rel_path, chunk_size, compression = commandline_args({"input_h5":"<required>", "dataset": "voxels", "output_relative_to_input":"..",
                                                                                         "chunk_size":6*20, "compression":"lzf"})
    output_dir = os.path.dirname(input_h5)+"/"+output_rel_path;
    basename   = os.path.basename(input_h5);

 #   scales = [2,3,4,6,8,9,12,16,24,32];
    scales = [2,4,8,16,32];

    print(f"Downscaling from {input_h5} to scales {scales} under {output_dir}")
    
    downsample2x_jit = jax.jit(downsample2x)
    downsample3x_jit = jax.jit(downsample3x)    
    
    print(f"Opening {input_h5}")
    f = h5py.File(input_h5,'r')
    print(f"Reading {input_h5}")
    voxels1x = f[dataset_name]
    (Nz,Ny,Nx) = voxels1x.shape
    print("Downscaling 2x and 3x")
    if(chunk_size % 6):
        print("Chunk size {chunk_size} is invalid: must be divisible by 6 (both 2 and 3).")
        sys.exit(-1)

    voxels2x = np.empty((Nz//2,Ny//2,Nx//2),dtype=voxels1x.dtype)
    voxels3x = np.empty((Nz//3,Ny//3,Nx//3),dtype=voxels1x.dtype)

    for z in range(0,Nz,chunk_size):
        zend  = min(z+chunk_size, Nz)
        print(f"Reading {(zend-z,Ny,Nx)}={(zend-z)*Ny*Nx} {voxels1x.dtype} from {input_h5}")   
        chunk = jp.array(voxels1x[z:zend])
        
        print(f"...Downsampling chunk {z}:{zend}.")
        voxels2x[z//2:zend//2] = downsample2x_jit(chunk)
        voxels3x[z//3:zend//3] = downsample3x_jit(chunk)
        del chunk
        
    f.close()

    print("Copying voxels2x and voxels3x to JAX")
    voxels2x = jp.array(voxels2x)
#    voxels3x = jp.array(voxels3x)
    
    print("Downscaling to all smaller scales: {scales[2:]}")
    voxels4x = downsample2x_jit(voxels2x)

#    voxels6x = downsample2x_jit(voxels3x)
#    voxels9x = downsample3x_jit(voxels3x)

    voxels8x = downsample2x_jit(voxels4x)
#    voxels12x = downsample3x_jit(voxels4x)
    voxels16x = downsample2x_jit(voxels8x)
#    voxels24x = downsample3x_jit(voxels8x)    
    voxels32x = downsample2x_jit(voxels16x)

    voxels = [voxels2x,voxels4x,voxels8x,voxels16x,voxels32x];

    for i in range(len(scales)):
        scale = scales[i]
        scaled_dir = f"{output_dir}/{scale}x"
        pathlib.Path(scaled_dir).mkdir(parents=True, exist_ok=True)

        print(f"Writing out scale {scale}x to {scaled_dir}/{basename}")
        f = h5py.File(f"{scaled_dir}/{basename}",'w')
        f.create_dataset("voxels",data=voxels[i].astype(voxels1x.dtype),compression=compression);
        # TODO: Copy over metadata
        f.close()
    
