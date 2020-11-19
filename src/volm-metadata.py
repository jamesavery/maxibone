# TODO:
# 1] shadow effects could perhaps be removed by histogram matching of final adjacent regions
#    - should probably be removed by kernel-operation around outer edge, before matching...
# 2] different scales should be incorporated when using any number of subvolumes
#    - currently only 1x scale is supported for 3,5,6 subvolumes -- need to expand compute_dex()

###########################
### Fix shifted volumes ###
###########################

import numpy as np
import jax.numpy as jp
import h5py, jax, sys
from PIL import Image
import matplotlib.pyplot as plt

sample, overlap, max_shift, hdf5_root = sys.argv[1:]

overlap, max_shift = int(overlap), int(max_shift)

def match_region(voxels_top, voxels_bot, overlap, max_shift):
    """
    Find shift that minimizes squared differences with overlap <= shift <= max_shift
    """
    norm = jp.prod(jp.array(voxels_top[0].shape))
    # Shifts smaller than the overlap overlap with shift
    sums_lt = jp.array( [ jp.sum(((voxels_top[-shift:] - voxels_bot[0:shift])/(norm*shift))**2)
                          for shift in range(2,overlap)] )
    # Shifts larger than the overlap overlap with overlap
    sums_ge = jp.array( [ jp.sum(((voxels_top[-overlap:] - voxels_bot[shift:shift+overlap])/(norm*overlap))**2)
                          for shift in range(0,max_shift-overlap)] )
    # print("sums_lt=",sums_lt)
    # print("sums_ge=",sums_ge)
    sums = jp.concatenate([sums_lt, sums_ge])
    return jp.argmin( sums ), sums.min()

match_shift_jit = jax.jit(match_region);

h5file = h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5", "r")

voxels = h5file['voxels']
subvolume_dimensions = h5file['subvolume_dimensions'][:]
(Nz,Ny,Nx) = h5file['voxels'].shape
crossings = np.cumsum(subvolume_dimensions[:-1,0]).astype(int)

print(f"Crossings: {crossings}")
shifts = np.zeros(len(crossings),dtype=np.int)
for i in range(len(crossings)):
    crossing = crossings[i]
    
    print(f"Reading top region: voxels[{crossing-max_shift}:{crossing}]")
    top_voxels = jp.array(voxels[crossing-max_shift:crossing]).astype(jp.float32)
    print(f"Reading bottom region: voxels[{crossing}:{crossing+max_shift}]")    
    bot_voxels = jp.array(voxels[crossing:crossing+max_shift]).astype(jp.float32)

    print(f"Matching regions (Shapes: {bot_voxels.shape} {top_voxels.shape})")
    shift, error = match_region(top_voxels,bot_voxels,overlap,max_shift)
    shifts[i] = shift
    
    print(f"Optimal shift is {shift} with error {error}")
    print(f"Writing images of  matched slices")
    merged_voxels = jp.concatenate([top_voxels,bot_voxels[shift:]])
    merged_zy_slice  = np.array(merged_voxels[:,:,Nx//2])

    Image.fromarray(merged_zy_slice.astype(np.uint8)).save(f"{hdf5_root}/processed/volume_matched/{sample}-{i}cross-zy.png")

    image = np.zeros((2*max_shift,1980),dtype=np.uint8)
    image[:max_shift,:1980//2] = top_voxels[:,Ny//2-1980//4:Ny//2+1980//4,Nx//2].astype(np.uint8)
    image[max_shift-shift:-shift,1980//2:] = bot_voxels[:,Ny//2-1980//4:Ny//2+1980//4,Nx//2].astype(np.uint8)    
    Image.fromarray(image).save(f"{hdf5_root}/processed/volume_matched/{sample}-{i}bottop-zx.png")


np.save(f"{hdf5_root}/processed/volume_matched/{sample}-shifts.npy",shifts)


