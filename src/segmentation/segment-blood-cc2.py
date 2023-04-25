import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi, tqdm, matplotlib.pyplot as plt
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, hdf5_root_fast, binary_root, commandline_args
from pybind_kernels.histograms import load_slice
from scipy import ndimage as ndi
from helper_functions import *
import cc3d

sample, mask_scale, region_mask, m, segment_threshold = \
    commandline_args({"sample":"<required>", "mask_scale":16, "region_mask":"bone_region",
                      "blood_material_id":0, "segment_threshold":0.75})

scales = [32, 16, 8, 4, 2, 1]

bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5')
Nz, Ny, Nx, _ = bi["dimensions"]

try:
    mask_filename = f"{hdf5_root}/masks/{mask_scale}x/{sample}.h5"
    mask_file     = h5py.File(mask_filename,"r")
    mask = mask_file[f"{region_mask}/mask"]
    mask_voxel_size = mask.attrs["voxel_size"]
    mz,my,mx   = np.nonzeros(mask[:])
    
    mask_bbox = np.array([[mz.min(),mz.max()],[my.min(),my.max()],[mx.min(),mx.max()]])


    del mask, mz,my,mx
    mask_file.close()
    
except Exception as e:
    print(f"Can't load {region_mask}/mask from {mask_filename}: {e}")
    sys.exit(-1)


for scale in tqdm.tqdm(scales, desc= 'Computing connected components'):
    P_file = f'{binary_root}/segmented/P{m}/{scale}x/{sample}.uint16'
    output_dir = f'{hdf5_root_fast}/masks/{scale}x'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    nz, ny, nx = Nz // scale, Ny // scale, Nx // scale
    voxel_size = bi["voxel_size"]*scale

    voxels = np.memmap(P_file,dtype=np.uint16, shape=(nz,ny,nx), mode='r')    

    [[zmin,zmax],[ymin,ymax],[xmin,xmax]] = (mask_bbox*(mask_voxel_size/voxel_size))
    zmin,xmin,ymin = int(np.floor([zmin,xmin,ymin]))
    zmax,xmax,ymax = int(np.ceil ([zmin,xmin,ymin]))+1
    
    soft_tissue_mask = voxels[zmin:zmax,ymin:ymax,xmin:xmax] > int(segment_threshold*(2**16-1))

    label, n_features = cc3d.largest_k(
        soft_tissue_mask, k=1, 
        return_N=True,
    )
    del soft_tissue_mask   
    
    stats = cc3d.statistics(label)

    print(f"Blood volume is {stats['voxel_counts'][1]*(voxel_size**3)/1e9} mm^3")    
    print(f"cc3d.statistics:\n{stats}")

    blood_mask = np.zeros((nz,ny,nx),dtype=np.bool)
    blood_mask[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1] = (label == largest_cc)
    
    update_hdf5(f"{output_dir}/{sample}.h5",
                group_name=f"blood",
                datasets={'mask':blood_mask},
                attributes={'scale':scale,'voxel_size':voxel_size,
                            'sample':sample, 'name':"blood_mask"})

