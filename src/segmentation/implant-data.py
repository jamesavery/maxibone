import h5py, sys, os.path, pathlib, numpy as np, numpy.linalg as la, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root, commandline_args
from pybind_kernels.geometry import center_of_mass, inertia_matrix, integrate_axes
from pybind_kernels.histograms import load_slice, erode_3d_sphere_gpu as erode_3d, dilate_3d_sphere_gpu as dilate_3d
import matplotlib.pyplot as plt
import scipy as sp, scipy.ndimage as ndi, scipy.interpolate as interpolate, scipy.signal as signal
import vedo, vedo.pointcloud as pc
from helper_functions import *
from numpy import array, newaxis as NA


sample, scale = commandline_args({"sample":"<required>","scale":2})

print(f"Loading principal axis and cylinder frame-of-references")
h5meta = h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5","r")
try:    
    h5g = h5meta["implant-FoR"]
    Muvwp = h5g["UVWp_transform"][:] # Actually all that's needed
    bbox = h5g["bounding_box_UVWp"][:]
    implant_radius = h5g.attrs["implant_radius"]
    theta_range    = h5g["theta_range"][:]
    h5meta.close()    
except Exception as e:
    print(f"Cant't read implant frame-of-reference: {e}")
    print(f"Make sure you have run implant-FoR.py for {sample} at scale {scale}x")
    sys.exit(-1)

print(f"Loading {scale}x implant mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
try:
    implant_file = h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5",'r')
    implant      = implant_file["implant/mask"][:]
    voxel_size   = implant_file["implant"].attrs["voxel_size"]
    nz,ny,nx     = implant.shape
    implant_file.close()
except Exception as e:
    print(f"Can't read implant mask {e}.\nDid you run segment-implant-cc.py?")
    sys.exit(-1)



((Up_min,Up_max),(Vp_min,Vp_max),(Wp_min,Wp_max)) = bbox

n_bins = 2048//scale
rsqr_fraction = 0.7 # Fill in whenever W>Wmin and r<rsqr_fraction*rsqr_maxs[U_i]
solid_implant_mask = np.empty(implant.shape, dtype=np.uint8)
rsqr_maxs          = np.zeros((n_bins,), dtype=np.float32)
profile            = np.zeros((n_bins,), dtype=np.float32)

print(f"Filling implant mask")
fill_implant_mask(implant.astype(np.uint8,copy=False),
                  voxel_size,tuple(bbox.flatten()), rsqr_fraction,
                  tuple(Muvwp.flatten()),
                  solid_implant_mask, rsqr_maxs, profile);

# back_mask  = (Ws<0)
# front_mask = largest_cc_of((Ws>50)*(~solid_implant))#*(thetas>=theta_from)*(thetas<=theta_to)

# # back_part = voxels*back_mask
# front_part = voxels*front_mask

update_hdf5(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5",
            group_name="implant-data",
            datasets  = {"quarter_profile":profile,
                         "r_maxs": np.sqrt(rsqr_maxs)}
)
            
update_hdf5_mask(f"{hdf5_root}/masks/{scale}x/{sample}.h5",
                 group_name="implant_solid",
                 datasets={"mask":solid_implant_mask.astype(bool,copy=False)},
                 attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})


