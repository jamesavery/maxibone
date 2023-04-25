import h5py, sys, os.path, pathlib, numpy as np, numpy.linalg as la, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root, commandline_args
from pybind_kernels.geometry import center_of_mass, inertia_matrix, integrate_axes, fill_implant_mask, compute_front_mask
from pybind_kernels.histograms import load_slice, erode_3d_sphere_gpu as erode_3d, dilate_3d_sphere_gpu as dilate_3d
import matplotlib.pyplot as plt
import scipy as sp, scipy.ndimage as ndi, scipy.interpolate as interpolate, scipy.signal as signal
import vedo, vedo.pointcloud as pc
from helper_functions import *
from numpy import array, newaxis as na

def largest_cc_of(mask):
    label, n_features = ndi.label(mask)
    bincnts           = np.bincount(label[label>0],minlength=n_features+1)
    
    largest_cc_ix   = np.argmax(bincnts)
    return (label==largest_cc_ix)

sample, coarse_scale, fine_scale, implant_dilation = commandline_args({"sample":"<required>","coarse_scale":8, "fine_scale":2, "implant_dilation":2})

#TODO: Output image checks

print("## 0A: Load coarse bone_region masks")
try:
    mask_filename = f"{hdf5_root}/masks/{coarse_scale}x/{sample}.h5"
    mask_file     = h5py.File(mask_filename,"r")
    coarse_bone_region     = mask_file["bone_region/mask"][:]
    coarse_back_mask       = mask_file["cut_cylinder_air/mask"][:]
    coarse_solid_implant   = mask_file["implant_solid/mask"][:]
    coarse_voxel_size = mask_file["bone_region"].attrs["voxel_size"]
    mask_file.close()
    
   
except Exception as e:
    print(f"Couldn't read {coarse_scale}x bone_region/mask from {mask_filename}: {e}")

print("## 0B: Load fine cut_cylinder_bone and implant_solid masks")
try:
    mask_filename   = f"{hdf5_root}/masks/{fine_scale}x/{sample}.h5"
    mask_file       = h5py.File(mask_filename,"r")
    front_mask      = mask_file["cut_cylinder_bone/mask"][:]
    solid_implant   = mask_file["implant_solid/mask"][:]    
    fine_voxel_size = mask_file["cut_cylinder_bone"].attrs["voxel_size"]    
    mask_file.close()
    
except Exception as e:
    print(f"Couldn't read {fine_scale}x cut_cylinder_bone/mask and implant_solid/mask from {mask_filename}: {e}")

print("## 0C: Reconcile dimensions")
(nz,ny,nx) = coarse_bone_region.shape
(mz,my,mx) = np.array([nz,ny,nx])*(coarse_scale//fine_scale) # Upscaled dimensions can differ from fine due to Nz not currently being forced divisible by 32
(Nz,Ny,Nx) = front_mask.shape

print("## 1A: Construct coarse resin mask")
cxs = np.linspace(-1,1,nx)
coarse_cylinder = (cxs[:,na]**2 + cxs[na,:]**2) <= 1

# Coarse bone region has smoothed away inner implant threads
for z in tqdm.tqdm(range(nz)):
    coarse_bone_region[z] = ndi.grey_dilation(coarse_bone_region[z],int(np.round(100/coarse_voxel_size)))

corase_back_mask     = coarse_back_mask & coarse_cylinder
coarse_back_mask     = ndi.grey_dilation(coarse_back_mask,10)    # and the back plane of the coarse bone region is pretty crappy, too
coarse_resin_mask    = (~coarse_back_mask) & (~coarse_bone_region) & (~coarse_solid_implant) & coarse_cylinder
dilate_diam   = 50
dilate_diam_v = int(np.round(dilate_diam/coarse_voxel_size))
print(f"##     Dilate coarse resin mask by {dilate_diam} micrometers ({dilate_diam_v} voxels)")
coarse_resin_mask    = ndi.grey_dilation(coarse_resin_mask,dilate_diam_v).astype(bool)


print("## 1B: Upscale coarse masks")
scale_factor         = coarse_scale // fine_scale
upscaled_resin_mask  = np.zeros((Nz,Ny,Nx), dtype=bool)
upscaled_back_mask   = np.zeros((Nz,Ny,Nx), dtype=bool)

upscaled_resin_mask[:mz]  = np.broadcast_to(coarse_resin_mask[:,na,:,na,:,na],
                                             (nz,scale_factor,ny,scale_factor,nx,scale_factor)).reshape(mz,Ny,Nx)
upscaled_resin_mask[mz:]  = upscaled_resin_mask[mz-1]

upscaled_back_mask[:mz]  = np.broadcast_to(coarse_back_mask[:,na,:,na,:,na],
                                           (nz,scale_factor,ny,scale_factor,nx,scale_factor)).reshape(mz,Ny,Nx)
upscaled_back_mask[mz:]  = upscaled_back_mask[mz-1]

xs = np.linspace(-1,1,Nx)
cylinder = (xs[:,na]**2 + xs[na,:]**2) <= 1
upscaled_resin_mask &= cylinder


print("## 1C: Fine bone region is: full front half-cylinder minus resin, and minus solid implant")
diam = int(np.round(implant_dilation/fine_voxel_size))
if diam>=1:
    print(f"##     Dilating implant a tiny bit: {implant_dilation} micrometers = {diam} voxels")
    solid_implant = ndi.grey_dilation(solid_implant,diam)
    
print("##     Computing mask")
bone_region = (~upscaled_back_mask) & (~upscaled_resin_mask) & (~solid_implant) & cylinder

bone_region = largest_cc_of(bone_region)

print(f"## 2A: Storing bone_region mask in {mask_filename}")
update_hdf5_mask(mask_filename,
                 group_name="bone_region",
                 datasets={"mask":bone_region},
                 attributes={"sample":sample,"scale":fine_scale,"voxel_size":fine_voxel_size})

print(f"## 2B: Smoothing resin mask with size {scale_factor} grey_closing ")
upscaled_resin_mask = ndi.grey_closing(upscaled_resin_mask,scale_factor)


print(f"## 2C: Storing resin mask in {mask_filename}")      
update_hdf5_mask(mask_filename,
                 group_name="resin",
                 datasets={"mask":upscaled_resin_mask},
                 attributes={"sample":sample,"scale":fine_scale,"voxel_size":fine_voxel_size})
