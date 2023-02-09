#!/usr/bin/env python3
import os, sys, h5py, numpy as np, pathlib, tqdm, vedo, matplotlib.pyplot as plt, edt, vedo.pointcloud as pc, scipy.ndimage as ndi
sys.path.append(sys.path[0]+"/../")
from config.paths import *
from helper_functions import *
NA = np.newaxis


def coordinate_image(shape):
    Nz,Ny,Nx   = shape
    print(f"Broadcasting coordinates for {shape} image")
    zs, ys, xs = np.broadcast_to(np.arange(Nz)[:,NA,NA],shape),np.broadcast_to(np.arange(Ny)[NA,:,NA],shape), np.broadcast_to(np.arange(Nx)[NA,NA,:],shape);
    zyxs = np.stack([zs,ys,xs],axis=-1)
    print(f"Done")
    return zyxs

def sphere(n):
    xs = np.linspace(-1,1,n)
    return (xs[:,NA,NA]**2 + xs[NA,:,NA]**2 + xs[NA,NA,:]**2) <= 1

def zyx_to_UVWp(zyxs):
    UVWs = (zyxs-cm) @ UVW.T    # uvw in scaled voxel coords
    UVWs*=voxel_size            # UVW in micrometers
    UVWs[:,2] -= W0             # centered on implant backplane
#    UVWps = UVWs @ UVWp.T  # shifted to cylinder center and transformed to U'V'W'
    UVWps = (UVWs-cp) @ UVWp.T  # shifted to cylinder center and transformed to U'V'W'

    return UVWps

def hom_vec(x):
    xh = np.ones((4,),dtype=x.dtype)
    xh[:3] = x
    return xh

def hom_translate(x):
    T = np.eye(4,dtype=float)
    T[0:3,3] = x
    return T

def hom_linear(A):
    M = np.eye(4,dtype=float)
    M[:3,:3] = A
    return M


def zyx_to_UVWp_transform():
    Tcm   = hom_translate(-cm*voxel_size)
    Muvw  = hom_linear(UVW)
    TW0   = hom_translate((0,0,-W0))
    Tcp   = hom_translate(-cp)
    Muvwp = hom_linear(UVWp)    

    return Muvwp @ Tcp @ TW0 @ Muvw @ Tcm



def homogeneous_transform(xs, M):
    shape = np.array(xs.shape)
    assert(shape[-1] == 3)
    shape[-1] = 4
    hxs = np.empty(shape,dtype=xs.dtype)
    hxs[...,:3] = xs;
    hxs[..., 3]  = 1

    print(hxs.shape, M.shape)
    return hxs @ M.T


def np_save(path,data):
    output_dir = os.path.dirname(path)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)        
    np.save(path,data)

    
# Requires: implant-FoR
#           soft-tissue/bone segmentation + blood analysis
#           EDT-field
sample, scale, segment_scale = commandline_args({"sample":"<required>","scale":8,"segment_Scale":1}) #,"segment_scale":8}) #TODO: Implement higher resolution segment scale


h5meta = h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5","r")
h5mask = h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5","r")


Cs_voxel_size     = h5meta["voxels"].attrs["voxelsize"]*segment_scale
mask_voxel_size     = h5meta["voxels"].attrs["voxelsize"]*scale
(Nz,Ny,Nx)     = h5meta["voxels"].shape
Nz            -= np.sum(h5meta["volume_matching_shifts"][:])
(nz,ny,nx)     = np.array([Nz,Ny,Nx])//scale


try:
    h5g = h5meta["implant-FoR"]
except Exception as e:
    print(f"Cant't read implant frame-of-reference: {e}")
    print(f"Make sure you have run segment-implant-cc.py and implant-FoR.py for {sample} at scale {scale}x")
    sys.exit(-1)
    
try:
    blood_mask    = h5mask["blood/mask"][:]
    bone_region   = h5mask["bone_region/mask"][:]
    solid_implantm= h5mask["implant_solid/mask"][:]
    solid_implant = ndi.binary_dilation(solid_implantm, sphere(3))
    implant_mask   = h5mask["implant/mask"][:]
    h5mask.close()    
except Exception as e:
    print(f"Cant't read blood mask for: {e}")
    print("Make sure you have run compute_histograms.py, generate_xx_probabilities.py, segment_from_distributions,\n"+
          "and segment-blood-cc.py")
    sys.exit(-1)


    
    
UVW  = h5g["UVW"][:]
UVWp = h5g["UVWp"][:]
Muvwp = h5g["UVWp_transform"][:]
cm   = h5g["center_of_mass"][:]/voxel_size
cp   = h5g["center_of_cylinder_UVW"][:]
Cp   = h5g["center_of_cylinder_zyx"][:]
W0             = h5g.attrs["backplane_W_shift"]
implant_radius = h5g.attrs["implant_radius"]
theta_range    = h5g["theta_range"][:]


#voxel_binfile         = f"{binary_root}/voxels/{scale}x/{sample}.uint16"
P0_binfile            = f"{binary_root}/segmented/P0/{segment_scale}x/{sample}.uint16"
P1_binfile            = f"{binary_root}/segmented/P1/{segment_scale}x/{sample}.uint16"
edt_binfile           = f"{binary_root}/fields/implant-edt/{scale}x/{sample}.uint16"


M1 = zyx_to_UVWp_transform()
M2 = h5meta["implant-FoR/UVWp_transform"][:]

# print(f"MUvpw1 = {np.round(M1,2)}")
# print(f"MUvpw2 = {np.round(M2,2)}")
# print(f"UVW  = {np.round(UVW,2)}")
# print(f"UVWp = {np.round(UVWp,2)}")
# print(f"Cp = {np.round(Cp,2)}")
# print(f"cp = {np.round(cp,2)}")
# print(f"cm = {np.round(cm,2)}")


# print(f"Loading {segment_scale}x voxels from {voxel_binfile}")
# voxels = np.memmap(voxel_binfile,dtype=np.uint16,mode="r",shape=(nz,ny,nx))

print(f"Loading {segment_scale}x segmentation from {P0_binfile} and {P1_binfile}")
P0 = np.memmap(P0_binfile,dtype=np.uint16,mode="r",shape=(nz,ny,nx))
P1 = np.memmap(P1_binfile,dtype=np.uint16,mode="r",shape=(nz,ny,nx))

edt_field = edt.edt(~solid_implant,parallel=16)*voxel_size

zyxs  = coordinate_image((nz,ny,nx))
UVWps  = zyx_to_UVWp(zyxs)
UVWps2 = homogeneous_transform(zyxs,Muvwp)
Ups, Vps, Wps = UVWps[...,0],UVWps[...,1],UVWps[...,2]
thetas, rs = np.arctan2(Vps,Wps), np.sqrt(Vps**2+Wps**2)    # This is the good reference frame for cylindrical coords

xx = np.linspace(-1,1,nx)
yy = np.linspace(-1,1,ny)
cylinder_mask = (xx[NA,NA,:]**2 + yy[NA,:,NA]**2) < 1


shell_thickness = 60 # micrometers
shell_mask = (edt_field<shell_thickness)&(edt_field>0)&(Wps>0)&cylinder_mask


update_hdf5_mask(f"{hdf5_root}/masks/{scale}x/{sample}.h5",
                 group_name="thin_shell",
                 datasets={"mask":shell_mask},
                 #,
                           # "thetas":shell_thetas2,
                           # "Ups":shell_Ups2,
                           # "indices":shell_indices                 
                 attributes={"shell_thickness":shell_thickness}
)


shell_points_zyx  = np.array(np.nonzero(shell_mask)).T
shell_UVWp = zyx_to_UVWp(shell_points_zyx)
shell_Ups, shell_Vps, shell_Wps = shell_UVWp[...,0],shell_UVWp[...,1],shell_UVWp[...,2]

shell_indices    = np.ravel_multi_index(shell_points_zyx.T,voxels.shape)

shell_thetas = np.arctan2(shell_Vps,shell_Wps)

shell_thetas2     = thetas.reshape(-1)[shell_indices]
shell_Ups2        = Ups.reshape(-1)[shell_indices]
shell_voxels      = voxels.reshape(-1)[shell_indices]

           

# assert(segment_scale <= scale)
# sq = scale//segment_scale
shell_P0      = P0.reshape(-1)[shell_indices]
shell_P1      = P1.reshape(-1)[shell_indices]
shell_blood   = blood_mask.reshape(-1)[shell_indices]

n_th, n_Up = 1800//scale,3200//scale
image  = np.zeros((n_th,n_Up),dtype=np.float32)
blood  = np.zeros((n_th,n_Up),dtype=np.float32)
bone   = np.zeros((n_th,n_Up),dtype=np.float32)
counts = np.zeros((n_th,n_Up),dtype=np.uint64)
th_counts, th_bins = np.histogram(shell_thetas2,n_th)
Up_counts, Up_bins = np.histogram(shell_Ups2,n_Up)

th_min, th_max = th_bins[0], th_bins[-1]
Up_min, Up_max = Up_bins[0], Up_bins[-1]

th_binned = np.floor((n_th-1)*(shell_thetas2 - th_min)/(th_max - th_min)).astype(int)
Up_binned = np.floor((n_Up-1)*(shell_Ups2    - Up_min)/(Up_max - Up_min)).astype(int)

for i in tqdm.tqdm(range(len(shell_indices)),"Binning voxels"):
    blood[th_binned[i],Up_binned[i]]  += shell_P0[i]*shell_blood[i]
    bone[th_binned[i], Up_binned[i]]  += shell_P1[i]
    counts[th_binned[i],Up_binned[i]] = 1

blood /= (counts + (counts==0))
bone /= (counts + (counts==0))



fig = plt.figure(figsize=(20,10))
ax = fig.subplots()
ax.set_xlabel("angle")
ax.set_ylabel("height/micrometer")
ax.set_xticklabels([round(th_min*180/np.pi,2),0,round(th_max*180/np.pi)])
ax.set_xticks([0,n_th/2,n_th-1])
ax.set_yticks(np.arange(0,n_Up,n_Up//10))
ax.set_yticklabels(np.round(Up_bins[np.arange(0,n_Up,n_Up//10)][::-1]-Up_min))
ax.set_title(f"Blood contact to implant surface")
ax.imshow(blood.T[::-1],cmap='Reds')
fig.savefig(f"{hdf5_root}/processed/output/cylinder_projection_blood_{scale}x.png")

fig = plt.figure(figsize=(20,10))
ax = fig.subplots()
ax.set_xlabel("angle")
ax.set_ylabel("height/micrometer")
ax.set_xticklabels([round(th_min*180/np.pi,2),0,round(th_max*180/np.pi)])
ax.set_xticks([0,n_th/2,n_th-1])
ax.set_yticks(np.arange(0,n_Up,n_Up//10))
ax.set_yticklabels(np.round(Up_bins[np.arange(0,n_Up,n_Up//10)][::-1]-Up_min))
ax.set_title(f"Bone contact to implant surface")
ax.imshow(bone.T[::-1],cmap='RdYlBu')
fig.savefig(f"{hdf5_root}/processed/output/cylinder_projection_bone_{scale}x.png")

fig = plt.figure(figsize=(20,10))
ax = fig.subplots()
ax.set_xlabel("angle")
ax.set_ylabel("height/micrometer")
ax.set_xticklabels([round(th_min*180/np.pi,2),0,round(th_max*180/np.pi)])
ax.set_xticks([0,n_th/2,n_th-1])
ax.set_yticks(np.arange(0,n_Up,n_Up//10))
ax.set_yticklabels(np.round(Up_bins[np.arange(0,n_Up,n_Up//10)][::-1]-Up_min))
ax.set_title(f"Combined blood and bone within {shell_thickness} micrometers of implant surface")
ax.imshow((blood-bone).T[::-1],cmap='YlOrRd')
fig.savefig(f"{hdf5_root}/processed/output/cylinder_projection_blood_and_bone_{scale}x.png")


blood_img = P0*blood_mask*shell_mask
bone_img  = P1*shell_mask
voxel_shell_img = voxels*shell_mask

np_save(f"{binary_root}/analysis/shell/blood/{scale}x/{sample}.npy",blood_img)
np_save(f"{binary_root}/analysis/shell/bone/{scale}x/{sample}.npy",bone_img)
np_save(f"{binary_root}/analysis/shell/blood_flat/{scale}x/{sample}.npy",blood)
np_save(f"{binary_root}/analysis/shell/bone_flat/{scale}x/{sample}.npy",bone)

