#!/usr/bin/env python3
import os, sys, h5py, numpy as np, pathlib, tqdm, vedo, matplotlib.pyplot as plt, edt, vedo.pointcloud as pc, scipy.ndimage as ndi
sys.path.append(sys.path[0]+"/../")
from config.paths import *
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
    UVWps = (UVWs-cp) #@ UVWp.T  # shifted to cylinder center and transformed to U'V'W'

    return UVWps


sample, scale = commandline_args({"sample":"<required>","scale":8}) #,"segment_scale":8}) #TODO: Implement higher resolution segment scale


h5meta = h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5","r")
h5mask = h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5","r")


voxel_size     = h5meta["voxels"].attrs["voxelsize"]*scale
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
    solid_implant = h5mask["implant_solid/mask"][:]
    implant_mask  = h5mask["implant/mask"][:]
    shell_mask    = h5mask["thin_shell/mask"][:]      
except Exception as e:
    print(f"Cant't read blood mask for: {e}")
    print("Make sure you have run compute_histograms.py, generate_xx_probabilities.py, segment_from_distributions,\n"+
          "and segment-blood-cc.py")
    sys.exit(-1)

    
    
UVW  = h5g["UVW"][:]
UVWp = h5g["UVWp"][:]
[Up_min,Up_max],[Vp_min,Vp_max],[Wp_min,Wp_max] = h5g["bounding_box_UVWp"] 
cm   = h5g["center_of_mass"][:]/voxel_size
cp   = h5g["center_of_cylinder_UVW"][:]
Cp   = h5g["center_of_cylinder_zyx"][:]/voxel_size
W0             = h5g.attrs["backplane_W_shift"]
implant_radius = h5g.attrs["implant_radius"]
theta_range    = h5g["theta_range"][:]

h5meta.close()

voxel_binfile         = f"{binary_root}/voxels/{scale}x/{sample}.uint16"
c0_binfile            = f"{binary_root}/segmented/c0/{scale}x/{sample}.uint16"
c1_binfile            = f"{binary_root}/segmented/c1/{scale}x/{sample}.uint16"
edt_binfile           = f"{binary_root}/fields/implant-edt/{scale}x/{sample}.uint16"

print(f"Loading {scale}x voxels from {voxel_binfile}")
voxels = np.memmap(voxel_binfile,dtype=np.uint16,mode="r",shape=(nz,ny,nx))

print(f"Loading {scale}x segmentation from {c0_binfile} and {c1_binfile}")
c0 = np.memmap(c0_binfile,dtype=np.uint16,mode="r",shape=(nz,ny,nx))
c1 = np.memmap(c1_binfile,dtype=np.uint16,mode="r",shape=(nz,ny,nx))

edt_field = edt.edt(~solid_implant,parallel=16)*voxel_size

blood_img = np.load(f"{binary_root}/analysis/shell/blood/{scale}x/{sample}.npy")
bone_img  = np.load(f"{binary_root}/analysis/shell/bone/{scale}x/{sample}.npy")
blood     = np.load(f"{binary_root}/analysis/shell/blood_flat/{scale}x/{sample}.npy",)
bone      = np.load(f"{binary_root}/analysis/shell/bone_flat/{scale}x/{sample}.npy",)


voxel_shell_img = voxels*shell_mask

blood_vol = vedo.Volume(blood_img,c='YlOrRd',mapper='gpu')
bone_vol  = vedo.Volume(bone_img,c='YlOrRd_r',mapper='gpu')
#bloodbone_vol  = vedo.Volume(blood_img-bone_img,c='YlOrRd_r',mapper='gpu')
implant_vol = vedo.Volume(implant_mask*voxels,c='PuBu_r',mapper='gpu')
voxel_shell_vol  = vedo.Volume(shell_mask*voxels,c='YlOrRd_r',mapper='gpu')

rotations_per_dataset = 2
seconds_per_rotation = 8
frames_per_second = 30

pp = vedo.Plotter(interactive=False)
pp.camera.SetFocalPoint(cm[::-1])
pp.camera.SetViewAngle(0)
pp.camera.SetRoll(0)
pp.camera.SetPosition((cm[2],0,cm[0]))
pp.camera.SetViewUp(UVW[0][::-1])
pp.camera.Elevation(0)
pp.camera.Pitch(0)
pp.camera.Roll(0)
pp.camera.Yaw(0)
pp.camera.Azimuth(180)

video = vedo.Video(f"{hdf5_root}/processed/output/video2-implant_contact_{scale}x.avi", backend='ffmpeg', fps=frames_per_second)

txt1 = vedo.Text2D("A 60mu shell around the implant is computed by a distance transform", pos='top-center')
txt2 = vedo.Text2D("Blood contact with implant surface", pos='top-center')
txt3 = vedo.Text2D("Bone contact with implant surface", pos='top-center')
txt4 = vedo.Text2D("Blood/bone contact to implant surface", pos='top-center')
txt5 = vedo.Text2D("To flatten the curved surface, we project onto a cylinder")

from time import sleep



Up = (UVW @ UVWp)[0]
cyl1 = vedo.Cylinder(Cp[::-1], r=1.05*implant_radius/voxel_size,height=(Up_max-Up_min)/voxel_size,axis=Up[::-1],alpha=0.1)
cyl2 = vedo.Cylinder(Cp[::-1], r=implant_radius/(30*voxel_size),height=(Up_max-Up_min)/voxel_size,axis=Up[::-1],c='r',alpha=0.4)

cyl_img1 = vedo.Picture(f"{hdf5_root}/processed/output/cp_bone_blood_gimp.png").rotateX(90)

       
for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second//2),"Step 1: implant"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt1,implant_vol])
    video.addFrame()
    pp.clear()

for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 2: shell"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt1,voxel_shell_vol])
    video.addFrame()
    pp.clear()        

    
for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 3: surface blood"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt2,blood_vol])
    video.addFrame()
    pp.clear()
        
for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 4: surface bone"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt3,bone_vol])
    video.addFrame()
    pp.clear()

for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 5: surface both"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt4,blood_vol,bone_vol])
    video.addFrame()
    pp.clear()
        

for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 6: cylinder"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt5],[bone_vol,blood_vol,cyl1,cyl2])
    video.addFrame()
    pp.clear()
   

for _ in tqdm.tqdm(range(3*seconds_per_rotation * frames_per_second),"Step 7: flat"):
    pp.show([txt5,cyl_img1])
    video.addFrame()
    pp.clear()
        

video.close()
pp.close()
