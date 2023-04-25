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
    cut_cylinder_air  = h5mask["cut_cylinder_air/mask"][:]
    cut_cylinder_bone = h5mask["cut_cylinder_bone/mask"][:]
    implant_mask  = h5mask["implant/mask"][:]
    solid_implant = h5mask["implant_solid/mask"][:]
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

blood_img = c0*blood_mask
bone_img  = c1 

full_vol = vedo.Volume(voxels,c='YlOrRd',mapper='gpu')
blood_vol = vedo.Volume(blood_img,c='YlOrRd',mapper='gpu')
bone_vol  = vedo.Volume(bone_img,c='YlOrRd_r',mapper='gpu')
implant_vol = vedo.Volume(implant_mask*voxels,c='PuBu_r',mapper='gpu')
air_vol   = vedo.Volume(cut_cylinder_air*voxels,c='Blues',alpha=[0,0.01,0.1,0.2],mapper='gpu')
front_vol = vedo.Volume(cut_cylinder_bone*voxels,mapper='gpu')
br_vol    = vedo.Volume(bone_region*voxels,mapper='gpu')
resin_vol = vedo.Volume(((~bone_region[:])&cut_cylinder_bone[:])*voxels,c='PuBu_r',alpha=[0,0.05,0.15,0.3],mapper='gpu')
#osteocyte_vol = vedo.Volume(c0*(~blood_mask),c='YlOrRd',mapper='gpu')

rotations_per_dataset = 1
seconds_per_rotation = 8
frames_per_second = 30

pp = vedo.Plotter(interactive=False)
pp.camera.SetFocalPoint(cm[::-1])
pp.camera.SetViewAngle(0)
pp.camera.SetRoll(0)
pp.camera.SetPosition((cm[2],0,cm[0]))
#pp.camera.SetViewUp(UVW[0][::-1])
pp.camera.SetViewUp((0,0,-1))
pp.camera.Elevation(0)
pp.camera.Pitch(0)
pp.camera.Roll(0)
pp.camera.Yaw(0)
pp.camera.Azimuth(180)

video = vedo.Video(f"{hdf5_root}/processed/output/video1-segmentation_{scale}x.avi", backend='ffmpeg', fps=frames_per_second)

txt1 = vedo.Text2D("Each sample is imaged in 1.875mu-resolution yielding 3456x3456x3456 3D images", pos='top-center')
txt2 = vedo.Text2D("Implant principal axes coordinate system ", pos='top-center')
txt3 = vedo.Text2D("Samples were sawed through for histology, leaving half-cylinder", pos='top-center')
txt4 = vedo.Text2D("Acrylic resin is separated from bone region", pos='top-center')
txt5 = vedo.Text2D("New tissue classification algorithm is used within bone region")
txt6 = vedo.Text2D("Network analysis yields blood vessel network; Shape analysis yields osteocytes.")


uvw =  (UVW @ UVWp)
Up, Vp, Wp = uvw[0], uvw[1], uvw[2]
cyl1 = vedo.Cylinder(Cp[::-1], r=1.05*implant_radius/voxel_size,height=(Up_max-Up_min)/voxel_size,axis=Up[::-1],alpha=0.1)
cyl2 = vedo.Cylinder(Cp[::-1], r=implant_radius/(30*voxel_size),height=(Up_max-Up_min)/voxel_size,axis=Up[::-1],c='r',alpha=0.4)

diffusion_img = vedo.Picture(f"{hdf5_root}/processed/implant-edt/2x/770c_pag-yz.png").rotateX(270)
cyl_img = vedo.Picture(f"{hdf5_root}/processed/output/cp_bone_blood_gimp.png").rotateX(270)

a1 = vedo.Arrow(cm[::-1], cm[::1]+Up[::-1]*1.5*Up_max/voxel_size,c='r',s=1)
a2 = vedo.Arrow(cm[::-1], cm[::1]+Vp[::-1]*2*implant_radius/voxel_size,c='g',s=1)
a3 = vedo.Arrow(cm[::-1], cm[::1]+Wp[::-1]*implant_radius/voxel_size,c='b',s=1)
cyl2 = vedo.Cylinder(Cp[::-1], r=implant_radius/(30*voxel_size),height=(Up_max-Up_min)/voxel_size,axis=Up[::-1],c='r',alpha=0.4)


pp.camera.Azimuth((180))

    
for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 1: Full"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt1,full_vol])
    video.addFrame()
    pp.clear()
       
for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 2: Implant FoR"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt2,air_vol,implant_vol,a1,a2,a3])
    video.addFrame()
    pp.clear()

for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 3: Air/Front"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt3,air_vol,front_vol])
    video.addFrame()
    pp.clear()    

for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 4: Front"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt3,front_vol])
    video.addFrame()
    pp.clear()

for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 5: Resin"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt4,br_vol, resin_vol])
    video.addFrame()
    pp.clear()

for _ in tqdm.tqdm(range(seconds_per_rotation * frames_per_second),"Step 6: Bone region"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt5,br_vol])
    video.addFrame()
    pp.clear()

for _ in tqdm.tqdm(range(2*seconds_per_rotation * frames_per_second),"Step 7: Bone mask"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt5,bone_vol])
    video.addFrame()
    pp.clear()

for _ in tqdm.tqdm(range(2*seconds_per_rotation * frames_per_second),"Step 8: Blood mask"):
    pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
    pp.show([txt6,blood_vol])
    video.addFrame()
    pp.clear()         
    

# for _ in range(seconds_per_rotation * frames_per_second//4):
#     pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
#     pp.show([full_vol])
#     video.addFrame()
#     pp.clear()    


# for _ in range(seconds_per_rotation * frames_per_second//4):
#     pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
#     pp.show([implant_vol,bone_vol])
#     video.addFrame()
#     pp.clear()

# for _ in range((3*seconds_per_rotation * frames_per_second)//4):
#     pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
#     pp.show([bone_vol])
#     video.addFrame()
#     pp.clear()    

# for _ in range(seconds_per_rotation * frames_per_second//3):
#     pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
#     pp.show([implant_vol,blood_vol])
#     video.addFrame()
#     pp.clear()

# for _ in range((54*seconds_per_rotation * frames_per_second)//2):
#     pp.camera.Azimuth((360 / (frames_per_second * seconds_per_rotation)))
#     pp.show([blood_vol])
#     video.addFrame()
#     pp.clear()

# distance = pp.camera.GetDistance()
# for _ in range(seconds_per_rotation * frames_per_second):
#     pp.camera.SetDistance(distance*(1-0.5/(seconds_per_rotation * frames_per_second)))
#     pp.show([blood_vol])
#     video.addFrame()

video.close()
pp.close()
