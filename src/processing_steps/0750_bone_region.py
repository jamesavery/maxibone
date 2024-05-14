import matplotlib
matplotlib.use('Agg')
import h5py, sys, os.path, pathlib, numpy as np, numpy.linalg as la, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root
from lib.cpp.gpu.morphology import erode_3d_sphere as erode_3d, dilate_3d_sphere as dilate_3d
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import scipy as sp, scipy.ndimage as ndi, scipy.interpolate as interpolate, scipy.signal as signal
import vedo, vedo.pointcloud as pc
from lib.py.helpers import update_hdf5, update_hdf5_mask, commandline_args
from numpy import array, newaxis as NA
from scipy.ndimage import gaussian_filter1d

# close = dilate then erode
# open = erode then dilate
def morph_3d(image, r, fa, fb):
    I1 = image.copy().astype(np.uint8)
    I2 = np.empty(image.shape, dtype=np.uint8)
    rmin = 16
    rmins = r // rmin
    rrest = r % rmin
    for _ in range(rmins):
        dilate_3d(I1, rmin, I2)
        I1, I2 = I2, I1
    if rrest > 0:
        dilate_3d(I1, rrest, I2)
        I1, I2 = I2, I1

    for i in range(rmins):
        erode_3d(I1, rmin, I2)
        I1, I2 = I2, I1
    if rrest > 0:
        erode_3d(I1, rrest, I2)
        I1, I2 = I2, I1

    return I1

def close_3d(image, r):
    return morph_3d(image, r, dilate_3d, erode_3d)

def open_3d(image, r):
    return morph_3d(image, r, erode_3d, dilate_3d)

def coordinate_image(shape):
    Nz,Ny,Nx   = shape
    if verbose >= 1: print(f"Broadcasting coordinates for {shape} image")
    zs, ys, xs = np.broadcast_to(np.arange(Nz)[:,NA,NA],shape),\
                 np.broadcast_to(np.arange(Ny)[NA,:,NA],shape),\
                 np.broadcast_to(np.arange(Nx)[NA,NA,:],shape);
    zyxs = np.stack([zs,ys,xs],axis=-1)
    if verbose >= 1: print(f"Done")
    return zyxs

def largest_cc_of(mask):
    label, n_features = ndi.label(mask)
    bincnts           = np.bincount(label[label>0],minlength=n_features+1)

    largest_cc_ix   = np.argmax(bincnts)
    return (label==largest_cc_ix)

if __name__ == "__main__":
    sample, scale, verbose = commandline_args({"sample" : "<required>",
                                               "scale" : 8,
                                               "verbose" : 1})

    image_output_dir = f"{hdf5_root}/processed/bone_region/{sample}/"
    if verbose >= 1: print(f"Storing all debug-images to {image_output_dir}")
    pathlib.Path(image_output_dir).mkdir(parents=True, exist_ok=True)

    if verbose >= 1: print(f"Loading {scale}x implant mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
    implant_file = h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5",'r')
    implant      = implant_file["implant/mask"][:].astype(np.uint8)
    voxel_size   = implant_file["implant"].attrs["voxel_size"]
    implant_file.close()

    nz, ny, nx = implant.shape

    #rmaxs, rs, Ws her!
    if verbose >= 1: print (f'Loading FoR values from {hdf5_root}/hdf5-byte/msb/{sample}.h5')
    with h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5",'r') as f:
        UVWp = f['implant-FoR/UVWp'][:]
        cp = f['implant-FoR/center_of_cylinder_UVW'][:]
        cm = (f['implant-FoR/center_of_mass'][:]) / voxel_size
        E = f['implant-FoR/E'][:]

    implant_zyxs = np.array(np.nonzero(implant)).T - cm   # Implant points in z,y,x-coordinates (relative to upper-left-left corner, in {scale}x voxel units)
    implant_uvws = implant_zyxs @ E                       # Implant points in u,v,w-coordinates (relative to origin cm, in {scale}x voxel units)

    w0  = implant_uvws[:,2].min();  # In {scale}x voxel units
    w0v = np.array([0,0,w0])        # w-shift to get to center of implant back-plane

    zyxs = coordinate_image(implant.shape)
    uvws = (zyxs - cm) @ E                  # raw voxel-scale relative to center of mass
    UVWs = (uvws - w0v) * voxel_size        # Micrometer scale relative to backplane-center
    UVWps = (UVWs - cp) @ UVWp                # relative to center-of-implant-before-sawing-in-half
    Ups,Vps,Wps = UVWps[...,0], UVWps[...,1], UVWps[...,2]      # U',V',W' physical image coordinates
    thetas, rs = np.arctan2(Vps,Wps), np.sqrt(Vps**2+Wps**2)    # This is the good reference frame for cylindrical coords
    rmaxs = (rs*(implant==True)).reshape(nz,-1).max(axis=1)[:,NA,NA]
    Us,Vs,Ws = UVWs[...,0], UVWs[...,1], UVWs[...,2]        # UVW physical image coordinates

    if verbose >= 1: print(f"Loading {scale}x voxels from {binary_root}/voxels/{scale}x/{sample}.uint16")
    voxels  = np.fromfile(f"{binary_root}/voxels/{scale}x/{sample}.uint16",dtype=np.uint16).reshape(implant.shape)

    implant_shell_mask = implant&(rs >= 0.7*rmaxs)
    solid_implant = (implant | (rs < 0.7*rmaxs) & (Ws >= 0))

    back_mask  = (Ws<0)
    front_mask = largest_cc_of((Ws>50)&(~solid_implant))#*(thetas>=theta_from)*(thetas<=theta_to)
    plt.imshow(front_mask[front_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-front_mask.png')
    plt.imshow(front_mask[:,front_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-front_mask.png')
    plt.imshow(front_mask[:,:,front_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-front_mask.png')

    # back_part = voxels*back_mask

    front_part = voxels*front_mask

    output_dir = f"{hdf5_root}/masks/{scale}x/"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if verbose >= 1: print(f"Saving implant_solid mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="implant_solid",
                     datasets={"mask":solid_implant},
                     attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plt.imshow(solid_implant[solid_implant.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-solid.png')
    plt.imshow(solid_implant[:,solid_implant.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-solid.png')
    plt.imshow(solid_implant[:,:,solid_implant.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-solid.png')

    if verbose >= 1: print(f"Saving implant_shell mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="implant_shell",
                     datasets={"mask":implant_shell_mask},
                     attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plt.imshow(implant_shell_mask[implant_shell_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-shell.png')
    plt.imshow(implant_shell_mask[:,implant_shell_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-shell.png')
    plt.imshow(implant_shell_mask[:,:,implant_shell_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-shell.png')

    if verbose >= 1: print(f"Saving cut_cylinder_air mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="cut_cylinder_air",
                     datasets={"mask":back_mask},
                     attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plt.imshow(back_mask[back_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-back.png')
    plt.imshow(back_mask[:,back_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-back.png')
    plt.imshow(back_mask[:,:,back_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-back.png')

    if verbose >= 1: print(f"Saving cut_cylinder_bone mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="cut_cylinder_bone",
                     datasets={"mask":front_mask},
                     attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})
    plt.imshow(front_mask[front_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-front.png')
    plt.imshow(front_mask[:,front_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-front.png')
    plt.imshow(front_mask[:,:,front_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-front.png')

    if verbose >= 1: print(f"Computing bone region")
    hist, bins = np.histogram(front_part, 256)
    hist[0] = 0
    hist = gaussian_filter1d(hist, 3)
    peaks, info = signal.find_peaks(hist,height=0.1*hist.max())


    if verbose >= 1: plt.clf(); plt.plot(bins[1:],hist); plt.savefig(f'{image_output_dir}/bone_histogram.png')
    print (f'peaks: {peaks}')

    p1, p2 = peaks[np.argsort(info['peak_heights'])[:2]]
    midpoint = int(round((bins[p1]+bins[p2+1])/2)) # p1 is left-edge of p1-bin, p2+1 is right edge of p2-bin
    if verbose >= 1: print(f"p1, p2 = ({p1,bins[p1]}), ({p2,bins[p2]}); midpoint = {midpoint}")

    bone_mask1 = front_part > midpoint
    plt.imshow(bone_mask1[bone_mask1.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-bone1.png')
    plt.imshow(bone_mask1[:,bone_mask1.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-bone1.png')
    plt.imshow(bone_mask1[:,:,bone_mask1.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-bone1.png')

    closing_diameter, opening_diameter, implant_dilate_diameter = 400, 300, 10           # micrometers
    closing_voxels = 2*int(round(closing_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length
    opening_voxels = 2*int(round(opening_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length
    implant_dilate_voxels = 2*int(round(implant_dilate_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length

    for i in tqdm.tqdm(range(1),f"Closing with sphere of diameter {closing_diameter} micrometers, {closing_voxels} voxels."):
        bone_region_mask = close_3d(bone_mask1, closing_voxels//2)

    for i in tqdm.tqdm(range(1),f"Opening with sphere of diameter {opening_diameter} micrometers, {opening_voxels} voxels."):
        bone_region_mask = open_3d(bone_region_mask, opening_voxels//2)

    for i in tqdm.tqdm(range(1),f'Dilating and removing implant with {implant_dilate_diameter} micrometers, {implant_dilate_voxels} voxels.'):
        dilated_implant = np.empty(solid_implant.shape, dtype=np.uint8)
        dilate_3d(solid_implant, implant_dilate_voxels, dilated_implant)
        bone_region_mask &= ~(dilated_implant.astype(bool))

    bone_region_mask = largest_cc_of(bone_region_mask)
    voxels_implanted = voxels.copy()
    voxels_implanted[~dilated_implant.astype(bool)] = 0

    plt.imshow(voxels_implanted[voxels_implanted.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-dilated-implant.png')
    plt.imshow(voxels_implanted[:,voxels_implanted.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-dilated-implant.png')
    plt.imshow(voxels_implanted[:,:,voxels_implanted.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-dilated-implant.png')

    plt.imshow(bone_region_mask[bone_region_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-bone.png')
    plt.imshow(bone_region_mask[:,bone_region_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-bone.png')
    plt.imshow(bone_region_mask[:,:,bone_region_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-bone.png')
    voxels_boned = voxels.copy()
    voxels_boned[~bone_region_mask] = 0
    plt.imshow(voxels_boned[voxels_boned.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/voxels-boned-xy.png')
    plt.imshow(voxels_boned[:,voxels_boned.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/voxels-boned-xz.png')
    plt.imshow(voxels_boned[:,:,voxels_boned.shape[2]//2]); plt.savefig(f'{image_output_dir}/voxels-boned-yz.png')

    if verbose >= 1: print(f"Saving bone_region mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                        group_name="bone_region",
                        datasets={"mask":bone_region_mask},
                        attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})