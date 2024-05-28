import matplotlib
matplotlib.use('Agg')
import h5py, sys, os.path, pathlib, numpy as np, numpy.linalg as la, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root
from lib.cpp.cpu_seq.geometry import compute_front_back_masks
from lib.cpp.gpu.morphology import erode_3d_sphere_bitpacked as erode_3d, dilate_3d_sphere_bitpacked as dilate_3d
from lib.cpp.gpu.bitpacking import encode as bitpacking_encode, decode as bitpacking_decode
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
    I1 = image.copy().astype(np.uint32)
    I2 = np.empty(image.shape, dtype=np.uint32)
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
    #zs, ys, xs = np.broadcast_to(np.arange(Nz)[:,NA,NA],shape),\
    #             np.broadcast_to(np.arange(Ny)[NA,:,NA],shape),\
    #             np.broadcast_to(np.arange(Nx)[NA,NA,:],shape);
    #zyxs = np.stack([zs,ys,xs],axis=-1)
    #del zs, ys, xs
    zyxs = np.moveaxis(np.indices(shape, np.uint16),0,-1)
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

    if verbose >= 1: print(f"Loading {scale}x voxels from {binary_root}/voxels/{scale}x/{sample}.uint16")
    voxels  = np.fromfile(f"{binary_root}/voxels/{scale}x/{sample}.uint16",dtype=np.uint16).reshape(implant.shape)
    plt.imshow(voxels[voxels.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/voxels-xy.png')
    plt.imshow(voxels[:,voxels.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/voxels-xz.png')
    plt.imshow(voxels[:,:,voxels.shape[2]//2]); plt.savefig(f'{image_output_dir}/voxels-yz.png')

    if verbose >= 1: print (f'Loading FoR values from {hdf5_root}/hdf5-byte/msb/{sample}.h5')
    with h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5",'r') as f:
        UVWp = f['implant-FoR/UVWp'][:]
        cp = f['implant-FoR/center_of_cylinder_UVW'][:]
        cm = (f['implant-FoR/center_of_mass'][:]) / voxel_size
        E = f['implant-FoR/E'][:]

    front_mask = np.empty_like(implant, dtype=np.uint8)
    back_mask = np.empty_like(implant, dtype=np.uint8)
    implant_shell_mask = np.empty_like(implant, dtype=np.uint8)
    solid_implant = np.empty_like(implant, dtype=np.uint8)
    compute_front_back_masks(implant, voxel_size, E, cm, cp, UVWp, front_mask, back_mask, implant_shell_mask, solid_implant)
    front_mask = largest_cc_of(front_mask)

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
    peaks, info = signal.find_peaks(hist,height=0.1*hist.max()) # Although, wouldn't the later argsort filter the smaller peaks away anyways?


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
    bone_region_mask_packed = np.empty((nz,ny,nx//32),dtype=np.uint32)
    print (bone_region_mask_packed.shape, bone_mask1.shape)
    bitpacking_encode(bone_mask1.astype(np.uint8), bone_region_mask_packed)

    for i in tqdm.tqdm(range(1),f"Closing with sphere of diameter {closing_diameter} micrometers, {closing_voxels} voxels."):
        bone_region_mask_packed = close_3d(bone_region_mask_packed, closing_voxels//2)

    for i in tqdm.tqdm(range(1),f"Opening with sphere of diameter {opening_diameter} micrometers, {opening_voxels} voxels."):
        bone_region_mask_packed = open_3d(bone_region_mask_packed, opening_voxels//2)


    for i in tqdm.tqdm(range(1),f'Dilating and removing implant with {implant_dilate_diameter} micrometers, {implant_dilate_voxels} voxels.'):
        packed_implant = np.empty((nz, ny, nx//32), dtype=np.uint32)
        bitpacking_encode(solid_implant.astype(np.uint8), packed_implant)
        dilated_implant = np.empty_like(packed_implant, dtype=np.uint32)
        dilate_3d(packed_implant, implant_dilate_voxels, dilated_implant)
        bone_region_mask_packed &= ~dilated_implant

    bone_region_mask = np.empty((nz,ny,nx),dtype=np.uint8)
    bitpacking_decode(bone_region_mask_packed, bone_region_mask)
    bone_region_mask = bone_region_mask.astype(bool)

    bone_region_mask = largest_cc_of(bone_region_mask)

    dilated_implant_unpacked = np.empty((nz,ny,nx),dtype=np.uint8)
    bitpacking_decode(dilated_implant, dilated_implant_unpacked)
    voxels_implanted = voxels.copy()
    voxels_implanted[~dilated_implant_unpacked.astype(bool)] = 0

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