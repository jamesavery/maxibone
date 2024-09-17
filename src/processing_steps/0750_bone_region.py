import matplotlib
matplotlib.use('Agg')
import h5py, sys, os.path, pathlib, numpy as np, numpy.linalg as la, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root
from lib.cpp.cpu.geometry import compute_front_back_masks
from lib.cpp.gpu.morphology import erode_3d_sphere as erode_3d, dilate_3d_sphere as dilate_3d, erode_3d_sphere_bitpacked as erode_3d_bitpacked, dilate_3d_sphere_bitpacked as dilate_3d_bitpacked
from lib.cpp.gpu.bitpacking import encode as bitpacking_encode, decode as bitpacking_decode
from lib.cpp.cpu.connected_components import largest_connected_component
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import scipy as sp, scipy.ndimage as ndi, scipy.interpolate as interpolate, scipy.signal as signal
import vedo, vedo.pointcloud as pc
from lib.py.helpers import update_hdf5, update_hdf5_mask, commandline_args
from numpy import array, newaxis as NA
from scipy.ndimage import gaussian_filter1d
import datetime
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from functools import partial

# close = dilate then erode
# open = erode then dilate
def morph_3d(image, r, fa, fb):
    I1 = image.copy().astype(image.dtype)
    I2 = np.empty(image.shape, dtype=image.dtype)
    rmin = 16
    rmins = r // rmin
    rrest = r % rmin
    for _ in range(rmins):
        fa(I1, rmin, I2)
        I1, I2 = I2, I1
    if rrest > 0:
        fa(I1, rrest, I2)
        I1, I2 = I2, I1

    for i in range(rmins):
        fb(I1, rmin, I2)
        I1, I2 = I2, I1
    if rrest > 0:
        fb(I1, rrest, I2)
        I1, I2 = I2, I1

    return I1

def close_3d(image, r):
    if image.dtype == np.uint32:
        return morph_3d(image, r, dilate_3d_bitpacked, erode_3d_bitpacked)
    else:
        return morph_3d(image, r, dilate_3d, erode_3d)

def open_3d(image, r):
    if image.dtype == np.uint32:
        return morph_3d(image, r, erode_3d_bitpacked, dilate_3d_bitpacked)
    else:
        return morph_3d(image, r, erode_3d, dilate_3d)

def label_chunk(i, chunk, chunk_prefix):
    label, n_features = ndi.label(chunk, output=np.int64)
    label.tofile(f'{chunk_prefix}{i}.int64')
    del label
    return n_features

def largest_cc_of(mask, mask_name):
    nz, ny, nx = mask.shape
    flat_size = nz*ny*nx
    layer_size = ny*nx
    n_cores = mp.cpu_count() // 2 # Only count physical cores
    available_memory = 1024**3 * 4 * n_cores # 1 GB per core-ish
    memory_per_core  = available_memory // n_cores
    elements_per_core = memory_per_core // 8 # 8 bytes per element
    layers_per_core = elements_per_core // layer_size
    n_chunks = max(1, int(2**np.ceil(np.log2(nz // layers_per_core))))
    layers_per_chunk = nz // n_chunks
    intermediate_folder = f"/tmp/maxibone/labels_bone_region_{mask_name}/{scale}x/"
    os.makedirs(intermediate_folder, exist_ok=True)

    if layers_per_chunk == 0 or layers_per_chunk >= nz:
        label, n_features = ndi.label(mask, output=np.int64)
        bincnts           = np.bincount(label[label>0],minlength=n_features+1)
        largest_cc_ix     = np.argmax(bincnts)
        return (label == largest_cc_ix)
    else:
        start = datetime.datetime.now()
        with ThreadPool(n_cores) as pool:
            label_chunk_partial = partial(label_chunk, chunk_prefix=f"{intermediate_folder}/{sample}_")
            chunks = [mask[i*layers_per_chunk:(i+1)*layers_per_chunk] for i in range(n_chunks-1)]
            chunks.append(mask[(n_chunks-1)*layers_per_chunk:])
            n_labels = pool.starmap(label_chunk_partial, enumerate(chunks))
            for chunk in chunks:
                del chunk
            del chunks
        end = datetime.datetime.now()
        # load uint16, threshold (uint16 > uint8), label (int64), write int64
        total_bytes_processed = flat_size*2 + flat_size*2 + flat_size*8 + flat_size*8
        gb_per_second = total_bytes_processed / (end-start).total_seconds() / 1024**3
        print (f'Loading and labelling {mask_name} took {end-start}. (throughput: {gb_per_second:.02f} GB/s)')

        np.array(n_labels, dtype=np.int64).tofile(f"{intermediate_folder}/{sample}_n_labels.int64")

        largest_component = np.zeros((nz,ny,nx),dtype=bool)
        largest_connected_component(largest_component, f"{intermediate_folder}/{sample}_", n_labels, (nz,ny,nx), (layers_per_chunk,ny,nx), True)

        return largest_component

if __name__ == "__main__":
    sample, scale, verbose = commandline_args({"sample" : "<required>",
                                               "scale" : 8,
                                               "verbose" : 1})

    image_output_dir = f"{hdf5_root}/processed/bone_region/{sample}/{scale}x/"
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
    plt.imshow(voxels[voxels.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/voxels-xy.png', bbox_inches='tight')
    plt.imshow(voxels[:,voxels.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/voxels-xz.png', bbox_inches='tight')
    plt.imshow(voxels[:,:,voxels.shape[2]//2]); plt.savefig(f'{image_output_dir}/voxels-yz.png', bbox_inches='tight')

    if verbose >= 1: print (f'Loading FoR values from {hdf5_root}/hdf5-byte/msb/{sample}.h5')
    with h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5",'r') as f:
        UVWp = f['implant-FoR/UVWp'][:]
        cp = f['implant-FoR/center_of_cylinder_UVW'][:]
        cm = (f['implant-FoR/center_of_mass'][:]) / voxel_size
        E = f['implant-FoR/E'][:]

    if verbose >= 1: print(f"Computing front/back/implant_shell/solid_implant masks")
    front_mask = np.empty_like(implant, dtype=np.uint8)
    back_mask = np.empty_like(implant, dtype=np.uint8)
    implant_shell_mask = np.empty_like(implant, dtype=np.uint8)
    solid_implant = np.empty_like(implant, dtype=np.uint8)
    if verbose >= 1: start = datetime.datetime.now()
    compute_front_back_masks(implant, voxel_size, E, cm, cp, UVWp, front_mask, back_mask, implant_shell_mask, solid_implant)
    if verbose >= 1: end = datetime.datetime.now()
    if verbose >= 1: print (f'Computing front/back/implant_shell/solid_implant masks took {end-start}')

    front_mask = largest_cc_of(front_mask, 'front')

    # back_part = voxels*back_mask

    front_part = voxels*front_mask

    output_dir = f"{hdf5_root}/masks/{scale}x/"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if verbose >= 1: print(f"Saving implant_solid mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="implant_solid",
                     datasets={"mask":solid_implant},
                     attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plt.imshow(solid_implant[solid_implant.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-solid.png', bbox_inches='tight')
    plt.imshow(solid_implant[:,solid_implant.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-solid.png', bbox_inches='tight')
    plt.imshow(solid_implant[:,:,solid_implant.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-solid.png', bbox_inches='tight')

    if verbose >= 1: print(f"Saving implant_shell mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="implant_shell",
                     datasets={"mask":implant_shell_mask},
                     attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plt.imshow(implant_shell_mask[implant_shell_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-shell.png', bbox_inches='tight')
    plt.imshow(implant_shell_mask[:,implant_shell_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-shell.png', bbox_inches='tight')
    plt.imshow(implant_shell_mask[:,:,implant_shell_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-shell.png', bbox_inches='tight')
    del implant_shell_mask

    if verbose >= 1: print(f"Saving cut_cylinder_air mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="cut_cylinder_air",
                     datasets={"mask":back_mask},
                     attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plt.imshow(back_mask[back_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-back.png', bbox_inches='tight')
    plt.imshow(back_mask[:,back_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-back.png', bbox_inches='tight')
    plt.imshow(back_mask[:,:,back_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-back.png', bbox_inches='tight')
    del back_mask

    if verbose >= 1: print(f"Saving cut_cylinder_bone mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="cut_cylinder_bone",
                     datasets={"mask":front_mask},
                     attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})
    plt.imshow(front_mask[front_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-front.png', bbox_inches='tight')
    plt.imshow(front_mask[:,front_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-front.png', bbox_inches='tight')
    plt.imshow(front_mask[:,:,front_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-front.png', bbox_inches='tight')
    del front_mask
    front_part_implanted = front_part.copy()
    front_part_implanted[implant == 1] = 0
    fpmin = front_part_implanted
    fpmin[fpmin==0] = 65535
    vmin = fpmin.min()
    fpmin[fpmin==65535] = vmin
    vmax = fpmin.max()
    del fpmin, front_part_implanted

    if verbose >= 1: print(f"Computing bone region")
    hist, bins = np.histogram(front_part, 2048, range=(vmin,vmax))
    hist[0] = 0
    hist_raw = hist.copy()
    hist = gaussian_filter1d(hist, 3)
    peaks, info = signal.find_peaks(hist,height=0.1*hist.max()) # Although, wouldn't the later argsort filter the smaller peaks away anyways?

    if verbose >= 1:  plt.figure(figsize=(20,10)); plt.plot(bins[1:], hist_raw); plt.plot(bins[1:], hist); plt.savefig(f'{image_output_dir}/bone_histogram.png', bbox_inches='tight'); plt.clf()
    print (f'peaks: {peaks}')

    two_largest_peaks = peaks[np.argsort(info['peak_heights'])[::-1][:2]]
    p1, p2 = sorted(two_largest_peaks)
    #midpoint = int(round((bins[p1]+bins[p2+1])/2)) # p1 is left-edge of p1-bin, p2+1 is right edge of p2-bin
    midpoint = bins[np.argmin(hist[p1:p2]) + p1]
    if verbose >= 1: print(f"p1, p2 = ({p1,bins[p1]}), ({p2,bins[p2]}); midpoint = {midpoint}")

    bone_mask1 = front_part > midpoint
    del front_part
    plt.imshow(bone_mask1[bone_mask1.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-bone1.png', bbox_inches='tight')
    plt.imshow(bone_mask1[:,bone_mask1.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-bone1.png', bbox_inches='tight')
    plt.imshow(bone_mask1[:,:,bone_mask1.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-bone1.png', bbox_inches='tight')

    if 'novisim' in sample:
        closing_diameter, opening_diameter, implant_dilate_diameter = 400, 300, 15           # micrometers
    else:
        closing_diameter, opening_diameter, implant_dilate_diameter = 400, 300, 5           # micrometers
    closing_voxels = 2*int(round(closing_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length
    opening_voxels = 2*int(round(opening_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length
    implant_dilate_voxels = 2*int(round(implant_dilate_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length
    bitpacked = nx % 32 == 0
    if bitpacked:
        bone_region_tmp = np.empty((nz,ny,nx//32),dtype=np.uint32)
        bitpacking_encode(bone_mask1.astype(np.uint8), bone_region_tmp)
    else:
        bone_region_tmp = bone_mask1.astype(np.uint8)
    del bone_mask1

    for i in tqdm.tqdm(range(1),f"Closing with sphere of diameter {closing_diameter} micrometers, {closing_voxels} voxels."):
        bone_region_tmp = close_3d(bone_region_tmp, closing_voxels//2)


    for i in tqdm.tqdm(range(1),f"Opening with sphere of diameter {opening_diameter} micrometers, {opening_voxels} voxels."):
        bone_region_tmp = open_3d(bone_region_tmp, opening_voxels//2)

    for i in tqdm.tqdm(range(1),f'Dilating and removing implant with {implant_dilate_diameter} micrometers, {implant_dilate_voxels} voxels.'):
        if bitpacked:
            packed_implant = np.empty((nz, ny, nx//32), dtype=np.uint32)
            bitpacking_encode(solid_implant.astype(np.uint8), packed_implant)
        else:
            packed_implant = solid_implant
        del solid_implant

        dilated_implant = np.empty_like(packed_implant, dtype=packed_implant.dtype)
        if bitpacked:
            dilate_3d_bitpacked(packed_implant, implant_dilate_voxels, dilated_implant)
        else:
            dilate_3d(packed_implant, implant_dilate_voxels, dilated_implant)
        bone_region_tmp &= ~dilated_implant

    if bitpacked:
        bone_region_mask = np.empty((nz,ny,nx),dtype=np.uint8)
        bitpacking_decode(bone_region_tmp, bone_region_mask)
        bone_region_mask = bone_region_mask.astype(bool)
    else:
        bone_region_mask = bone_region_tmp.astype(bool)
    del bone_region_tmp

    bone_region_mask = largest_cc_of(bone_region_mask, 'bone_region')

    if verbose >= 2:
        if bitpacked:
            dilated_implant_unpacked = np.empty((nz,ny,nx),dtype=np.uint8)
            print (dilated_implant.shape, dilated_implant_unpacked.shape)
            bitpacking_decode(dilated_implant, dilated_implant_unpacked)
        else:
            dilated_implant_unpacked = dilated_implant
        voxels_implanted = voxels.copy()
        voxels_implanted[dilated_implant_unpacked == 0] = 0

        plt.imshow(voxels_implanted[voxels_implanted.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-dilated-implant.png', bbox_inches='tight')
        plt.imshow(voxels_implanted[:,voxels_implanted.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-dilated-implant.png', bbox_inches='tight')
        plt.imshow(voxels_implanted[:,:,voxels_implanted.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-dilated-implant.png', bbox_inches='tight')

    plt.imshow(bone_region_mask[bone_region_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-bone.png', bbox_inches='tight')
    plt.imshow(bone_region_mask[:,bone_region_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-bone.png', bbox_inches='tight')
    plt.imshow(bone_region_mask[:,:,bone_region_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-bone.png', bbox_inches='tight')

    voxels[~bone_region_mask] = 0
    plt.imshow(voxels[voxels.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/voxels-boned-xy.png', bbox_inches='tight')
    plt.imshow(voxels[:,voxels.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/voxels-boned-xz.png', bbox_inches='tight')
    plt.imshow(voxels[:,:,voxels.shape[2]//2]); plt.savefig(f'{image_output_dir}/voxels-boned-yz.png', bbox_inches='tight')

    if verbose >= 1: print(f"Saving bone_region mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                        group_name="bone_region",
                        datasets={"mask":bone_region_mask},
                        attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})