#/usr/bin/env python3
# TODO:
# 1] shadow effects could perhaps be removed by histogram matching of final adjacent regions
#    - should probably be removed by kernel-operation around outer edge, before matching...

###########################
### Fix shifted volumes ###
###########################

import h5py, sys, jax, os.path, pathlib, tqdm
import numpy as np
import jax.numpy as jp
import h5py, jax, sys
from PIL import Image
sys.path.append(sys.path[0]+"/../")
from config.paths import hdf5_root
from lib.py.helpers import commandline_args

verbose = 1
volume_matched_dir = f"{hdf5_root}/processed/volume_matched"

def match_region(voxels_top, voxels_bot, overlap, max_shift):
    """
    Find shift that minimizes squared differences with overlap <= shift <= max_shift
    """
    # Shifts smaller than the overlap overlap with shift
    slice_size = voxels_top.shape[1]*voxels_top.shape[2] # Normalize by number of voxels (to make sums_lt and sums_ge comparable)
    sums_lt = jp.array( [ jp.sum(((voxels_top[-shift:] - voxels_bot[0:shift])/(shift*slice_size))**2)
                          for shift in tqdm.tqdm(range(2,overlap),f"Mathcings shifted 0 to {overlap}")] )
    # Shifts larger than the overlap overlap with overlap
    sums_ge = jp.array( [ jp.sum(((voxels_top[-overlap:] - voxels_bot[shift:shift+overlap])/(overlap*slice_size))**2)
                          for shift in tqdm.tqdm(range(0,max_shift-overlap),f"Mathcings shifted {overlap} to {max_shift}")] )
    # print("sums_lt=",sums_lt)
    # print("sums_ge=",sums_ge)
    sums = jp.concatenate([sums_lt, sums_ge])
    result = jp.argmin( sums ), jp.sqrt(sums.min())
    del sums
    del sums_lt
    del sums_ge
    return result


def match_all_regions(voxels,crossings,write_image_checks=True):
    shifts = np.zeros(len(crossings),dtype=np.int32)
    errors = np.zeros(len(crossings),dtype=np.float32)
    match_region_jit = jax.jit(match_region,static_argnums=(2,3));

    if verbose >= 1: print(f"Crossings at z-indices: {crossings}")
    for i in range(len(crossings)):
        crossing = crossings[i]
        if verbose >= 1: print(f"Processing crossing at z={crossing}:")
        if verbose >= 1: print(f"Reading top region:    voxels[{crossing-max_shift}:{crossing}]")
        top_voxels = jp.array(voxels[crossing-max_shift:crossing]).astype(jp.float32)
        if verbose >= 1: print(f"Reading bottom region: voxels[{crossing}:{crossing+max_shift}]")
        bot_voxels = jp.array(voxels[crossing:crossing+max_shift]).astype(jp.float32)

        if verbose >= 1: print(f"Matching regions (Shapes: {bot_voxels.shape} {top_voxels.shape})")
        shift, error = match_region_jit(top_voxels,bot_voxels,overlap,max_shift)
        shifts[i] = shift
        errors[i] = error
        if verbose >= 1: print(f"Optimal shift is {shift} with error {error} per voxel")

        if(write_image_checks):
            image_dir = f"{volume_matched_dir}/verification"
            pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)
            if verbose >= 1: print(f"Writing images of matched slices to {image_dir} to check correctness.")
            merged_zy_slice = np.concatenate([top_voxels[:,:,Nx//2],bot_voxels[shift:,:,Nx//2]])
#            merged_zy_slice  = np.array(merged_voxels[:,:,Nx//2])

            Image.fromarray(merged_zy_slice.astype(np.uint8)).save(f"{image_dir}/{sample}-{i}cross-zy.png")

            image = np.zeros((2*max_shift,1980),dtype=np.uint8)
            image[:max_shift,:1980//2] = top_voxels[:,Ny//2-1980//4:Ny//2+1980//4,Nx//2].astype(np.uint8)
            shift = max(1, shift)
            image[max_shift-shift:-shift,1980//2:] = bot_voxels[:,Ny//2-1980//4:Ny//2+1980//4,Nx//2].astype(np.uint8)
            Image.fromarray(image).save(f"{image_dir}/{sample}-{i}bottop-zx.png")

        del top_voxels
        del bot_voxels
    return shifts,errors

# Copy through the volume matched volume from the original - general interface
# that works equally well for anything indexed like numpy arrays - including cupy, HDF5 and netCDF.
def write_matched(voxels_in, voxels_out, crossings, shifts):
    voxels_out[:crossings[0]] = voxels_in[:crossings[0]]
    cum_shifts = [0]+list(np.cumsum(shifts))
    crossings  = list(crossings) + [voxels_in.shape[0]]

    if verbose >= 1: print(f"Cumulative shifts: {cum_shifts}")
    if verbose >= 1: print(f"Duplicating subvolume 0: 0:{crossings[0]}")
    voxels_out[:crossings[0]] = voxels_in[:crossings[0]];
    for i in range(len(crossings)-1):
        if verbose >= 1: print(f"Duplicating unmatched part of subvolume {i+1}: voxels_out[{crossings[i]-cum_shifts[i]}:{crossings[i+1]-cum_shifts[i]-shifts[i]}] = voxels_in[{crossings[i]+shifts[i]}:{crossings[i+1]}];")
        voxels_out[crossings[i]-cum_shifts[i]:crossings[i+1]-cum_shifts[i]-shifts[i]] = voxels_in[crossings[i]+shifts[i]:crossings[i+1]];


# Create and populate a volume matched HDF5-file from the original
def write_matched_hdf5(h5_filename_in, h5_filename_out, crossings, shifts, compression='lzf'):
    h5in  = h5py.File(h5_filename_in, "r")
    h5out = h5py.File(h5_filename_out,"w")
    voxels_in = h5in['voxels'];

    (Nz,Ny,Nx) = voxels_in.shape
    matched_Nz = Nz - np.sum(shifts)
    voxels_out = h5out.create_dataset("voxels",(matched_Nz,Ny,Nx), dtype=voxels_in.dtype, compression=compression)
    write_matched(voxels_in, voxels_out, crossings, shifts)

    # TODO: Duplicate all metadata into new HDF5 (Nice to have, but can always be found in original HDF5)



if __name__ == "__main__":
    sample, overlap, max_shift, generate_h5, verbose = commandline_args({"sample" : "<required>",
                                                                         "overlap" : 10,
                                                                         "max_shift" : 150,
                                                                         "generate_h5" : False,
                                                                         "verbose" : 1})

    input_h5name  = f"{hdf5_root}/hdf5-byte/msb/{sample}.h5"
    output_h5name = f"{volume_matched_dir}/1x/{sample}.h5"

    outdir = os.path.dirname(output_h5name)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    h5file = h5py.File(input_h5name, "r+")
    voxels = h5file['voxels']
    subvolume_dimensions = h5file['subvolume_dimensions'][:]
    (Nz,Ny,Nx) = h5file['voxels'].shape

    crossings = np.cumsum(subvolume_dimensions[:-1,0]).astype(int)
    if verbose >= 1: print(f"Matching all regions for sample {sample} at crossings {crossings}.")
    shifts, errors = match_all_regions(voxels,crossings)

    np.save(f"{volume_matched_dir}/{sample}-shifts.npy",shifts)

    if("volume_matching_shifts" not in h5file):
        h5file.create_dataset("volume_matching_shifts",data=shifts)
    else:
        h5file["volume_matching_shifts"][...] = shifts

    h5file.close()

    if(generate_h5):
        if verbose >= 1: print(f"Copying over volume from {input_h5name} shifted by {shifts} to {output_h5name}")
        write_matched_hdf5(input_h5name, output_h5name, crossings, shifts)
