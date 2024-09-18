#! /usr/bin/python3
'''
Diffusion approximation by repeated Gaussian blurs.
This provides an estimate to counter the "glow" of the implant in the tomogram.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')
from config.paths import hdf5_root, binary_root
import h5py
from lib.cpp.gpu.diffusion import diffusion
from lib.cpp.cpu.io import load_slice, write_slice
from lib.py.helpers import commandline_args, gauss_kernel, generate_cylinder_mask, plot_middle_planes, to_int
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import psutil
import timeit
from tqdm import tqdm
from scipy import ndimage as ndi

# sigma is given in physical units, i.e. in micrometers, in order to give scale-invariant results.
if __name__ == '__main__':
    sample, scale, sigma, reps, verify, verbose = commandline_args({
        "sample" : "<required>",
        "scale" : 1,
        "sigma" : 10.0,
        "repititions" : 42,
        "verify_against_ndimage" : False,
        "verbose" : 2
    })
    if verbose >= 1: print(f"Diffusion approximation by repeated Gaussian blurs.\n")

    # The internal type is the type used for the computation, the result type is the type used for the output.
    internal_type = np.float32
    result_type = np.uint16

    output_dir = f"{binary_root}/fields/implant-gauss/{scale}x"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_image_dir = f"{hdf5_root}/processed/field-gauss/{scale}x/{sample}"
    if verbose >= 2:
        pathlib.Path(output_image_dir).mkdir(parents=True, exist_ok=True)

    if verbose >= 1: print(f"Loading implant_solid mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
    with h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5","r") as f:
        implant_solid = f['implant_solid/mask']
        voxel_size = f["implant"].attrs["voxel_size"]
        nz,ny,nx = implant_solid.shape
        implant_mask = implant_solid[:]

    sigma_voxels = sigma / voxel_size

    if verbose >= 1:
        print(f"At scale {scale}x, voxel size is {voxel_size} micrometers.")
        print(f"Using sigma={sigma} micrometers, sigma_voxels={sigma_voxels}.")
        print(f"Implant mask has shape {(nz,ny,nx)}")

    if verbose >= 2:
        plot_middle_planes(implant_mask, output_image_dir, f'{sample}-mask')

    kernel = gauss_kernel(sigma_voxels)

    mem_available = psutil.virtual_memory().available
    n_elements = nz * ny * nx
    mem_input = n_elements * np.dtype(implant_mask.dtype).itemsize
    mem_internal = 2 * n_elements * np.dtype(internal_type).itemsize
    mem_output = n_elements * np.dtype(result_type).itemsize
    mem_total = mem_input + mem_internal + mem_output

    if mem_total > mem_available: # We need to split the computation into chunks across disk
        # Dump the mask
        masks_dir = f"{binary_root}/masks/{scale}x"
        pathlib.Path(masks_dir).mkdir(parents=True, exist_ok=True)
        input_path = f"{masks_dir}/{sample}-implant_solid.{np.dtype(np.uint8).name}"
        output_path = f"{output_dir}/{sample}.{np.dtype(result_type).name}"
        write_slice(implant_mask.astype(np.uint8), input_path, (0,0,0), implant_mask.shape)

        if scale == 8:
            n_layers = nz // 2 + 1
        else:
            gigabyte = 1024**3
            gigabyte_internal = gigabyte / np.dtype(internal_type).itemsize
            n_layers = int(np.floor((1 * gigabyte_internal) / (ny*nx)))
            n_layers = min(n_layers, nz)

        if verbose >= 1:
            print(f"Repeated Gauss blurs ({reps} iterations, sigma_voxels={sigma_voxels}, kernel length={kernel.shape} coefficients)")
            print(f"Reading from {input_path}, writing to {output_path}")
            print(f"Using {n_layers} layers of {ny}x{nx} slices")
            start = timeit.default_timer()

        diffusion(input_path, kernel, output_path, (nz, ny, nx), (n_layers, ny, nx), reps)

        if verbose >= 1:
            diffusion_time = timeit.default_timer() - start
            print (f'C++ edition took {diffusion_time:.02f} seconds')

        result = np.empty((nz,ny,nx), dtype=result_type)
        load_slice(result, output_path, (0,0,0), (nz,ny,nx))
    else:
        result = np.zeros(implant_mask.shape, dtype=result_type)
        if verbose >= 1:
            start = timeit.default_timer()
            print(f"Repeated Gauss blurs ({reps} iterations, sigma_voxels={sigma_voxels}, kernel length={kernel.shape} coefficients)")

        diffusion(implant_mask, kernel, result, reps)

        if verbose >= 1:
            diffusion_time = timeit.default_timer() - start
            print (f'C++ edition took {diffusion_time:.02f} seconds')

    cylinder_mask = generate_cylinder_mask(nx)

    if verbose >= 2:
        plot_middle_planes(result, output_image_dir, f'{sample}-gauss')
        plot_middle_planes(result, output_image_dir, f'{sample}-gauss-nonzero', lambda x: (np.abs(x) != 0).astype(np.uint8))

    if verbose >= 1: print(f"Writing diffusion-field to {output_dir}/{sample}.npy")
    np.save(f'{output_dir}/{sample}.npy', result*cylinder_mask)

    if verify and scale > 1: # generate ndimage comparison, but only for scale > 1
        start = timeit.default_timer()
        control = (implant_mask > 0).astype(internal_type)
        for _ in tqdm(range(reps), desc='ndimage repititions'):
            control[:] = ndi.gaussian_filter(control, sigma_voxels, mode='constant', cval=0)
            control[implant_mask] = 1 # Illuminate
        control = np.floor(control * np.iinfo(result_type).max).astype(result_type)
        ndimage_time = timeit.default_timer() - start
        print (f'ndimage edition took {ndimage_time:.02f} seconds')
        print (f'C++ edition is {ndimage_time/diffusion_time:.02f} times faster')
        np.save(f'{output_dir}/{sample}_ndimage.npy',control)

        if verbose >= 2:
            plot_middle_planes(control, output_image_dir, f'{sample}-control')

        if result_type == np.uint8 or result_type == np.uint16:
            diff = result.astype(np.int32) - control.astype(np.int32)
        else:
            diff = result - control
        diff_abs = np.abs(diff)

        if verbose >= 2:
            plot_middle_planes(diff, output_image_dir, f'{sample}-diff')
            plot_middle_planes(diff_abs, output_image_dir, f'{sample}-diff-abs')

        diff_sum = diff_abs.sum()
        diff_max = diff_abs.max()
        diff_mean = diff_abs.mean()

        print (f'Total difference: {diff_sum}')
        print (f'Max abs difference: {diff_max}')
        print (f'Min and max difference: {diff.min()} {diff.max()}')
        print (f'Mean difference: {diff_mean}')