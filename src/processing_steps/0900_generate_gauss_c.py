#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import sys, pathlib, datetime, numpy as np, h5py, timeit, matplotlib.pyplot as plt, edt
sys.path.append(sys.path[0]+"/../")
from matplotlib import image
from PIL import Image
from tqdm import tqdm
from math import pi, sqrt, exp
from scipy import ndimage as ndi
import multiprocessing as mp
from config.paths import hdf5_root, binary_root
from lib.py.helpers import commandline_args, to_int
from lib.cpp.gpu.diffusion import diffusion
from lib.cpp.cpu.io import load_slice, write_slice
NA = np.newaxis
import psutil

internal_type = np.float32
result_type = np.uint16

def gauss_kernel(sigma):
    radius = round(4.0 * sigma) # stolen from the default scipy parameters
    # Deprecated:
    #kernel = ndi.filters._gaussian_kernel1d(sigma_voxels, 0, radius).astype(internal_type)

    if False:
        # Create a 1D Gaussian
        x = np.arange(-radius, radius + 1)
        kernel = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))
        return kernel
    else:
        # Stolen from ndimage
        sigma2 = sigma * sigma
        x = np.arange(-radius, radius+1)
        phi_x = np.exp(-0.5 / sigma2 * x ** 2)
        phi_x = phi_x / phi_x.sum()
        return phi_x

# sigma is given in physical units, i.e. in micrometers, in order to give scale-invariant results.
if __name__ == '__main__':
    sample, scale, sigma, reps, verify, verbose = \
        commandline_args({
            "sample" : "<required>",
            "scale" : 1,
            "sigma" : 10.0,
            "repititions" : 42,
            "verify_against_ndimage" : False,
            "verbose" : 2
        })
    if verbose >= 1: print(f"Diffusion approximation by repeated Gaussian blurs.\n")

    output_dir = f"{binary_root}/fields/implant-gauss/{scale}x"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if verbose >= 1: print(f"Loading implant_solid mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
    with h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5","r") as f:
        implant_solid = f['implant_solid/mask']
        voxel_size   = f["implant"].attrs["voxel_size"]
        nz,ny,nx = implant_solid.shape
        implant_mask = implant_solid[:]

    sigma_voxels = sigma / voxel_size
    if verbose >= 1: print(f"At scale {scale}x, voxel size is {voxel_size} micrometers.")
    if verbose >= 1: print(f"Using sigma={sigma} micrometers, sigma_voxels={sigma_voxels}.")

    if verbose >= 1: print(f"Implant mask has shape {(nz,ny,nx)}")

    if verbose >= 2:
        print(f"Writing PNGs of implant mask slices to {output_dir}")
        Image.fromarray(to_int(implant_mask[:,:,nx//2].astype(internal_type),np.uint8)).save(f"{output_dir}/{sample}-mask-yz.png")
        Image.fromarray(to_int(implant_mask[:,ny//2,:].astype(internal_type),np.uint8)).save(f"{output_dir}/{sample}-mask-xz.png")
        Image.fromarray(to_int(implant_mask[nz//2,:,:].astype(internal_type),np.uint8)).save(f"{output_dir}/{sample}-mask-xy.png")

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
        result = np.zeros(implant_mask.shape,dtype=result_type)
        if verbose >= 1:
            start = timeit.default_timer()

        if verbose >= 1: print(f"Repeated Gauss blurs ({reps} iterations, sigma_voxels={sigma_voxels}, kernel length={kernel.shape} coefficients)")
        diffusion(implant_mask, kernel, result, reps)
        if verbose >= 1:
            diffusion_time = timeit.default_timer() - start
            print (f'C++ edition took {diffusion_time:.02f} seconds')

    xs = np.linspace(-1,1,nx)
    rs = np.sqrt(xs[NA,NA,:]**2 + xs[NA,:,NA]**2)
    cylinder_mask = (rs<=1)

    if verbose >= 2:
        print(f"Debug: Writing PNGs of result slices to {output_dir}")
        Image.fromarray(to_int(result[nz//2,:,:],np.uint8)).save(f'{output_dir}/{sample}-gauss-xy.png')
        Image.fromarray(to_int(result[:,ny//2,:],np.uint8)).save(f'{output_dir}/{sample}-gauss-xz.png')
        Image.fromarray(to_int(result[:,:,nx//2],np.uint8)).save(f'{output_dir}/{sample}-gauss-yz.png')
        Image.fromarray(to_int((np.abs(result[nz//2,:,:])!=0).astype(np.uint8),np.uint8)).save(f'{output_dir}/{sample}-gauss-xy-nonzero.png')
        Image.fromarray(to_int((np.abs(result[:,ny//2,:])!=0).astype(np.uint8),np.uint8)).save(f'{output_dir}/{sample}-gauss-xz-nonzero.png')
        Image.fromarray(to_int((np.abs(result[:,:,nx//2])!=0).astype(np.uint8),np.uint8)).save(f'{output_dir}/{sample}-gauss-yz-nonzero.png')

    if verbose >= 1: print(f"Writing diffusion-field to {output_dir}/{sample}.npy")
    np.save(f'{output_dir}/{sample}.npy', result*cylinder_mask)

    if verify and scale > 1: # generate ndimage comparison, but only for scale > 1
        start = timeit.default_timer()
        control = (implant_mask>0).astype(internal_type)
        for _ in tqdm(range(reps), desc='ndimage repititions'):
            control[:] = ndi.gaussian_filter(control, sigma_voxels, mode='constant', cval=0)
            control[implant_mask] = 1 # Illuminate
        control = np.floor(control * np.iinfo(result_type).max).astype(result_type)
        ndimage_time = timeit.default_timer() - start
        print (f'ndimage edition took {ndimage_time:.02f} seconds')
        print (f'C++ edition is {ndimage_time/diffusion_time:.02f} times faster')
        np.save(f'{output_dir}/{sample}_ndimage.npy',control)
        if verbose >= 2:
            Image.fromarray(to_int(control[nz//2,:,:],np.uint8)).save(f'{output_dir}/{sample}-control-xy.png')
            Image.fromarray(to_int(control[:,ny//2,:],np.uint8)).save(f'{output_dir}/{sample}-control-xz.png')
            Image.fromarray(to_int(control[:,:,nx//2],np.uint8)).save(f'{output_dir}/{sample}-control-yz.png')
        if result_type == np.uint8 or result_type == np.uint16:
            diff = result.astype(np.int32) - control.astype(np.int32)
        else:
            diff = result - control
        diff_abs = np.abs(diff)
        diff_sum = diff_abs.sum()
        diff_max = diff_abs.max()
        diff_mean = diff_abs.mean()
        print (f'Total difference: {diff_sum}')
        print (f'Max abs difference: {diff_max}')
        print (f'Min and max difference: {diff.min()} {diff.max()}')
        print (f'Mean difference: {diff_mean}')
        if diff_max > 1e-7:
            for name, diff_img in [
                    ('xy', diff[nz//2,:,:]),
                    ('xz', diff[:,ny//2,:]),
                    ('yz', diff[:,:,nx//2])]:
                plt.figure(figsize=(20,20))
                plt.imshow(diff_img)
                plt.colorbar()
                plt.savefig(f'{output_dir}/{sample}-diff-{name}.png', bbox_inches='tight')