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
from lib.py.commandline_args import default_parser
from lib.py.helpers import gauss_kernel, generate_cylinder_mask, plot_middle_planes
import numpy as np
import pathlib
import psutil
import timeit
from tqdm import tqdm
from scipy import ndimage as ndi

if __name__ == '__main__':
    argparser = default_parser(__doc__)
    argparser.add_argument('sigma', action='store', type=float, default=10.0, nargs='?',
        help='The sigma of the Gaussian blur in micrometers. Default is 10.0.')
    argparser.add_argument('repetitions', action='store', type=int, default=42, nargs='?',
        help='The number of repetitions of the Gaussian blur. Default is 42.')
    argparser.add_argument('--verify', action='store_true',
        help='Verify the results against scipy.ndimage.')
    args = argparser.parse_args()

    if args.verbose >= 1: print(f"Diffusion approximation by repeated Gaussian blurs.\n")

    # The internal type is the type used for the computation, the result type is the type used for the output.
    internal_type = np.float32
    result_type = np.uint16

    output_dir = f"{binary_root}/fields/implant-gauss/{args.sample_scale}x"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_image_dir = f"{hdf5_root}/processed/field-gauss/{args.sample_scale}x/{args.sample}"
    if args.verbose >= 2:
        pathlib.Path(output_image_dir).mkdir(parents=True, exist_ok=True)

    if args.verbose >= 1: print(f"Loading implant_solid mask from {hdf5_root}/masks/{args.sample_scale}x/{args.sample}.h5")
    with h5py.File(f"{hdf5_root}/masks/{args.sample_scale}x/{args.sample}.h5","r") as f:
        implant_solid = f['implant_solid/mask']
        voxel_size = f["implant"].attrs["voxel_size"]
        nz,ny,nx = implant_solid.shape
        implant_mask = implant_solid[:]

    sigma_voxels = args.sigma / voxel_size

    if args.verbose >= 1:
        print(f"At scale {args.sample_scale}x, voxel size is {voxel_size} micrometers.")
        print(f"Using sigma={args.sigma} micrometers, sigma_voxels={sigma_voxels}.")
        print(f"Implant mask has shape {(nz,ny,nx)}")

    if args.verbose >= 2:
        plot_middle_planes(implant_mask, output_image_dir, f'{args.sample}-mask')

    kernel = gauss_kernel(sigma_voxels)

    mem_available = psutil.virtual_memory().available
    n_elements = nz * ny * nx
    mem_input = n_elements * np.dtype(implant_mask.dtype).itemsize
    mem_internal = 2 * n_elements * np.dtype(internal_type).itemsize
    mem_output = n_elements * np.dtype(result_type).itemsize
    mem_total = mem_input + mem_internal + mem_output

    if mem_total > mem_available: # We need to split the computation into chunks across disk
        # Dump the mask
        masks_dir = f"{binary_root}/masks/{args.sample_scale}x"
        pathlib.Path(masks_dir).mkdir(parents=True, exist_ok=True)
        input_path = f"{masks_dir}/{args.sample}-implant_solid.{np.dtype(np.uint8).name}"
        output_path = f"{output_dir}/{args.sample}.{np.dtype(result_type).name}"
        write_slice(implant_mask.astype(np.uint8), input_path, (0,0,0), implant_mask.shape)

        if args.sample_scale == 8:
            n_layers = nz // 2 + 1
        else:
            gigabyte = 1024**3
            gigabyte_internal = gigabyte / np.dtype(internal_type).itemsize
            n_layers = int(np.floor((1 * gigabyte_internal) / (ny*nx)))
            n_layers = min(n_layers, nz)

        if args.verbose >= 1:
            print(f"Repeated Gauss blurs ({args.repetitions} iterations, sigma_voxels={sigma_voxels}, kernel length={kernel.shape} coefficients)")
            print(f"Reading from {input_path}, writing to {output_path}")
            print(f"Using {n_layers} layers of {ny}x{nx} slices")
            start = timeit.default_timer()

        diffusion(input_path, kernel, output_path, (nz, ny, nx), (n_layers, ny, nx), args.repetitions)

        if args.verbose >= 1:
            diffusion_time = timeit.default_timer() - start
            print (f'C++ edition took {diffusion_time:.02f} seconds')

        result = np.empty((nz,ny,nx), dtype=result_type)
        load_slice(result, output_path, (0,0,0), (nz,ny,nx))
    else:
        result = np.zeros(implant_mask.shape, dtype=result_type)
        if args.verbose >= 1:
            start = timeit.default_timer()
            print(f"Repeated Gauss blurs ({args.repetitions} iterations, sigma_voxels={sigma_voxels}, kernel length={kernel.shape} coefficients)")

        diffusion(implant_mask, kernel, result, args.repetitions)

        if args.verbose >= 1:
            diffusion_time = timeit.default_timer() - start
            print (f'C++ edition took {diffusion_time:.02f} seconds')

    cylinder_mask = generate_cylinder_mask(nx)

    if args.verbose >= 2:
        plot_middle_planes(result, output_image_dir, f'{args.sample}-gauss')
        plot_middle_planes(result, output_image_dir, f'{args.sample}-gauss-nonzero', lambda x: (np.abs(x) != 0).astype(np.uint8))

    if args.verbose >= 1: print(f"Writing diffusion-field to {output_dir}/{args.sample}.npy")
    np.save(f'{output_dir}/{args.sample}.npy', result*cylinder_mask)

    if args.verify and args.sample_scale > 1: # generate ndimage comparison, but only for scale > 1
        start = timeit.default_timer()
        control = (implant_mask > 0).astype(internal_type)
        for _ in tqdm(range(args.repetitions), desc='ndimage repetitions'):
            control[:] = ndi.gaussian_filter(control, sigma_voxels, mode='constant', cval=0)
            control[implant_mask] = 1 # Illuminate
        control = np.floor(control * np.iinfo(result_type).max).astype(result_type)
        ndimage_time = timeit.default_timer() - start
        print (f'ndimage edition took {ndimage_time:.02f} seconds')
        print (f'C++ edition is {ndimage_time/diffusion_time:.02f} times faster')
        np.save(f'{output_dir}/{args.sample}_ndimage.npy',control)

        if args.verbose >= 2:
            plot_middle_planes(control, output_image_dir, f'{args.sample}-control')

        if result_type == np.uint8 or result_type == np.uint16:
            diff = result.astype(np.int32) - control.astype(np.int32)
        else:
            diff = result - control
        diff_abs = np.abs(diff)

        if args.verbose >= 2:
            plot_middle_planes(diff, output_image_dir, f'{args.sample}-diff')
            plot_middle_planes(diff_abs, output_image_dir, f'{args.sample}-diff-abs')

        diff_sum = diff_abs.sum()
        diff_max = diff_abs.max()
        diff_mean = diff_abs.mean()

        print (f'Total difference: {diff_sum}')
        print (f'Max abs difference: {diff_max}')
        print (f'Min and max difference: {diff.min()} {diff.max()}')
        print (f'Mean difference: {diff_mean}')