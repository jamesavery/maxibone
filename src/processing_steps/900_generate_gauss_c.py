#!/usr/bin/env python3
import sys, pathlib, datetime, numpy as np, h5py, timeit, matplotlib.pyplot as plt, edt
sys.path.append(sys.path[0]+"/../")
from matplotlib import image
from PIL import Image
from tqdm import tqdm
from math import pi, sqrt, exp
from scipy import ndimage as ndi

from config.paths import hdf5_root, binary_root
from lib.py.helpers import commandline_args
from lib.cpp.cpu_seq import gauss_filter
NA = np.newaxis

impl_type = np.float32

def toint(arr, dtype=np.uint8):
    vmin, vmax = arr.min(), arr.max()
    int_max = np.iinfo(dtype).max
    return np.round((((arr - vmin) / (vmax - vmin + (vmin==vmax))) * (int_max-1))).astype(dtype) + 1

# sigma is given in physical units, i.e. in micrometers, in order to give scale-invariant results.
if __name__ == '__main__':
    sample, sigma, reps, scale, voxel_size_1x, verify, debug = commandline_args({"sample":"<required>","sigma":40.0,"repititions":10,"scale":2,
                                                                                 "voxel_size_1x":1.85, "verify_against_ndimage":False, "debug_images":True})
    print(f"Diffusion approximation by repeated Gaussian blurs.\n")
    voxel_size   = voxel_size_1x*scale
    sigma_voxels = sigma/voxel_size
    print(f"At scale {scale}x, voxel size is {voxel_size} micrometers.")
    print(f"Using sigma={sigma} micrometers, sigma_voxels={sigma_voxels}.")

    output_dir = f"{binary_root}/fields/implant-gauss/{scale}x"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading implant_solid mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
    with h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5","r") as f:
        implant_mask = f['implant_solid/mask'][:]

    nz,ny,nx = implant_mask.shape
    print(f"Implant mask has shape {implant_mask.shape}")

    if debug:
        print(f"Writing PNGs of implant mask slices to {output_dir}")
        Image.fromarray(toint(implant_mask[:,:,nx//2].astype(impl_type))).save(f"{output_dir}/{sample}-mask-yz.png")
        Image.fromarray(toint(implant_mask[:,ny//2,:].astype(impl_type))).save(f"{output_dir}/{sample}-mask-xz.png")
        Image.fromarray(toint(implant_mask[nz//2,:,:].astype(impl_type))).save(f"{output_dir}/{sample}-mask-xy.png")

    radius = int(4.0 * sigma_voxels + .5) # stolen from the default scipy parameters
    kernel = ndi.filters._gaussian_kernel1d(sigma_voxels, 0, radius).astype(impl_type)

    result = np.zeros(implant_mask.shape,dtype=impl_type)
    if verify:
        start = timeit.default_timer()

    print(f"Repeated Gauss blurs ({reps} iterations, sigma_voxels={sigma_voxels}, kernel length={radius} coefficients)")
    gauss_filter(implant_mask, implant_mask.shape, kernel, reps, result)
    if verify:
        print (f'Parallel C edition took {timeit.default_timer() - start} seconds')

    xs = np.linspace(-1,1,nx)
    rs = np.sqrt(xs[NA,NA,:]**2 + xs[NA,:,NA]**2)
    cylinder_mask = (rs<=1)

    print(f"Writing diffusion-field to {output_dir}/{sample}.npy")
    np.save(f'{output_dir}/{sample}.npy', toint(result*cylinder_mask,np.uint16)*cylinder_mask)

    
    if debug:
        print(f"Debug: Writing PNGs of result slices to {output_dir}")
        Image.fromarray(toint(result[nz//2,:,:])).save(f'{output_dir}/{sample}-gauss-xy.png')
        Image.fromarray(toint(result[:,ny//2,:])).save(f'{output_dir}/{sample}-gauss-xz.png')
        Image.fromarray(toint(result[:,:,nx//2])).save(f'{output_dir}/{sample}-gauss-yz.png')
        Image.fromarray(toint((np.max(np.abs(result),axis=0)!=0).astype(float))).save(f'{output_dir}/{sample}-gauss-xy-nonzero.png')
        Image.fromarray(toint((np.max(np.abs(result),axis=1)!=0).astype(float))).save(f'{output_dir}/{sample}-gauss-xz-nonzero.png')
        Image.fromarray(toint((np.max(np.abs(result),axis=2)!=0).astype(float))).save(f'{output_dir}/{sample}-gauss-yz-nonzero.png')
        
    if verify: # generate ndimage comparison
        control = implant_mask.astype(impl_type)
        start = timeit.default_timer()
        for _ in tqdm(range(reps), desc='ndimage repititions'):
            control[:] = ndi.gaussian_filter(control, sigma, mode='constant', cval=0)
            control[implant_mask] = 1
        print (f'ndimage edition took {timeit.default_timer() - start} seconds')
        np.save(f'{output_dir}/{sample}_ndimage.npy',control)
        if debug:
            Image.fromarray(toint(control[nz//2,:,:])).save(f'{output_dir}/{sample}-control-xy.png')
            Image.fromarray(toint(control[:,ny//2,:])).save(f'{output_dir}/{sample}-control-xz.png')
            Image.fromarray(toint(control[:,:,nx//2])).save(f'{output_dir}/{sample}-control-yz.png')
        diff = np.abs(result - control)
        diff_sum = diff.sum()
        diff_max = diff.max()
        diff_mean = diff.mean()
        print (f'Total difference: {diff.sum()}')
        print (f'Max difference: {diff.max()}')
        print (f'Mean difference: {diff.mean()}')
        if diff_max > 1e-10:
            for name, res, ctrl in [
                    ('xy', result[nz//2,:,:], control[nz//2,:,:]),
                    ('xz', result[:,ny//2,:], control[:,ny//2,:]),
                    ('yz', result[:,:,nx//2], control[:,:,nx//2])]:
                plt.figure(figsize=(20,20))
                plt.imshow(np.abs(res - ctrl))
                plt.savefig(f'{output_dir}/{sample}-diff-{name}.png')


    print(f"Computing Euclidean distance transform.")
    fedt = edt.edt(~implant_mask,parallel=16)
    del implant_mask
    
    edt_output_dir = f"{binary_root}/fields/implant-edt/{scale}x"
    pathlib.Path(edt_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Writing EDT-field to {edt_output_dir}/{sample}.npy")
    np.save(f'{edt_output_dir}/{sample}.npy', toint(fedt*cylinder_mask,np.uint16)*cylinder_mask)
                

    mixed_output_dir = f"{binary_root}/fields/implant-gauss+edt/{scale}x"    
    print(f"Writing combined field to {mixed_output_dir}/{sample}.npy")                
    pathlib.Path(mixed_output_dir).mkdir(parents=True, exist_ok=True)
    result = (result-fedt/(fedt.max()))*cylinder_mask
    result -= result.min()
    result /= result.max()
    print(f"Result (min,max) = ({result.min(),result.max()})")
    np.save(f'{mixed_output_dir}/{sample}.npy', toint(result*cylinder_mask,np.uint16)*cylinder_mask)    
