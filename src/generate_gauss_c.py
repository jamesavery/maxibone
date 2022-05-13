import pybind_kernels.histograms as histograms
#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0]+"/../")
from matplotlib import image
import pybind_kernels.histograms as histograms
import numpy as np, h5py, timeit
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from config.paths import hdf5_root_fast as hdf5_root, commandline_args
from math import pi, sqrt, exp
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

impl_type = np.float32

def tobyt(arr):
    mi, ma = arr.min(), arr.max()
    return (((arr - mi) / (ma - mi + 1)) * 255).astype(np.uint8)

if __name__ == '__main__':
    sample, sigma, reps, scale, output, verify, debug = commandline_args({"sample":"<required>","sigma":"<required>","repititions":"<required>","scale":1,"output":"processed/implant-gauss","verify_against_ndimage":False, "debug_images":False})
    sigma = int(sigma)
    reps = int(reps)
    outpath = f'{hdf5_root}/{output}/{scale}x/'

    with h5py.File(f'{hdf5_root}/processed/implant/{scale}x/{sample}.h5', 'r') as f:
        voxels = f['voxels'][512:576,:,:]
        implant_mask = voxels != 0

    nz,ny,nx = implant_mask.shape

    if debug:
        Image.fromarray(tobyt(implant_mask[nz//2,:,:].astype(impl_type))).save(f"{outpath}/original.png")

    radius = int(4.0 * float(sigma) + .5) # stolen from the default scipy parameters
    kernel = ndi.filters._gaussian_kernel1d(sigma, 0, radius).astype(impl_type)

    result = np.empty(implant_mask.shape,dtype=impl_type)
    if verify:
        start = timeit.default_timer()
    histograms.gauss_filter_par_cpu(implant_mask, implant_mask.shape, kernel, reps, result)
    if verify:
        print (f'Parallel C edition took {timeit.default_timer() - start} seconds')

    with h5py.File(f'{outpath}/{sample}.h5', 'w') as f:
        f['voxels'] = result

    if debug:
        Image.fromarray(tobyt(result[nz//2,:,:])).save(f'{outpath}/gauss-xy.png')
        Image.fromarray(tobyt(result[:,ny//2,:])).save(f'{outpath}/gauss-xz.png')
        Image.fromarray(tobyt(result[:,:,nx//2])).save(f'{outpath}/gauss-yz.png')
        Image.fromarray(tobyt((np.max(np.abs(result),axis=0)!=0).astype(float))).save(f'{outpath}/gauss-xy-nonzero.png')
        Image.fromarray(tobyt((np.max(np.abs(result),axis=1)!=0).astype(float))).save(f'{outpath}/gauss-xz-nonzero.png')
        Image.fromarray(tobyt((np.max(np.abs(result),axis=2)!=0).astype(float))).save(f'{outpath}/gauss-yz-nonzero.png')

    if verify: # generate ndimage comparison
        control = implant_mask.astype(impl_type)
        start = timeit.default_timer()
        for _ in tqdm(range(reps), desc='ndimage repititions'):
            control[:] = ndi.gaussian_filter(control, sigma, mode='constant', cval=0)
            control[implant_mask] = 1
        print (f'ndimage edition took {timeit.default_timer() - start} seconds')
        with h5py.File(f'{outpath}/{sample}_ndimage.h5', 'w') as f:
            f['voxels'] = control
        if debug:
            Image.fromarray(tobyt(control[nz//2,:,:])).save(f'{outpath}/control-xy.png')
            Image.fromarray(tobyt(control[:,ny//2,:])).save(f'{outpath}/control-xz.png')
            Image.fromarray(tobyt(control[:,:,nx//2])).save(f'{outpath}/control-yz.png')
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
                plt.savefig(f'{outpath}/diff-{name}.png')