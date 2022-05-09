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
from config.paths import hdf5_root_fast, commandline_args
from math import pi, sqrt, exp
from scipy import ndimage as ndi

impl_type = np.float32

def tobyt(arr):
    mi, ma = arr.min(), arr.max()
    return (((arr - mi) / (ma - mi + 1)) * 255).astype(np.uint8)

if __name__ == '__main__':
    sample, scale = commandline_args({"sample":"<required>","scale":1})
    outpath = 'dummy'
    sigma = 13
    reps = 5

    vf = h5py.File(f'{hdf5_root_fast}/processed/implant-edt/{scale}x/{sample}.h5', 'r')
    voxels = vf['voxels'][512:576,:,:]
    vf.close()

    nz,ny,nx = voxels.shape

    Image.fromarray(tobyt(voxels[nz//2,:,:])).save(f"{outpath}/original.png")

    vmax = voxels.max()
    implant_mask = voxels >= vmax

    Image.fromarray(tobyt(implant_mask[nz//2,:,:].astype(impl_type))).save(f"{outpath}/masked.png")

    radius = int(4.0 * float(sigma) + .5) # stolen from the default scipy parameters
    kernel = ndi.filters._gaussian_kernel1d(sigma, 0, radius).astype(impl_type)

    result = np.empty(implant_mask.shape,dtype=impl_type)
    start = timeit.default_timer()
    histograms.gauss_filter_par_cpu(implant_mask, implant_mask.shape, kernel, reps, result)
    print (f'Parallel C edition: {timeit.default_timer() - start}')
    np.save(f'{outpath}/mine', result)

    Image.fromarray(tobyt(result[nz//2,:,:])).save(f'{outpath}/gauss-xy.png')
    Image.fromarray(tobyt(result[:,ny//2,:])).save(f'{outpath}/gauss-xz.png')
    Image.fromarray(tobyt(result[:,:,nx//2])).save(f'{outpath}/gauss-yz.png')
    Image.fromarray(tobyt((np.max(np.abs(result),axis=0)!=0).astype(float))).save(f'{outpath}/gauss-xy-nonzero.png')
    Image.fromarray(tobyt((np.max(np.abs(result),axis=1)!=0).astype(float))).save(f'{outpath}/gauss-xz-nonzero.png')
    Image.fromarray(tobyt((np.max(np.abs(result),axis=2)!=0).astype(float))).save(f'{outpath}/gauss-yz-nonzero.png')

    if True: # generate ndimage comparison
        control = implant_mask.astype(impl_type)
        start = timeit.default_timer()
        for _ in range(reps):
            control[:] = ndi.gaussian_filter(control, sigma, mode='constant', cval=0)
            control[implant_mask] = 1
        print (f'ndimage edition: {timeit.default_timer() - start}')
        np.save(f'{outpath}/control', control)
        Image.fromarray(tobyt(control[nz//2,:,:])).save(f'{outpath}/control1.png')
        diff = np.abs(result - control)
        print (f'Total difference: {diff.sum()}')
        print (f'Max difference: {diff.max()}')
        print (f'Mean difference: {diff.mean()}')