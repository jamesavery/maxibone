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

# From stack overflow
def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    g = [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    return np.array(g, dtype=np.float32)

# A copy of the scipy implementation
def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x

def tobyt(arr):
    mi, ma = arr.min(), arr.max()
    return (((arr - mi) / (ma - mi + 1)) * 255).astype(np.uint8)

if __name__ == '__main__':
    sample, scale = commandline_args({"sample":"<required>","scale":1})
    outpath = 'dummy'
    display = 10
    sigma = 13
    reps =20

    vf = h5py.File(f'{hdf5_root_fast}/processed/implant-edt/{scale}x/{sample}.h5', 'r')
    voxels = vf['voxels'][512:576,:,:]
    vf.close()

    Image.fromarray(tobyt(voxels[display,:,:])).save(f"{outpath}/original.png")

    vmax = voxels.max(); 
    implant_mask = voxels >= vmax
    implant_mask = implant_mask.astype(np.float32)

    Image.fromarray(tobyt(implant_mask[display,:,:])).save(f"{outpath}/masked.png")

    radius = int(4.0 * float(sigma) + .5) # stolen from the default scipy parameters
    kernel = _gaussian_kernel1d(sigma, 0, radius)[::-1]
    #kernel = gauss(radius*2+1, sigma)

    result = np.zeros_like(implant_mask)
    start = timeit.default_timer()
    histograms.gauss_filter_par_cpu(implant_mask, implant_mask.shape, kernel, reps, result)
    print (f'Parallel C edition: {timeit.default_timer() - start}')
    np.save(f'{outpath}/mine', result)

    nz,ny,nx = result.shape    
    Image.fromarray(tobyt(result[nz//2,:,:])).save(f'{outpath}/gauss-xy.png')
    Image.fromarray(tobyt(result[:,ny//2,:])).save(f'{outpath}/gauss-xz.png')
    Image.fromarray(tobyt(result[:,:,nx//2])).save(f'{outpath}/gauss-yz.png')
    Image.fromarray(tobyt((np.max(np.abs(result),axis=0)!=0).astype(float))).save(f'{outpath}/gauss-xy-nonzero.png')

    control = implant_mask.copy()
    start = timeit.default_timer()
    #for _ in range(reps):
    #    control = ndi.gaussian_filter(control, sigma, mode='constant')
    print (f'ndimage edition: {timeit.default_timer() - start}')
    #np.save(f'{outpath}/control', control)
    Image.fromarray(tobyt(control[display,:,:])).save(f'{outpath}/control1.png')

    diff = np.abs(result - control)
    print (f'Total difference: {diff.sum()}')
    print (f'Max difference: {diff.max()}')
    print (f'Mean difference: {diff.mean()}')