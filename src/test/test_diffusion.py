'''
Unit test for diffusion.

Currently only tests in-memory diffusion, not the file-based one.
'''
import sys
sys.path.append(sys.path[0]+'/../')
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as ndi

from lib.cpp.cpu.diffusion import diffusion as diffusion_cpu
from lib.cpp.gpu.diffusion import diffusion as diffusion_gpu

n = 128
sigma = 3 # Radius has to be <= 16 for the faster GPU implementation
reps = 100
plot = False
plotting_dir = sys.path[0] + '/debug_plots/'
if plot: os.makedirs(plotting_dir, exist_ok=True)

run_py = True
run_cpp = True
run_gpu = True

# Generate a random 3d array and binarize it
np.random.seed(42)
a = np.random.rand(n, n, n)
a = a > 0.99

# Get a gaussian kernel
# Stolen from ndimage
radius = round(4.0 * sigma) # stolen from the default scipy parameters
sigma2 = sigma * sigma
x = np.arange(-radius, radius+1)
phi_x = np.exp(-0.5 / sigma2 * x ** 2)
phi_x = phi_x / phi_x.sum()
kernel = phi_x.astype(np.float32)
print (f'Kernel shape: {kernel.shape}, radius: {radius}')

#
# Python implementation
#
if run_py:
    python_start = datetime.datetime.now()
    buf0 = a.copy().astype(np.float32)
    buf1 = np.zeros_like(buf0)
    for _ in range(reps):
        buf1[:] = ndi.convolve1d(buf0, kernel, axis=0, mode='constant')
        buf0[:] = ndi.convolve1d(buf1, kernel, axis=1, mode='constant')
        buf1[:] = ndi.convolve1d(buf0, kernel, axis=2, mode='constant')
        buf1[a] = 1
        buf0, buf1 = buf1, buf0
    python_impl = np.floor(buf0 * 65535).astype(np.uint16)
    del buf1
    python_end = datetime.datetime.now()
    print (f"Python implementation took {python_end - python_start}")

    if plot:
        plt.imshow(python_impl[n//2], cmap='gray')
        plt.savefig(f'{plotting_dir}/python_impl.png')
        plt.close()
        plt.imshow(a[n//2], cmap='gray')
        plt.savefig(f'{plotting_dir}/original.png')
        plt.close()

#
# Parallel CPU C++ implementation
#
if run_cpp:
    cpp_start = datetime.datetime.now()
    cpp_impl = np.empty(a.shape, np.uint16)
    diffusion_cpu(a.astype(np.uint8), kernel, cpp_impl, reps)
    cpp_end = datetime.datetime.now()
    if verify:
        print (f"Parallel CPU C++ implementation took {cpp_end - cpp_start} ({(python_end - python_start) / (cpp_end - cpp_start):.2f}x)")
    else:
        print (f"Parallel CPU C++ implementation took {cpp_end - cpp_start}")

    if plot:
        plt.imshow(cpp_impl[n//2], cmap='gray')
        plt.savefig(f'{plotting_dir}/cpp_impl.png')
        plt.close()

    if verify:
        # Check if the results are the same
        diff = np.abs(python_impl.astype(np.int32) - cpp_impl.astype(np.int32))
        divergers = np.sum(diff > 0)
        divergers2 = np.sum(diff > 1)
        if divergers2 > 0:
            print (f"Found {divergers} diverging pixels out of ({n**3}) ({divergers / n**3 * 100:.2f}%)")
            print (f"Found {divergers2} pixels with a difference greater than 1 ({divergers2 / n**3 * 100:.2f}%)")
        assert np.all(diff <= 1)

#
# Parallel GPU C++ implementation
#
if run_gpu:
    gpu_start = datetime.datetime.now()
    gpu_impl = np.empty(a.shape, np.uint16)
    diffusion_gpu(a.astype(np.uint8), kernel, gpu_impl, reps)
    gpu_end = datetime.datetime.now()
    if verify:
        print (f"Parallel GPU C++ implementation took {gpu_end - gpu_start} ({(python_end - python_start) / (gpu_end - gpu_start):.2f}x)")
    else:
        print (f"Parallel GPU C++ implementation took {gpu_end - gpu_start}")

    if plot:
        plt.imshow(gpu_impl[n//2], cmap='gray')
        plt.savefig(f'{plotting_dir}/gpu_impl.png')
        plt.close()

    if verify:
        # Check if the results are the same
        diff = np.abs(python_impl.astype(np.int32) - gpu_impl.astype(np.int32))
        divergers = np.sum(diff > 0)
        divergers2 = np.sum(diff > 1)
        if divergers2 > 0:
            print (f"Found {divergers} diverging pixels out of ({n**3}) ({divergers / n**3 * 100:.2f}%)")
            print (f"Found {divergers2} pixels with a difference greater than 1 ({divergers2 / n**3 * 100:.2f}%)")
            if plot:
                plt.imshow(diff[n//2])
                plt.colorbar()
                plt.savefig(f'{plotting_dir}/diff.png')
                plt.close()
            checksum = np.sum(gpu_impl)
            print (f"Checksum of GPU: {checksum != 0} ({checksum})")
        assert np.all(diff <= 1)
