import sys
sys.path.append(sys.path[0]+"/../")
import h5py
import pybind_kernels.histograms as histograms
import pybind_kernels.label as label
import numpy as np
from config.paths import binary_root, hdf5_root_fast as hdf5_root, commandline_args
from tqdm import tqdm
from scipy import ndimage as ndi
import timeit
NA = np.newaxis

def sphere(n):
    xs = np.linspace(-1,1,n)
    return (xs[:,NA,NA]**2 + xs[NA,:,NA]**2 + xs[NA,NA,:]**2) <= 1

if __name__ == '__main__':
    sample, scale = commandline_args({"sample":"<required>","scale":4})

    with h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5") as f:
        implant_mask = f['implant/mask'][:,:,:]
    
    voxels = implant_mask.astype(np.bool)
    result_cpu = np.empty_like(voxels)
    cpu_start = timeit.default_timer()
    histograms.dilate_3d_sphere_cpu(voxels, 5.0, result_cpu)
    cpu_end = timeit.default_timer()
    print (f'cpu did it in {cpu_end - cpu_start} seconds')

    result_gpu = np.empty_like(voxels)
    gpu_start = timeit.default_timer()
    histograms.dilate_3d_sphere_gpu(voxels, 5.0, result_gpu)
    gpu_end = timeit.default_timer()
    print (f'gpu did it in {gpu_end - gpu_start} seconds')

    sph = sphere(10)
    ndi_start = timeit.default_timer()
    ndi_result = ndi.binary_dilation(voxels, sphere(10))
    ndi_end = timeit.default_timer()
    print (f'ndi did it in {ndi_end - ndi_start}')

    if not np.allclose(result_cpu, ndi_result): print ('!!!!! cpu version did not match ndimage !!!!!')
    if not np.allclose(result_gpu, ndi_result): print ('!!!!! gpu version did not match ndimage !!!!!')
    if not np.allclose(result_cpu, result_gpu): print ('!!!!! CPU AND GPU DID NOT MATCH !?!?!? !!!!!')
