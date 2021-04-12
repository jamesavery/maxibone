import histograms
import numpy as np
import sys
import h5py
import timeit
from datetime import datetime

# TODO: Currently specialized to uint16_t
def masked_minmax(voxels):
    return histograms.masked_minmax(voxels)

def axes_histogram(voxels, func=histograms.axis_histogram_seq_cpu, ranges=None, voxel_bins=256):
    (Nz,Ny,Nx) = voxels.shape
    Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1

    x_bins = np.zeros((Nx,voxel_bins), dtype=np.uint64)
    y_bins = np.zeros((Ny,voxel_bins), dtype=np.uint64)
    z_bins = np.zeros((Nz,voxel_bins), dtype=np.uint64)
    r_bins = np.zeros((Nr,voxel_bins), dtype=np.uint64)

    if ranges is None:
        vmin, vmax = masked_minmax(voxels)
    else:
        vmin, vmax = ranges
    print ("Entering call", datetime.now())
    func(voxels, x_bins, y_bins, z_bins, r_bins, vmin, vmax, True)
    print ("Exited call", datetime.now())
    return x_bins, y_bins, z_bins, r_bins

def verify_axes_histogram(voxels, ranges=(1,4095), voxel_bins=256):
    tolerance = 1e-5
    schx, schy, schz, schr = axes_histogram(voxels, func=histograms.axis_histogram_seq_cpu, ranges=ranges, voxel_bins=voxel_bins)
    pchx, pchy, pchz, pchr = axes_histogram(voxels, func=histograms.axis_histogram_par_cpu, ranges=ranges, voxel_bins=voxel_bins)

    dx = np.abs(schx - pchx).sum()
    dy = np.abs(schy - pchy).sum()
    dz = np.abs(schz - pchz).sum()
    dr = np.abs(schr - pchr).sum()

    par_cpu_verified = False
    if (dx < tolerance and dy < tolerance and dz < tolerance and dr < tolerance):
        par_cpu_verified = True
    else:
        print (f'diff x = {dx}')
        print (f'diff y = {dy}')
        print (f'diff z = {dz}')
        print (f'diff r = {dr}')

    pghx, pghy, pghz, pghr = axes_histogram(voxels, func=histograms.axis_histogram_par_gpu, ranges=ranges, voxel_bins=voxel_bins)

    dx = np.abs(schx - pghx).sum()
    dy = np.abs(schy - pghy).sum()
    dz = np.abs(schz - pghz).sum()
    dr = np.abs(schr - pghr).sum()

    par_gpu_verified = False
    if (dx < tolerance and dy < tolerance and dz < tolerance and dr < tolerance):
        par_gpu_verified = True
    else:
        print (f'diff x = {dx}')
        print (f'diff y = {dy}')
        print (f'diff z = {dz}')
        print (f'diff r = {dr}')

    verified = par_cpu_verified and par_gpu_verified
    if verified:
        print ('Both parallel CPU and GPU matched sequential CPU version')
    return verified

def benchmark_axes_histograms(voxels, ranges=(1,4095), voxel_bins=256, runs=10):
    print()
    print('----- Benchmarking -----')
    print()
    seq_cpu = timeit.timeit(lambda: axes_histogram(voxels, func=histograms.axis_histogram_seq_cpu, ranges=ranges, voxel_bins=voxel_bins), number=runs)
    par_cpu = timeit.timeit(lambda: axes_histogram(voxels, func=histograms.axis_histogram_par_cpu, ranges=ranges, voxel_bins=voxel_bins), number=runs)
    par_gpu = timeit.timeit(lambda: axes_histogram(voxels, func=histograms.axis_histogram_par_gpu, ranges=ranges, voxel_bins=voxel_bins), number=runs)
    print (f'Average of {runs} runs:')
    print (f'Seq CPU: {seq_cpu / runs:9.04f}')
    print (f'Par CPU: {par_cpu / runs:9.04f}')
    print (f'Par GPU: {par_gpu / runs:9.04f}')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = h5py.File(sys.argv[1], 'r')
    else:
        dataset = h5py.File('../../6x/770c_pag.h5', 'r')

    vxs = dataset['voxels']

    vmin, vmax = masked_minmax(vxs)

    verified = verify_axes_histogram(vxs, ranges=(vmin,vmax), voxel_bins=4096)
    if verified:
        benchmark_axes_histograms(vxs, ranges=(vmin,vmax), voxel_bins=4096, runs=1)
