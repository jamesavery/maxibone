#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0]+"/../")
from matplotlib import image
import pybind_kernels.histograms as histograms
import numpy as np, h5py, timeit
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from config.paths import hdf5_root, commandline_args

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

def field_histogram(voxels, field, field_bins, voxel_bins, ranges):
    bins = np.zeros((field_bins, voxel_bins), dtype=np.uint64)
    vmin, vmax = ranges
    # python3 histograms_tester.py 770c_pag  1849.98s user 170.42s system 512% cpu 6:33.95 total
    histograms.field_histogram_par_cpu(voxels, field, bins, vmin, vmax)
    # python3 histograms_tester.py 770c_pag  1095.49s user 141.76s system 104% cpu 19:44.64 total
    #histograms.field_histogram_seq_cpu(voxels, field, bins, vmin, vmax)

    return bins

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

def tobyt(arr):
    mi, ma = arr.min(), arr.max()
    return (((arr - mi) / (ma - mi + 1)) * 255).astype(np.uint8)

def row_normalize(A):
    return A/(1+np.max(A,axis=1))[:,np.newaxis]

def load_block(sample, offset, block_size, y_cutoff, binary=True):
    '''
    Loads a block of data from disk into memory.
    For binary files, it assumes that y_cutoff has been applied.
    '''
    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    dl = h5py.File(f'{hdf5_root}/hdf5-byte/lsb/{sample}.h5', 'r')
    fi = h5py.File(f'{hdf5_root}/processed/implant-edt/2x/{sample}.h5', 'r')
    Nz, Ny, Nx = dm['voxels'].shape
    Ny -= y_cutoff
    fz, fy, fx = fi['voxels'].shape
    fy -= y_cutoff // 2
    voxels = np.empty((block_size,Ny,Nx), dtype=np.uint16)
    field = np.zeros((block_size//2,fy,fx), dtype=np.uint16)

    if binary:
        histograms.load_slice(voxels, f'{hdf5_root}/binary/{sample}_voxels.uint16', (offset, 0, 0), (Nz, Ny, Nx))
        histograms.load_slice(field, f'{hdf5_root}/binary/{sample}_field.uint16', (offset//2, 0, 0), (fz, fy, fx))
    else:
        zstart = offset
        zstop = min(offset+block_size, Nz)
        fzstart = offset // 2
        fzstop = min((offset//2)+(block_size//2), fz)
        voxels[:zstop-zstart,:,:] = \
                (dm['voxels'][zstart:zstop,y_cutoff:,:].astype(np.uint16) << 8) | \
                (dl['voxels'][zstart:zstop,y_cutoff:,:].astype(np.uint16))
        if fzstop-fzstart > 0: # Guard since field is smaller than voxels
            field[:fzstop-fzstart,:fy,:fx] = fi['voxels'][fzstart:fzstop,y_cutoff//2:,:].astype(np.uint16)

    dm.close()
    dl.close()
    fi.close()

    return voxels, field

# Edition where the histogram is loaded and processed in chunks
# If block_size is less than 0, then the whole thing is loaded and processed.
def run_out_of_core(sample, block_size=128, voxel_bins=4096, y_cutoff=1300, implant_threshold=32000):
    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    dl = h5py.File(f'{hdf5_root}/hdf5-byte/lsb/{sample}.h5', 'r')
    fi = h5py.File(f'{hdf5_root}/processed/implant-edt/2x/{sample}.h5', 'r')

    Nz, Ny, Nx = dm['voxels'].shape
    Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1
    center = ((Ny//2) - y_cutoff, Nx//2)
    Ny -= y_cutoff

    block_size = block_size if block_size > 0 else Nz
    blocks = (Nz // block_size) + (1 if Nz % block_size > 0 else 0)

    vmin, vmax = 0.0, float(implant_threshold)
    fmin, fmax = 4.0, 65535.0 # TODO don't hardcode.

    x_bins = np.zeros((Nx, voxel_bins), dtype=np.uint64)
    y_bins = np.zeros((Ny, voxel_bins), dtype=np.uint64)
    z_bins = np.zeros((Nz, voxel_bins), dtype=np.uint64)
    r_bins = np.zeros((Nr, voxel_bins), dtype=np.uint64)
    f_bins = np.zeros((voxel_bins//2, voxel_bins), dtype=np.uint64)

    fz, fy, fx = fi['voxels'].shape
    fy -= y_cutoff // 2

    dm.close()
    dl.close()
    fi.close()

    for i in tqdm(range(blocks), desc='Computing histograms'):
        zstart = i*block_size
        voxels, field = load_block(sample, zstart, block_size, y_cutoff, True)
        histograms.axis_histogram_par_gpu(voxels, (zstart, 0, 0), block_size, x_bins, y_bins, z_bins, r_bins, center, (vmin, vmax), False)
        histograms.field_histogram_resample_par_cpu(voxels, field, (zstart, 0, 0), (Nz, Ny, Nx), (fz, fy, fx), block_size, f_bins, (vmin, vmax), (fmin, fmax))

    f_bins[-1] = 0 # TODO "bright" mask hack

    return x_bins, y_bins, z_bins, r_bins, f_bins

if __name__ == '__main__':
    sample, y_cutoff = commandline_args({"sample":"770c_pag", "y_cutoff":1300})

    implant_threshold_u16 = 32000 # TODO: to config.constants
    voxel_bins = 4096
    block_size = 256

    outpath = f'{hdf5_root}/processed/histograms/{sample}/'

    xb, yb, zb, rb, fb = run_out_of_core(sample, block_size)

    Image.fromarray(tobyt(row_normalize(xb))).save(f"{outpath}/xb.png")
    Image.fromarray(tobyt(row_normalize(yb))).save(f"{outpath}/yb.png")
    Image.fromarray(tobyt(row_normalize(zb))).save(f"{outpath}/zb.png")
    Image.fromarray(tobyt(row_normalize(rb))).save(f"{outpath}/rb.png")
    Image.fromarray(tobyt(row_normalize(fb))).save(f"{outpath}/fb.png")
    np.savez(f'{outpath}/bins.npz', x_bins=xb, y_bins=yb, z_bins=zb, r_bins=rb, field_bins=fb)
