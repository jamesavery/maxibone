#!/usr/bin/env python3
import os, sys, pathlib, copy
sys.path.append(sys.path[0]+"/../")
import pybind_kernels.histograms as histograms
import numpy as np, h5py, timeit
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from config.paths import *
from config.constants import implant_threshold

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

#TODO: Get load_slice to work with ranges so we can look at sub-regions (right now y_cutoff doesn't with binary)
def load_block(sample, offset, block_size, y_cutoff, field_names):
    '''
    Loads a block of data from disk into memory.
    For binary files, it assumes that y_cutoff has been applied.
    '''
    Nfields = len(field_names)
    pbar = tqdm(["Load HDF5 metadata",f"Load {sample}.npz", "Load voxels",f"Load {Nfields} fields","Done"],desc="load_block", leave=False)
    pbar.update(); pbar.refresh();

    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    Nz, Ny, Nx = dm['voxels'].shape
    Nz -= np.sum(dm["volume_matching_shifts"][:])
    Ny -= y_cutoff              # TODO: Specify block range to extract instead

    pbar.update(); pbar.refresh();
    block_size       = min(block_size, Nz-offset)
    
    voxels = np.zeros((block_size,Ny,Nx),    dtype=np.uint16)
    fields = np.zeros((Nfields,block_size//2,(Ny-y_cutoff)//2,Nx//2), dtype=np.uint16)    

    pbar.update(); pbar.refresh();
    #TODO: Make voxel & field scale command line parameters
    histograms.load_slice(voxels, f'{binary_root}/voxels/1x/{sample}.uint16', (offset, 0, 0), (Nz, Ny, Nx)) # TODO: Don't use 3 different methods for load/store
    pbar.update(); pbar.refresh();

    for i in tqdm(range(Nfields),f"Loading {binary_root}/fields/implant-{field_names}/2x/{sample}.npy",leave=False):
        fi = np.load(f"{binary_root}/fields/implant-{field_names[i]}/2x/{sample}.npy", mmap_mode='r')
        fields[i,:] = fi[offset//2:offset//2 + block_size//2,:(Ny-y_cutoff)//2,:Nx//2]

    pbar.update(); pbar.refresh();
    pbar.close()
    dm.close()

    return voxels, fields

# Edition where the histogram is loaded and processed in chunks
# If block_size is less than 0, then the whole thing is loaded and processed.
def run_out_of_core(sample, block_size=128, z_offset=0, n_blocks=0, voxel_bins=4096, y_cutoff=0, implant_threshold=32000, field_names=["gauss","edt","gauss+edt"]):
    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')

    vm_shifts  = dm["volume_matching_shifts"][:]
    Nz, Ny, Nx = dm['voxels'].shape
    Nz -= np.sum(vm_shifts)
    Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1
    center = ((Ny//2) - y_cutoff, Nx//2)
    Ny -= y_cutoff

    if block_size == 0:
        # If block_size is 0, let each block be exactly a full subvolume
        subvolume_dimensions = dm['subvolume_dimensions'][:]
        subvolume_nzs        = subvolume_dimensions[:,0] - np.append(vm_shifts,0)
        subvolume_starts     = np.cumsum(subvolume_nsz)
        # Do either n_blocks subvolumes, or if n_blocks == 0: all remaining afster offset.
        if n_blocks == 0:
            n_blocks = len(subvolume_nzs)-z_offset
    else:
        if n_blocks == 0:
            n_blocks = Nz // block_size + (Nz % block_size > 0)

    vmin, vmax = 0.0, float(implant_threshold)
    fmin, fmax = 1.0, 65535.0 # TODO don't hardcode.

    Nfields = len(field_names)
    x_bins = np.zeros((Nx, voxel_bins), dtype=np.uint64)
    y_bins = np.zeros((Ny, voxel_bins), dtype=np.uint64)
    z_bins = np.zeros((Nz, voxel_bins), dtype=np.uint64)
    r_bins = np.zeros((Nr, voxel_bins), dtype=np.uint64)
    f_bins = np.zeros((Nfields,voxel_bins//2, voxel_bins), dtype=np.uint64)

    dm.close()


    for b in tqdm(range(n_blocks), desc='Computing histograms'):
        if block_size == 0:
            zstart  = subvolume_starts[z_offset+b]
            block_size = subvolume_nzs[z_offset+b]
        else:
            zstart = z_offset + b*block_size
            
        voxels, fields = load_block(sample, zstart, block_size, y_cutoff, field_names)
        for i in tqdm(range(1),"Histogramming over x,y,z axes and radius", leave=False):
            histograms.axis_histogram_par_gpu(voxels, (zstart, 0, 0), voxels.shape[0], x_bins, y_bins, z_bins, r_bins, center, (vmin, vmax), False)
        for i in tqdm(range(Nfields),f"Histogramming w.r.t. fields {field_names}", leave=False):
            histograms.field_histogram_resample_par_cpu(voxels, fields[i], (zstart, 0, 0), (Nz, Ny, Nx), (Nz//2,Ny//2,Nx//2), voxels.shape[0], f_bins[i], (vmin, vmax), (fmin, fmax))

    f_bins[-1] = 0 # TODO "bright" mask hack

    return x_bins, y_bins, z_bins, r_bins, f_bins

if __name__ == '__main__':
    # Special parameter values:
    # - block_size == 0 means "do one full subvolume at the time, interpret z_offset as start-at-subvolume-number"
    # - n_blocks   == 0 means "all blocks"
    sample, y_cutoff, block_size, z_offset, n_blocks, suffix, voxel_bins = commandline_args({"sample":"<required>", "y_cutoff":0, "block_size":256, "z_offset": 0, "n_blocks":0, "suffix":"", "voxel_bins":4096})

    implant_threshold_u16 = 32000 # TODO: use config.constants
    field_names=["gauss","edt","gauss+edt"] # Should this be commandline defined?
    
    outpath = f'{hdf5_root}/processed/histograms/{sample}/'
    
    xb, yb, zb, rb, fb = run_out_of_core(sample, block_size, z_offset, n_blocks, voxel_bins, y_cutoff, implant_threshold_u16, field_names)

    Image.fromarray(tobyt(row_normalize(xb))).save(f"{outpath}/xb{suffix}.png")
    Image.fromarray(tobyt(row_normalize(yb))).save(f"{outpath}/yb{suffix}.png")
    Image.fromarray(tobyt(row_normalize(zb))).save(f"{outpath}/zb{suffix}.png")
    Image.fromarray(tobyt(row_normalize(rb))).save(f"{outpath}/rb{suffix}.png")

    for i in range(len(field_names)):
        Image.fromarray(tobyt(row_normalize(fb[i]))).save(f"{outpath}/fb-{field_names[i]}{suffix}.png")
        
    np.savez(f'{outpath}/bins{suffix}.npz', x_bins=xb, y_bins=yb, z_bins=zb, r_bins=rb, field_bins=fb,
             field_names=field_names, suffix=suffix, y_cutoff=y_cutoff,
             sample=sample, z_offset=z_offset, block_size=block_size, n_blocks=n_blocks)
    
