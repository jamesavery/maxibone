#! /usr/bin/python3
'''
Computes the axes and field histograms for a given sample.
'''
import sys
sys.path.append(sys.path[0]+"/../")

from config.constants import implant_threshold_u16
from config.paths import *
from datetime import datetime
import h5py
from lib.cpp.cpu_seq.histograms import axis_histograms as axis_histogram_seq_cpu, field_histogram as field_histogram_seq_cpu
from lib.cpp.cpu.histograms import axis_histograms as axis_histogram_par_cpu, field_histogram as field_histogram_par_cpu
from lib.cpp.gpu.histograms import axis_histograms as axis_histogram_par_gpu, field_histogram as field_histogram_par_gpu
from lib.cpp.cpu_seq.general import masked_minmax
from lib.cpp.cpu_seq.io import load_slice
from lib.py.helpers import block_info, commandline_args, load_block, row_normalize, to_int
import numpy as np
import pathlib
from PIL import Image
import scipy.ndimage as ndi
import timeit
from tqdm import tqdm

def axes_histogram_in_memory(voxels, func=axis_histogram_seq_cpu, ranges=None, voxel_bins=256):
    '''
    Compute the axes histograms for a given volume.
    This function assumes that the target device has enough memory to compute the histograms.

    Parameters
    ----------
    `voxels` : numpy.array[uint16]
        The volume to compute the histograms for.
    `func` : function
        The function to use to compute the histograms.
    `ranges` : tuple
        The value ranges for the voxels.
    `voxel_bins` : int
        The number of bins to use for the histograms.

    Returns
    -------
    `x_bins` : numpy.array[uint64]
        The histogram for the x-axis.
    `y_bins` : numpy.array[uint64]
        The histogram for the y-axis.
    `z_bins` : numpy.array[uint64]
        The histogram for the z-axis.
    `r_bins` : numpy.array[uint64]
        The histogram for the radius.
    '''

    (Nz, Ny, Nx) = voxels.shape
    center = (Ny//2, Nx//2)
    Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1

    x_bins = np.zeros((Nx, voxel_bins), dtype=np.uint64)
    y_bins = np.zeros((Ny, voxel_bins), dtype=np.uint64)
    z_bins = np.zeros((Nz, voxel_bins), dtype=np.uint64)
    r_bins = np.zeros((Nr, voxel_bins), dtype=np.uint64)

    if ranges is None:
        vmin, vmax = masked_minmax(voxels)
    else:
        vmin, vmax = ranges
    if verbose >= 1: print ("Entering call", datetime.now())
    func(voxels, (0,0,0), x_bins, y_bins, z_bins, r_bins, center, (vmin, vmax), verbose >= 1)
    if verbose >= 1: print ("Exited call", datetime.now())

    return x_bins, y_bins, z_bins, r_bins

def field_histogram_in_memory(voxels, field, func=field_histogram_seq_cpu, ranges=None, voxel_bins=256, field_bins=256):
    '''
    Compute the field histogram for a given volume.
    This function assumes that the target device has enough memory to compute the histogram.

    Parameters
    ----------
    `voxels` : numpy.array[uint16]
        The volume to compute the histogram for.
    `field` : numpy.array[uint16]
        The field to compute the histogram for.
    `func` : function
        The function to use to compute the histogram.
    `ranges` : tuple
        The value ranges for the voxels and the field.
    `voxel_bins` : int
        The number of bins to use for the voxel histogram.
    `field_bins` : int
        The number of bins to use for the field histogram.

    Returns
    -------
    `bins` : numpy.array[uint64]
        The histogram for the field.
    '''

    bins = np.zeros((field_bins, voxel_bins), dtype=np.uint64)
    if ranges is None:
        vrange, frange = ( (1e4, 3e4), (1, 2**16-1) )
    else:
        vrange, frange = ranges

    if verbose >= 1: print ("Entering call", datetime.now())
    func(voxels, field, (0,0,0), bins, vrange, frange, verbose >= 1)
    if verbose >= 1: print ("Exited call", datetime.now())

    return bins

def verify_axes_histogram(voxels, ranges, outpath, voxel_bins=256):
    '''
    Verify that the different implementations (Sequential CPU, Parallel CPU and GPU) of the axes histogram produce the same results.

    Parameters
    ----------
    `voxels` : numpy.array[uint16]
        The volume to compute the histograms for.
    `ranges` : tuple[int, int]
        The value ranges for the voxels.
    `outpath` : str
        The directory to save the verification images to.
    `voxel_bins` : int
        The number of bins to use for the histograms.

    Returns
    -------
    `verified` : bool
        Whether the different implementations of the axes histogram produced the same results.
    '''

    tolerance = 1e-5
    print ('Running sequential CPU version')
    schx, schy, schz, schr = axes_histogram_in_memory(voxels, func=axis_histogram_seq_cpu, ranges=ranges, voxel_bins=voxel_bins)

    # Check that the sequential CPU version produced any results
    seq_cpu_verified = False
    if (schx.sum() > 0 and schy.sum() > 0 and schz.sum() > 0 and schr.sum() > 0):
        seq_cpu_verified = True

    print ('Running parallel CPU version')
    pchx, pchy, pchz, pchr = axes_histogram_in_memory(voxels, func=axis_histogram_par_cpu, ranges=ranges, voxel_bins=voxel_bins)

    Image.fromarray(to_int(row_normalize(pchx), np.uint8)).save(f"{outpath}/xb_verification.png")
    Image.fromarray(to_int(row_normalize(pchy), np.uint8)).save(f"{outpath}/yb_verification.png")
    Image.fromarray(to_int(row_normalize(pchz), np.uint8)).save(f"{outpath}/zb_verification.png")
    Image.fromarray(to_int(row_normalize(pchr), np.uint8)).save(f"{outpath}/rb_verification.png")

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

    print ('Running parallel GPU version')
    pghx, pghy, pghz, pghr = axes_histogram_in_memory(voxels, func=axis_histogram_par_gpu, ranges=ranges, voxel_bins=voxel_bins)

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

    verified = seq_cpu_verified and par_cpu_verified and par_gpu_verified
    if verified:
        print ('Both parallel CPU and GPU matched sequential CPU version')

    return verified

def verify_field_histogram(voxels, field, ranges, outpath, voxel_bins=256, field_bins=256):
    '''
    Verify that the different implementations (Sequential CPU, Parallel CPU and GPU) of the field histogram produce the same results.

    Parameters
    ----------
    `voxels` : numpy.array[uint16]
        The volume to compute the histograms for.
    `field` : numpy.array[uint16]
        The field to compute the histograms for.
    `ranges` : tuple
        The value ranges for the voxels and the field.
    `outpath` : str
        The directory to save the verification images to.
    `voxel_bins` : int
        The number of bins to use for the voxel histogram.
    `field_bins` : int
        The number of bins to use for the field histogram.

    Returns
    -------
    `verified` : bool
        Whether the different implementations of the field histogram produced the same results.
    '''

    tolerance = 1e-5
    print ('Running sequential CPU version')
    sch = field_histogram_in_memory(voxels, field, func=field_histogram_seq_cpu, ranges=ranges, voxel_bins=voxel_bins, field_bins=field_bins)

    Image.fromarray(to_int(row_normalize(sch), np.uint8)).save(f"{outpath}/field_verification_seq_cpu.png")

    # Check that the sequential CPU version produced any results
    seq_cpu_verified = False
    if (sch.sum() > 0):
        seq_cpu_verified = True

    print ('Field - Running parallel CPU version')
    pch = field_histogram_in_memory(voxels, field, func=field_histogram_par_cpu, ranges=ranges, voxel_bins=voxel_bins, field_bins=field_bins)

    Image.fromarray(to_int(row_normalize(pch), np.uint8)).save(f"{outpath}/field_verification_par_cpu.png")

    d = np.abs(sch - pch).sum()

    par_cpu_verified = False
    if (d < tolerance):
        par_cpu_verified = True
    else:
        print (f'par cpu diff = {d}')

    print ('Field - Running parallel GPU version')
    pgh = field_histogram_in_memory(voxels, field, func=field_histogram_par_gpu, ranges=ranges, voxel_bins=voxel_bins, field_bins=field_bins)

    Image.fromarray(to_int(row_normalize(pgh), np.uint8)).save(f"{outpath}/field_verification_gpu.png")

    d = np.abs(sch - pgh).sum()

    par_gpu_verified = False
    if (d < tolerance):
        par_gpu_verified = True
    else:
        print (f'gpu diff = {d}')

    verified = seq_cpu_verified and par_cpu_verified and par_gpu_verified
    if verified:
        print ('Field: Both parallel CPU and GPU matched sequential CPU version')

    return verified

def benchmark_axes_histograms(voxels, ranges=(1,4095), voxel_bins=256, runs=10):
    '''
    Benchmark the different implementations (Sequential CPU, Parallel CPU and GPU) of the axes histogram.

    Parameters
    ----------
    `voxels` : numpy.array[uint16]
        The volume to compute the histograms for.
    `ranges` : tuple
        The value ranges for the voxels.
    `voxel_bins` : int
        The number of bins to use for the histograms.
    `runs` : int
        The number of runs to average the benchmark over.

    Returns
    -------
    None
    '''

    # TODO Move benchmarking out of this script.
    print()
    print('----- Benchmarking -----')
    print()
    seq_cpu = timeit.timeit(lambda: axes_histogram_in_memory(voxels, func=axis_histogram_seq_cpu, ranges=ranges, voxel_bins=voxel_bins), number=1)
    par_cpu = timeit.timeit(lambda: axes_histogram_in_memory(voxels, func=axis_histogram_par_cpu, ranges=ranges, voxel_bins=voxel_bins), number=runs)
    par_gpu = timeit.timeit(lambda: axes_histogram_in_memory(voxels, func=axis_histogram_par_gpu, ranges=ranges, voxel_bins=voxel_bins), number=runs)

    mean_seq_cpu = seq_cpu / 1
    mean_par_cpu = par_cpu / runs
    mean_par_gpu = par_gpu / runs

    print (f'Average of {runs} runs:')
    print (f'Seq CPU: {mean_seq_cpu:9.04f}')
    print (f'Par CPU: {mean_par_cpu:9.04f} (speedup: {mean_seq_cpu / mean_par_cpu:7.02f}x)')
    print (f'Par GPU: {mean_par_gpu:9.04f} (speedup: {mean_seq_cpu / mean_par_gpu:7.02f}x)')

def benchmark_field_histograms(voxels, field, ranges, voxel_bins=256, field_bins=256, runs=10):
    '''
    Benchmark the different implementations (Sequential CPU, Parallel CPU and GPU) of the field histogram.

    Parameters
    ----------
    `voxels` : numpy.array[uint16]
        The volume to compute the histograms for.
    `field` : numpy.array[uint16]
        The field to compute the histograms for.
    `ranges` : tuple
        The value ranges for the voxels and the field.
    `voxel_bins` : int
        The number of bins to use for the voxel histogram.
    `field_bins` : int
        The number of bins to use for the field histogram.
    `runs` : int
        The number of runs to average the benchmark over.

    Returns
    -------
    None
    '''

    print()
    print('----- Benchmarking -----')
    print()
    seq_cpu = timeit.timeit(lambda: field_histogram_in_memory(voxels, field, func=field_histogram_seq_cpu, ranges=ranges, voxel_bins=voxel_bins, field_bins=field_bins), number=1)
    par_cpu = timeit.timeit(lambda: field_histogram_in_memory(voxels, field, func=field_histogram_par_cpu, ranges=ranges, voxel_bins=voxel_bins, field_bins=field_bins), number=runs)
    par_gpu = timeit.timeit(lambda: field_histogram_in_memory(voxels, field, func=field_histogram_par_gpu, ranges=ranges, voxel_bins=voxel_bins, field_bins=field_bins), number=runs)

    mean_seq_cpu = seq_cpu / 1
    mean_par_cpu = par_cpu / runs
    mean_par_gpu = par_gpu / runs

    print (f'Average of {runs} runs:')
    print (f'Seq CPU: {mean_seq_cpu:9.04f}')
    print (f'Par CPU: {mean_par_cpu:9.04f} (speedup: {mean_seq_cpu / mean_par_cpu:7.02f}x)')
    print (f'Par GPU: {mean_par_gpu:9.04f} (speedup: {mean_seq_cpu / mean_par_gpu:7.02f}x)')

def verify_and_benchmark(voxels, field, outpath, bins=4096):
    '''
    Verify and benchmark the different implementations (Sequential CPU, Parallel CPU and GPU) of the axes and field histograms.

    Parameters
    ----------
    `voxels` : numpy.array[uint16]
        The volume to compute the histograms for.
    `field` : numpy.array[uint16]
        The field to compute the histograms for.
    `outpath` : str
        The directory to save the verification images to.
    `bins` : int
        The number of bins to use for the histograms.

    Returns
    -------
    None
    '''

    vrange, frange = ( (1, 32000), (1, 2**16-1) )
    axes_verified = verify_axes_histogram(voxels, vrange, outpath, voxel_bins=bins)
    assert axes_verified
    fields_verified = verify_field_histogram(voxels, field, (vrange, frange), outpath, voxel_bins=bins, field_bins=bins)
    assert fields_verified
    benchmark_axes_histograms(voxels, vrange, voxel_bins=bins)
    benchmark_field_histograms(voxels, field, (vrange, frange), voxel_bins=bins, field_bins=bins)

def run_out_of_core(sample, scale=1, block_size=128, z_offset=0, n_blocks=0,
                    mask=None, mask_scale=8, voxel_bins=4096, field_bins=4096,
                    field_names=["gauss","edt","gauss+edt"],
                    field_scale=2,
                    value_ranges=((1e4,3e4),(1,2**16-1)),
                    verbose=1
):
    '''
    Compute the axes and field histograms for a given sample.
    This function assumes that the target device does not have enough memory to compute the histograms in memory.
    It loads the volume in chunks from disk and processes them one at a time.

    Parameters
    ----------
    `sample` : str
        The sample to compute the histograms for.
    `scale` : int
        The scale of the sample.
    `block_size` : int
        The size of the blocks to load from disk. If set to 0, the block size is the size of a subvolume.
    `z_offset` : int
        The offset to start loading the volume from.
    `n_blocks` : int
        The number of blocks to load from disk. If set to 0, all blocks are loaded.
    `mask` : str
        The mask to use for the volume. If set to None, no mask is used.
    `mask_scale` : int
        The scale of the mask.
    `voxel_bins` : int
        The number of bins to use for the voxel histograms.
    `field_bins` : int
        The number of bins to use for the field histograms.
    `field_names` : list
        The names of the fields to compute the histograms for.
    `field_scale` : int
        The scale of the fields.
    `value_ranges` : tuple
        The value ranges for the voxels and the fields.
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `x_bins` : numpy.array[uint64]
        The histogram for the x-axis.
    `y_bins` : numpy.array[uint64]
        The histogram for the y-axis.
    `z_bins` : numpy.array[uint64]
        The histogram for the z-axis.
    `r_bins` : numpy.array[uint64]
        The histogram for the radius.
    `f_bins` : numpy.array[uint64]
        The histogram for the fields.
    '''

    bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', scale, block_size, n_blocks, z_offset)
    (Nz,Ny,Nx,Nr) = bi['dimensions']
    block_size    = bi['block_size']
    n_blocks      = bi['n_blocks']
    blocks_are_subvolumes = bi['blocks_are_subvolumes']

    center = (Ny//2,Nx//2)
    (vmin,vmax), (fmin,fmax) = value_ranges

    Nfields = len(field_names)
    x_bins = np.zeros((Nx, voxel_bins), dtype=np.uint64)
    y_bins = np.zeros((Ny, voxel_bins), dtype=np.uint64)
    z_bins = np.zeros((Nz, voxel_bins), dtype=np.uint64)
    r_bins = np.zeros((Nr, voxel_bins), dtype=np.uint64)
    f_bins = np.zeros((Nfields,field_bins, voxel_bins), dtype=np.uint64)

    for b in tqdm(range(n_blocks), desc='Computing histograms'):
        if blocks_are_subvolumes:
            zstart     = bi['subvolume_starts'][z_offset+b]
            block_size = bi['subvolume_nzs'][z_offset+b]
        else:
            zstart = z_offset + b*block_size

        voxels, fields = load_block(sample, scale, zstart, block_size, mask, mask_scale, field_names, field_scale)

        for i in tqdm(range(1),"Histogramming over x,y,z axes and radius", leave=True):
            axis_histogram_par_gpu(voxels, (zstart, 0, 0), x_bins, y_bins, z_bins, r_bins, center, (vmin, vmax), verbose >= 1)

        for i in tqdm(range(Nfields),f"Histogramming w.r.t. fields {field_names}", leave=True):
            field_histogram_par_gpu(voxels, fields[i], (zstart, 0, 0), f_bins[i], (vmin, vmax), (fmin, fmax), verbose >= 1)

    f_bins[:, 0,:] = 0 # TODO EDT mask hack
    f_bins[:,-1,:] = 0 # TODO "bright" mask hack

    sigma = 3
    x_bins = ndi.gaussian_filter(x_bins, sigma, mode='constant', cval=0)
    y_bins = ndi.gaussian_filter(y_bins, sigma, mode='constant', cval=0)
    z_bins = ndi.gaussian_filter(z_bins, sigma, mode='constant', cval=0)
    r_bins = ndi.gaussian_filter(r_bins, sigma, mode='constant', cval=0)

    for i, bins in enumerate(f_bins):
        f_bins[i] = ndi.gaussian_filter(bins, sigma, mode='constant', cval=0)

    return x_bins, y_bins, z_bins, r_bins, f_bins

if __name__ == '__main__':
    sample, scale, field_scale, block_size, z_offset, n_blocks, suffix, \
    mask, mask_scale, voxel_bins, field_bins, benchmark, verbose \
        = commandline_args({
            "sample" : "<required>",
            "scale" : 1,
            "field_scale" : 2,
            "block_size" : 256,
            "z_offset" :  0,
            "n_blocks" : 0,
            "suffix" : "",
            "mask" : "None",
            "mask_scale" :  8,
            "voxel_bins" : 4096,
            "field_bins" : 2048,
            "benchmark" : False,
            "verbose" : 0
        })

    outpath = f'{hdf5_root}/processed/histograms/{sample}/'
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)

    if benchmark:
        print(f'Benchmarking axes_histograms for {sample} at scale {scale}x')
        h5meta = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
        Nz, Ny, Nx = h5meta['voxels'].shape # TODO this is not the volume matched shape!
        scaled_shape = (Nz//scale, Ny//scale, Nx//scale)
        voxels = np.empty(scaled_shape, dtype=np.uint16)
        start = datetime.now()
        load_slice(voxels, f'{binary_root}/voxels/{scale}x/{sample}.uint16', (0, 0, 0), scaled_shape)
        field = np.load(f'{binary_root}/fields/implant-gauss/{scale*2}x/{sample}.npy')
        end = datetime.now()
        # Align them
        voxels = voxels[:field.shape[0]*2,:,:]

        print (voxels.shape, field.shape, (Nz, Ny, Nx), field.dtype)
        gb = (voxels.nbytes + field.nbytes) / 1024**3
        print(f"Loaded {gb:.02f} GB in {end-start} ({gb/(end-start).total_seconds()} GB/s)")
        verify_and_benchmark(voxels, field, outpath, voxel_bins // scale)
    else:
        if 'novisim' in sample:
            implant_threshold = 40000
        else:
            implant_threshold = implant_threshold_u16
        (vmin,vmax), (fmin,fmax) = ((1,implant_threshold), (1,2**16-1)) # TODO: Compute from total voxel histogram resp. total field histogram
        field_names = ["edt", "gauss", "gauss+edt"] # Should this be commandline defined?

        xb, yb, zb, rb, fb = run_out_of_core(sample, scale, block_size, z_offset, n_blocks,
                                            None if mask=="None" else mask, mask_scale, voxel_bins, field_bins,
                                            field_names, field_scale, ((vmin,vmax),(fmin,fmax)),
                                            verbose)

        Image.fromarray(to_int(row_normalize(xb), np.uint8)).save(f"{outpath}/xb{suffix}.png")
        Image.fromarray(to_int(row_normalize(yb), np.uint8)).save(f"{outpath}/yb{suffix}.png")
        Image.fromarray(to_int(row_normalize(zb), np.uint8)).save(f"{outpath}/zb{suffix}.png")
        Image.fromarray(to_int(row_normalize(rb), np.uint8)).save(f"{outpath}/rb{suffix}.png")

        for i in range(len(field_names)):
            Image.fromarray(to_int(row_normalize(fb[i]), np.uint8)).save(f"{outpath}/fb-{field_names[i]}{suffix}.png")

        np.savez(f'{outpath}/bins{suffix}.npz',
                x_bins=xb, y_bins=yb, z_bins=zb, r_bins=rb, field_bins=fb,
                axis_names=np.array(["x","y","z","r"]),
                field_names=field_names, suffix=suffix, mask=mask,
                sample=sample, z_offset=z_offset, block_size=block_size, n_blocks=n_blocks, value_ranges=np.array(((vmin,vmax),(fmin,fmax))))
