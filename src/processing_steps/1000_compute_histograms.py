#! /usr/bin/python3
'''
Computes the axes and field histograms for a given sample. If mask is provided, it is applied to the volume before computing the histograms.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.constants import implant_threshold_u16
from config.paths import binary_root, hdf5_root, get_plotting_dir
from datetime import datetime
import h5py
from lib.cpp.cpu_seq.histograms import axis_histograms as axis_histogram_seq_cpu, field_histogram as field_histogram_seq_cpu
from lib.cpp.cpu.histograms import axis_histograms as axis_histogram_par_cpu, field_histogram as field_histogram_par_cpu
from lib.cpp.gpu.histograms import axis_histograms as axis_histogram_par_gpu, field_histogram as field_histogram_par_gpu
from lib.cpp.cpu_seq.general import masked_minmax
from lib.cpp.cpu_seq.io import load_slice
from lib.py.commandline_args import add_volume, default_parser
from lib.py.helpers import chunk_info, load_chunk, row_normalize, to_int
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import timeit
from tqdm import tqdm

def axes_histogram_in_memory(voxels, func, ranges, voxel_bins, verbose):
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
    if verbose >= 2: print ("Entering call", datetime.now())
    func(voxels, (0,0,0), x_bins, y_bins, z_bins, r_bins, center, (vmin, vmax), verbose)
    if verbose >= 2: print ("Exited call", datetime.now())

    return x_bins, y_bins, z_bins, r_bins

def field_histogram_in_memory(voxels, field, func, ranges, voxel_bins, field_bins, verbose):
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
    `verbose` : int
        The verbosity level.

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

    if verbose >= 2: print ("Entering call", datetime.now())
    func(voxels, field, (0,0,0), bins, vrange, frange, verbose)
    if verbose >= 2: print ("Exited call", datetime.now())

    return bins

def verify_axes_histogram(voxels, ranges, outpath, voxel_bins, verbose):
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
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `verified` : bool
        Whether the different implementations of the axes histogram produced the same results.
    '''

    tolerance = 1e-5
    if verbose >= 1: print ('Axes: Running sequential CPU version')
    schx, schy, schz, schr = axes_histogram_in_memory(voxels, axis_histogram_seq_cpu, ranges, voxel_bins, verbose)

    # Check that the sequential CPU version produced any results
    seq_cpu_verified = False
    if (schx.sum() > 0 and schy.sum() > 0 and schz.sum() > 0 and schr.sum() > 0):
        seq_cpu_verified = True

    if verbose >= 1: print ('Axes: Running parallel CPU version')
    pchx, pchy, pchz, pchr = axes_histogram_in_memory(voxels, axis_histogram_par_cpu, ranges, voxel_bins, verbose)

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
        print ('Parallel CPU version did not match sequential CPU version.')
        print (f'diff x = {dx}')
        print (f'diff y = {dy}')
        print (f'diff z = {dz}')
        print (f'diff r = {dr}')
        print ('---------------------------------')

    if verbose >= 1: print ('Axes: Running parallel GPU version')
    pghx, pghy, pghz, pghr = axes_histogram_in_memory(voxels, axis_histogram_par_gpu, ranges, voxel_bins, verbose)

    dx = np.abs(schx - pghx).sum()
    dy = np.abs(schy - pghy).sum()
    dz = np.abs(schz - pghz).sum()
    dr = np.abs(schr - pghr).sum()

    par_gpu_verified = False
    if (dx < tolerance and dy < tolerance and dz < tolerance and dr < tolerance):
        par_gpu_verified = True
    else:
        print ('Parallel GPU version did not match sequential CPU version.')
        print (f'diff x = {dx}')
        print (f'diff y = {dy}')
        print (f'diff z = {dz}')
        print (f'diff r = {dr}')
        print ('---------------------------------')

    verified = seq_cpu_verified and par_cpu_verified and par_gpu_verified
    if verified:
        if verbose >= 1: print ('Axes: Both parallel CPU and GPU matched sequential CPU version')

    return verified

def verify_field_histogram(voxels, field, ranges, outpath, voxel_bins, field_bins, verbose):
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
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `verified` : bool
        Whether the different implementations of the field histogram produced the same results.
    '''

    tolerance = 1e-5
    if verbose >= 1: print ('Field - Running sequential CPU version')
    sch = field_histogram_in_memory(voxels, field, field_histogram_seq_cpu, ranges, voxel_bins, field_bins, verbose)

    Image.fromarray(to_int(row_normalize(sch), np.uint8)).save(f"{outpath}/field_verification_seq_cpu.png")

    # Check that the sequential CPU version produced any results
    seq_cpu_verified = False
    if (sch.sum() > 0):
        seq_cpu_verified = True

    if verbose >= 1: print ('Field - Running parallel CPU version')
    pch = field_histogram_in_memory(voxels, field, field_histogram_par_cpu, ranges, voxel_bins, field_bins, verbose)

    Image.fromarray(to_int(row_normalize(pch), np.uint8)).save(f"{outpath}/field_verification_par_cpu.png")

    d = np.abs(sch - pch).sum()

    par_cpu_verified = False
    if (d < tolerance):
        par_cpu_verified = True
    else:
        print ('Parallel CPU version did not match sequential CPU version.')
        print (f'par cpu diff = {d}')
        print ('---------------------------------')

    if verbose >= 1: print ('Field - Running parallel GPU version')
    pgh = field_histogram_in_memory(voxels, field, field_histogram_par_gpu, ranges, voxel_bins, field_bins, verbose)

    Image.fromarray(to_int(row_normalize(pgh), np.uint8)).save(f"{outpath}/field_verification_gpu.png")

    d = np.abs(sch - pgh).sum()

    par_gpu_verified = False
    if (d < tolerance):
        par_gpu_verified = True
    else:
        print ('Parallel GPU version did not match sequential CPU version.')
        print (f'gpu diff = {d}')
        print ('---------------------------------')

    verified = seq_cpu_verified and par_cpu_verified and par_gpu_verified
    if verified and verbose >= 1:
        print ('Field: Both parallel CPU and GPU matched sequential CPU version')

    return verified

def benchmark_axes_histograms(voxels, ranges, voxel_bins, runs, verbose):
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
    `verbose` : int
        The verbosity level.

    Returns
    -------
    None
    '''

    # TODO Move benchmarking out of this script.
    print()
    print('----- Benchmarking axes histograms -----')
    print()
    seq_cpu = timeit.timeit(lambda: axes_histogram_in_memory(voxels, axis_histogram_seq_cpu, ranges, voxel_bins, verbose), number=1)
    par_cpu = timeit.timeit(lambda: axes_histogram_in_memory(voxels, axis_histogram_par_cpu, ranges, voxel_bins, verbose), number=runs)
    par_gpu = timeit.timeit(lambda: axes_histogram_in_memory(voxels, axis_histogram_par_gpu, ranges, voxel_bins, verbose), number=runs)

    mean_seq_cpu = seq_cpu / 1
    mean_par_cpu = par_cpu / runs
    mean_par_gpu = par_gpu / runs

    print (f'Average of {runs} runs:')
    print (f'Seq CPU: {mean_seq_cpu:9.04f}')
    print (f'Par CPU: {mean_par_cpu:9.04f} (speedup: {mean_seq_cpu / mean_par_cpu:7.02f}x)')
    print (f'Par GPU: {mean_par_gpu:9.04f} (speedup: {mean_seq_cpu / mean_par_gpu:7.02f}x)')

def benchmark_field_histograms(voxels, field, ranges, voxel_bins, field_bins, runs, verbose):
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
    `verbose` : int
        The verbosity level.

    Returns
    -------
    None
    '''

    print()
    print('----- Benchmarking field histograms -----')
    print()
    seq_cpu = timeit.timeit(lambda: field_histogram_in_memory(voxels, field, field_histogram_seq_cpu, ranges, voxel_bins, field_bins, verbose), number=1)
    par_cpu = timeit.timeit(lambda: field_histogram_in_memory(voxels, field, field_histogram_par_cpu, ranges, voxel_bins, field_bins, verbose), number=runs)
    par_gpu = timeit.timeit(lambda: field_histogram_in_memory(voxels, field, field_histogram_par_gpu, ranges, voxel_bins, field_bins, verbose), number=runs)

    mean_seq_cpu = seq_cpu / 1
    mean_par_cpu = par_cpu / runs
    mean_par_gpu = par_gpu / runs

    print (f'Average of {runs} runs:')
    print (f'Seq CPU: {mean_seq_cpu:9.04f}')
    print (f'Par CPU: {mean_par_cpu:9.04f} (speedup: {mean_seq_cpu / mean_par_cpu:7.02f}x)')
    print (f'Par GPU: {mean_par_gpu:9.04f} (speedup: {mean_seq_cpu / mean_par_gpu:7.02f}x)')

def verify_and_benchmark(voxels, field, outpath, bins, runs, verbose):
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
    `runs` : int
        The number of runs to average the benchmark over.
    `verbose` : int
        The verbosity level.

    Returns
    -------
    None
    '''

    vrange, frange = ( (1, 32000), (1, 2**16-1) )
    axes_verified = verify_axes_histogram(voxels, vrange, outpath, bins, verbose)
    assert axes_verified
    fields_verified = verify_field_histogram(voxels, field, (vrange, frange), outpath, bins, bins, verbose)
    assert fields_verified
    benchmark_axes_histograms(voxels, vrange, bins, runs, verbose)
    benchmark_field_histograms(voxels, field, (vrange, frange), bins, bins, runs, verbose)

def run_out_of_core(sample, scale=1, chunk_size=128, z_offset=0, n_chunks=0,
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
    `chunk_size` : int
        The size of the chunks to load from disk. If set to 0, the chunk size is the size of a subvolume.
    `z_offset` : int
        The offset to start loading the volume from.
    `n_chunks` : int
        The number of chunks to load from disk. If set to 0, all chunks are loaded.
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

    bi = chunk_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', scale, chunk_size, n_chunks, z_offset, verbose)
    (Nz,Ny,Nx,Nr) = bi['dimensions']
    chunk_size    = bi['chunk_size']
    n_chunks      = bi['n_chunks']
    chunks_are_subvolumes = bi['chunks_are_subvolumes']

    center = (Ny//2,Nx//2)
    (vmin,vmax), (fmin,fmax) = value_ranges

    Nfields = len(field_names)
    x_bins = np.zeros((Nx, voxel_bins), dtype=np.uint64)
    y_bins = np.zeros((Ny, voxel_bins), dtype=np.uint64)
    z_bins = np.zeros((Nz, voxel_bins), dtype=np.uint64)
    r_bins = np.zeros((Nr, voxel_bins), dtype=np.uint64)
    f_bins = np.zeros((Nfields,field_bins, voxel_bins), dtype=np.uint64)

    chunk_iter = tqdm(range(n_chunks), desc='Computing histograms') if verbose >= 1 else range(n_chunks)
    for b in chunk_iter:
        if chunks_are_subvolumes:
            zstart     = bi['subvolume_starts'][z_offset+b]
            chunk_size = bi['subvolume_nzs'][z_offset+b]
        else:
            zstart = z_offset + b*chunk_size

        voxels, fields = load_chunk(sample, scale, zstart, chunk_size, mask, mask_scale, field_names, field_scale, verbose)

        axis_histogram_par_gpu(voxels, (zstart, 0, 0), x_bins, y_bins, z_bins, r_bins, center, (vmin, vmax), verbose)

        for i in range(Nfields):
            field_histogram_par_gpu(voxels, fields[i], (zstart, 0, 0), f_bins[i], (vmin, vmax), (fmin, fmax), verbose)

    if verbose >= 1: print ("Smoothing histograms")

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
    argparser = default_parser(__doc__)
    argparser = add_volume(argparser, 'field', 2, 'implant-gauss')
    argparser = add_volume(argparser, 'mask', 8, 'None')
    argparser.add_argument('--z-offset', type=int, default=0,
        help='The offset to start loading the volume from.')
    argparser.add_argument('--n-chunks', type=int, default=0,
        help='The number of chunks to load from disk. If set to 0, all chunks are loaded.')
    argparser.add_argument('--suffix', type=str, default='',
        help='The suffix to append to the output files.')
    argparser.add_argument('--voxel-bins', type=int, default=4096,
        help='The number of bins to use for the voxel histograms.')
    argparser.add_argument('--field-bins', type=int, default=2048,
        help='The number of bins to use for the field histograms.')
    argparser.add_argument('--benchmark', action='store_true',
        help='Whether to benchmark the different implementations of the histogram.')
    argparser.add_argument('--benchmark-runs', type=int, default=10,
        help='The number of runs to average the benchmark over.')
    args = argparser.parse_args()

    outpath = f'{hdf5_root}/processed/histograms/{args.sample}'
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    if args.plotting:
        plotting_dir = get_plotting_dir(args.sample, args.sample_scale)
        pathlib.Path(plotting_dir).mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        if args.verbose >= 1: print(f'Benchmarking axes_histograms for {args.sample} at scale {args.sample_scale}x')
        h5meta = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{args.sample}.h5', 'r')
        Nz, Ny, Nx = h5meta['voxels'].shape
        Nz -= sum(h5meta['volume_matching_shifts'][:])
        scaled_shape = (Nz//args.sample_scale, Ny//args.sample_scale, Nx//args.sample_scale)
        voxels = np.empty(scaled_shape, dtype=np.uint16)
        start = datetime.now()
        load_slice(voxels, f'{binary_root}/voxels/{args.sample_scale}x/{args.sample}.uint16', (0, 0, 0), scaled_shape)
        field = np.load(f'{binary_root}/fields/implant-gauss/{args.field_scale}x/{args.sample}.npy')
        end = datetime.now()
        # Align them
        voxels = voxels[:field.shape[0]*2,:,:]

        if args.verbose >= 1: print (voxels.shape, field.shape, (Nz, Ny, Nx), field.dtype)
        gb = (voxels.nbytes + field.nbytes) / 1024**3
        if args.verbose >= 1: print(f"Loaded {gb:.02f} GB in {end-start} ({gb / (end-start).total_seconds()} GB/s)")
        verify_and_benchmark(voxels, field, plotting_dir, args.voxel_bins // args.sample_scale, args.benchmark_runs, args.verbose)
    else:
        h5meta = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{args.sample}.h5', 'r')
        if h5meta["novisim"]:
            implant_threshold = 40000
        else:
            implant_threshold = implant_threshold_u16
        (vmin,vmax), (fmin,fmax) = ((1,implant_threshold), (1,2**16-1)) # TODO: Compute from total voxel histogram resp. total field histogram
        field_names = ["edt", "gauss", "gauss+edt"] # Should this be commandline defined?

        xb, yb, zb, rb, fb = run_out_of_core(args.sample, args.sample_scale, args.chunk_size, args.z_offset, args.n_chunks,
                                            None if args.mask=="None" else args.mask, args.mask_scale, args.voxel_bins, args.field_bins,
                                            field_names, args.field_scale, ((vmin,vmax),(fmin,fmax)),
                                            args.verbose)

        if args.plotting:
            if args.verbose >= 1: print(f"Saving histograms plots to {plotting_dir}")

            Image.fromarray(to_int(row_normalize(xb), np.uint8)).save(f"{plotting_dir}/xb{args.suffix}.png")
            Image.fromarray(to_int(row_normalize(yb), np.uint8)).save(f"{plotting_dir}/yb{args.suffix}.png")
            Image.fromarray(to_int(row_normalize(zb), np.uint8)).save(f"{plotting_dir}/zb{args.suffix}.png")
            Image.fromarray(to_int(row_normalize(rb), np.uint8)).save(f"{plotting_dir}/rb{args.suffix}.png")

            for i in range(len(field_names)):
                Image.fromarray(to_int(row_normalize(fb[i]), np.uint8)).save(f"{plotting_dir}/fb-{field_names[i]}{args.suffix}.png")

        if args.verbose >= 1: print(f"Saving histograms to {outpath}/bins{args.suffix}.npz")

        np.savez(f'{outpath}/bins{args.suffix}.npz',
                x_bins=xb, y_bins=yb, z_bins=zb, r_bins=rb, field_bins=fb,
                axis_names=np.array(["x","y","z","r"]),
                field_names=field_names, suffix=args.suffix, mask=args.mask,
                sample=args.sample, z_offset=args.z_offset, chunk_size=args.chunk_size, n_chunks=args.n_chunks, value_ranges=np.array(((vmin,vmax),(fmin,fmax))))
