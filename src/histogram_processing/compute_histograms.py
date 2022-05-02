from typing_extensions import assert_type
from matplotlib import image
import pybind_kernels.histograms as histograms
import numpy as np
import sys
import h5py
import timeit
from datetime import datetime
from PIL import Image
from tqdm import tqdm

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
    histograms.field_histogram_resample_par_cpu(voxels, field, bins, vmin, vmax)
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

def load_data(experiment):
    dm = h5py.File(f'{h5root}/hdf5-byte/msb/{experiment}.h5', 'r')
    dl = h5py.File(f'{h5root}/hdf5-byte/lsb/{experiment}.h5', 'r')
    Nz, Ny, Nx = dm['voxels'].shape
    block_size = 256
    blocks = int(np.ceil(Nz / block_size))
    blocks = 7
    result = np.empty((blocks*block_size, Ny, Nx), dtype=np.uint16)
    for i in tqdm(range(blocks), desc='Loading voxels'): 
        start, stop = i*block_size, min((i+1)*block_size, dm['voxels'].shape[0]-1)
        result[start:stop] = (dm['voxels'][start:stop].astype(np.uint16) << 8) | dl['voxels'][start:stop].astype(np.uint16)
    dm.close()
    dl.close()
    return result

def run_in_core(sample, voxel_bins=4096, y_cutoff=1300, implant_threshold=32000):
    dm = h5py.File(f'{h5root}/hdf5-byte/msb/{sample}.h5', 'r')
    dl = h5py.File(f'{h5root}/hdf5-byte/lsb/{sample}.h5', 'r')
    fi = h5py.File(f'{h5root}/processed/implant-edt/2x/{sample}.h5', 'r')

    Nz, Ny, Nx = dm['voxels'].shape
    Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1
    center = ((Ny//2) - y_cutoff, Nx//2)
    Ny -= y_cutoff
    fz, fy, fx = fi['voxels'].shape
    fy -= y_cutoff // 2

    vmin, vmax = 0.0, float(implant_threshold)
    fmin, fmax = 4.0, 65535.0 # TODO don't hardcode.

    x_bins = np.zeros((Nx, voxel_bins), dtype=np.uint64)
    y_bins = np.zeros((Ny, voxel_bins), dtype=np.uint64)
    z_bins = np.zeros((Nz, voxel_bins), dtype=np.uint64)
    r_bins = np.zeros((Nr, voxel_bins), dtype=np.uint64)
    f_bins = np.zeros((voxel_bins//2, voxel_bins), dtype=np.uint64)

    voxels = np.empty((Nz,Ny,Nx), dtype=np.uint16)
    voxels[:,:,:] = \
            (dm['voxels'][:,y_cutoff:,:].astype(np.uint16) << 8) | \
            (dl['voxels'][:,y_cutoff:,:].astype(np.uint16))
    field = np.zeros(np.array(voxels.shape)//2, dtype=np.uint16)
    field[:fz,:fy,:fx] = fi['voxels'][:,y_cutoff//2,:].astype(np.uint16)

    histograms.axis_histogram_par_cpu(voxels, (0, 0, 0), Nz, x_bins, y_bins, z_bins, r_bins, center, (vmin, vmax), False)
    histograms.field_histogram_par_cpu(voxels, field, (0, 0, 0), (Nz, Ny, Nx), (fz, fy, fx), Nz, f_bins, (vmin, vmax), (fmin, fmax))

    dm.close()
    dl.close()
    fi.close()

    f_bins[-1] = 0 # TODO "bright" mask hack

    return x_bins, y_bins, z_bins, r_bins, f_bins


# Edition where the histogram is loaded and processed in chunks
def run_out_of_core(sample, block_size=128, voxel_bins=4096, y_cutoff=1300, implant_threshold=32000):
    dm = h5py.File(f'{h5root}/hdf5-byte/msb/{sample}.h5', 'r')
    dl = h5py.File(f'{h5root}/hdf5-byte/lsb/{sample}.h5', 'r')
    fi = h5py.File(f'{h5root}/processed/implant-edt/2x/{sample}.h5', 'r')

    Nz, Ny, Nx = dm['voxels'].shape
    Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1
    center = ((Ny//2) - y_cutoff, Nx//2)
    Ny -= y_cutoff
    voxels = np.empty((block_size, Ny, Nx), dtype=np.uint16)

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

    for i in tqdm(range(blocks), desc='Computing histograms'):
        field = np.zeros((block_size//2,fy,fx), dtype=np.uint16)
        zstart = i*block_size
        zstop = min((i+1)*block_size, Nz)
        fzstart = i*(block_size//2)
        fzstop = min((i+1)*(block_size//2), fz)

        if fzstop-fzstart > 0:
            field[:fzstop-fzstart,:,:] = fi['voxels'][fzstart:fzstop,y_cutoff_770c_pag//2:,:].astype(np.uint16)

        voxels[:zstop-zstart,:,:] = \
            (dm['voxels'][zstart:zstop,y_cutoff:,:].astype(np.uint16) << 8) | \
            (dl['voxels'][zstart:zstop,y_cutoff:,:].astype(np.uint16))
        
        
        histograms.axis_histogram_par_cpu(voxels, (zstart, 0, 0), block_size, x_bins, y_bins, z_bins, r_bins, center, (vmin, vmax), False)
        histograms.field_histogram_par_cpu(voxels, field, (zstart, 0, 0), (Nz, Ny, Nx), (fz, fy, fx), block_size, f_bins, (vmin, vmax), (fmin, fmax))
    
    dm.close()
    dl.close()
    fi.close()

    f_bins[-1] = 0 # TODO "bright" mask hack

    return x_bins, y_bins, z_bins, r_bins, f_bins

if __name__ == '__main__':
    h5root = '/mnt/shared/MAXIBONE/Goats/tomograms/'

    y_cutoff_770c_pag = 1300
    implant_threshold_u16 = 32000
    voxel_bins = 4096

    if len(sys.argv) > 1:
        sample = sys.argv[1]
    else:
        sample = '770c_pag'
    outpath = f'{h5root}/processed/histograms/{sample}/'

    xb, yb, zb, rb, fb = run_out_of_core(sample)

    Image.fromarray(tobyt(xb)).save(f"{outpath}/xb.png")
    Image.fromarray(tobyt(yb)).save(f"{outpath}/yb.png")
    Image.fromarray(tobyt(zb)).save(f"{outpath}/zb.png")
    Image.fromarray(tobyt(rb)).save(f"{outpath}/rb.png")
    Image.fromarray(tobyt(fb)).save(f"{outpath}/fb.png")
    np.savez(f'{outpath}/bins.npz', x_bins=xb, y_bins=yb, z_bins=zb, r_bins=rb, field_bins=fb)
