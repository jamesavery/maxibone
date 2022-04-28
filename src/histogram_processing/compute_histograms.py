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

def load_data(experiment):
    dm = h5py.File(f'{h5root}/hdf5-byte/msb/{experiment}.h5', 'r')
    dl = h5py.File(f'{h5root}/hdf5-byte/lsb/{experiment}.h5', 'r')
    Nz, Ny, Nx = dm['voxels'].shape
    block_size = 256
    blocks = int(np.ceil(Nz / block_size))
    blocks = 7
    result = np.ndarray((blocks*block_size, Ny, Nx), dtype=np.uint16)
    for i in tqdm(range(blocks), desc='Loading voxels'): # TODO nu 
        start, stop = i*block_size, min((i+1)*block_size, dm['voxels'].shape[0]-1)
        result[start:stop] = (dm['voxels'][start:stop].astype(np.uint16) << 8) | dl['voxels'][start:stop].astype(np.uint16)
    dm.close()
    dl.close()
    return result

# TODO make blocked so Anna doesn't kill it :(
if __name__ == '__main__':
    h5root = '/mnt/shared/MAXIBONE/Goats/tomograms/'

    y_cutoff_770c_pag = 1300
    implant_threshold_u16 = 32000
    voxel_bins = 2048

    if len(sys.argv) > 1:
        sample = sys.argv[1]
    else:
        sample = '770c_pag'
    outpath = f'{h5root}/processed/histograms/{sample}/'

    dataset = load_data(sample)
    Nz,Ny,Nx = dataset.shape
    vxs = np.empty((Nz, Ny-y_cutoff_770c_pag, Nx), dtype=np.uint16)
    vxs[:,:,:] = dataset[:,y_cutoff_770c_pag:,:]


    with h5py.File(f'{h5root}/processed/implant-edt/2x/770c_pag.h5', 'r') as field_h5:
        field = np.empty(np.array(vxs.shape)//2, dtype=np.uint16)
        field[:,:,:] = field_h5['voxels'][:(256*7)//2,y_cutoff_770c_pag//2:,:]

    ranges = masked_minmax(vxs) # vmin, vmax
    ranges = ranges[0], min(ranges[1], implant_threshold_u16)

    #axes_histogram(vxs, func=histograms.axis_histogram_par_cpu, ranges=(vmin,vmax), voxel_bins=4096)

    xb, yb, zb, rb = axes_histogram(vxs, func=histograms.axis_histogram_par_cpu, ranges=ranges, voxel_bins=voxel_bins)
    fb = field_histogram(vxs, field, field_bins=voxel_bins>>1, voxel_bins=voxel_bins, ranges=ranges)
    fb[-1] = 0 # TODO "bright" mask hack

    Image.fromarray(tobyt(xb)).save(f"{outpath}/xb.png")
    Image.fromarray(tobyt(yb)).save(f"{outpath}/yb.png")
    Image.fromarray(tobyt(zb)).save(f"{outpath}/zb.png")
    Image.fromarray(tobyt(rb)).save(f"{outpath}/rb.png")
    Image.fromarray(tobyt(fb)).save(f"{outpath}/fb.png")
    np.savez(f'{outpath}/bins.npz', x_bins=xb, y_bins=yb, z_bins=zb, r_bins=rb, field_bins=fb)

    #verified = verify_axes_histogram(vxs, ranges=(vmin,vmax), voxel_bins=4096)
    #if verified:
    #    benchmark_axes_histograms(vxs, ranges=(vmin,vmax), voxel_bins=4096, runs=1)
