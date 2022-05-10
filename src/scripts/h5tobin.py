#!/usr/bin/env python3
import sys, pathlib, h5py, numpy as np
sys.path.append(sys.path[0]+"/../")
import pybind_kernels.histograms as histograms
from config.paths import hdf5_root, binary_root, commandline_args
from tqdm import tqdm

slice_all = slice(None)

def slice_length(s,n):
    start = s.start if s.start is not None else 0
    stop  = s.stop  if s.stop  is not None else n # -1?
    step  = s.step  if s.step  is not None else 1
    return (stop - start)//step

def h5tobin(sample,region=(slice_all,slice_all,slice_all),shift_volume_match=1):
    # Generate 16 bit flat binary blob for full sample tomogram
    # Read/write a full subbvolume at a time.x
    # For each subvolume, correct for subvolume_matching_shifts
    msb_file    = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    lsb_file    = h5py.File(f'{hdf5_root}/hdf5-byte/lsb/{sample}.h5', 'r')
    dmsb, dlsb  = msb_file['voxels'], lsb_file['voxels']
    Nz, Ny, Nx  = dmsb.shape    


    pathlib.Path(f"{binary_root}/voxels/1x/").mkdir(parents=True, exist_ok=True)    
    outfile = f'{binary_root}/voxels/1x/{sample}_voxels.uint16'

    subvolume_dims = msb_file['subvolume_dimensions'][:]
    vm_shifts      = msb_file['volume_matching_shifts'][:]
    Nvols          = len(subvolume_dims)
    Nzs            = subvolume_dims[:,0]

    # The ith subvolume in input  starts at sum(Nzs[:(i-1)]) and ends at sum(Nzs[:i])
    # The ith subvolume in output starts at sum(Nzs[:(i-1)]) - sum(vm_shifts[:i])
    input_zstarts         = np.concatenate([[0], np.cumsum(Nzs[:-1])]).astype(int)
    input_zends           = (np.cumsum(Nzs) - np.concatenate([vm_shifts,[0]])).astype(int)
    
    output_zstarts        = np.concatenate([[0], np.cumsum(Nzs[:-1]) - np.cumsum(vm_shifts)])    
    output_zends          = np.concatenate([output_zstarts[1:], [output_zstarts[-1]+Nzs[-1]]])

    print(f'HDF5 voxel data:')
    print(f'subvolume_dims =\n{subvolume_dims}')
    print(f'Nzs = {Nzs}')
    print(f'vm_shifts = {vm_shifts}')    
    print(f'input_zstarts  = {input_zstarts}')
    print(f'input_zends    = {input_zends}')
    
    print(f'output_zstarts = {output_zstarts}')
    print(f'output_zends   = {output_zends}')

    print(f'Shape to extract:\n{region}')
    
    assert((input_zends - input_zstarts == output_zends - output_zstarts).all())
    nzs = input_zends - input_zstarts # Actual number of z-slices per subvolume after vm-correction
 #   sys.exit(0)

    # TODO: z_range is ignored
    # TODO: Store metadata about region range in json
    # TODO: Come up with appropriate "file format" scheme
    # TODO: append_file should be in io pybind module, not histograms
    # TODO: command-line specified output dtype
    # TODO: cross-section thumbnails
    z_range, y_range, x_range = region
    for i in tqdm(range(Nvols), desc=f'Loading {sample} from HDF5 and writing binary'):
        subvolume = \
            (dmsb[input_zstarts[i]:input_zends[i],y_range,x_range].astype(np.uint16) << 8) | \
            (dlsb[input_zstarts[i]:input_zends[i],y_range,x_range].astype(np.uint16))
        print(f'subvolume.shape = {subvolume.shape}')
        histograms.append_slice(subvolume, outfile)
    
        
if __name__ == "__main__":
    sample, y_cutoff, shift_volume_match = commandline_args({"sample":"<required>",
                                                             "y_cutoff": 0,
                                                             "shift_volume_match":1})

    region = (slice_all,slice(y_cutoff,None), slice_all)
    h5tobin(sample,region,shift_volume_match)
    
