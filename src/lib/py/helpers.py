import sys
sys.path.append(sys.path[0]+"/../")

from config.paths import binary_root, hdf5_root
import h5py
from lib.cpp.cpu.io import load_slice
import os
import numpy as np
import pathlib
import tqdm

def commandline_args(defaults):
    keys = list(defaults.keys())

    helpstring = f"Syntax: {sys.argv[0]} "
    for k in keys:
        if(defaults[k] == "<required>"): helpstring += f" <{k}>"
        else:                            helpstring += f" [{k}:{defaults[k]}]"

    # Do we just want to know how to call this script?
    if(len(sys.argv)==2):
        if(sys.argv[1] == "--help" or sys.argv[1] == "-h"):
            print(helpstring, file=sys.stderr)
            sys.exit(1)

    # Fill in parameters from commandline and defaults, converting to appropriate types
    args = []
    for i in range(len(keys)):
        default = defaults[keys[i]]
        if(len(sys.argv)<=i+1):
            if(default == "<required>"):
                print(helpstring, file=sys.stderr)
                sys.exit(1)
            else:
                args.append(default)
        else:
            args.append(type(default)(sys.argv[i+1]))

    return args

def generate_cylinder_mask(nx):
    xs = np.linspace(-1, 1, nx)
    rs = np.sqrt(xs[np.newaxis,np.newaxis,:]**2 + xs[np.newaxis,:,np.newaxis]**2)
    return rs <= 1

def update_hdf5(filename,group_name,datasets={},attributes={},dimensions=None,
                compression=None,chunk_shape=None):

    output_dir = os.path.dirname(filename)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    f = h5py.File(filename,'a')

    if((group_name is not None) and (group_name != "/")):
        g = f.require_group(group_name)
    else:
       g = f

    for k in datasets:
        v = datasets[k]
        if(k in g): del g[k]
        g.create_dataset(k,shape=v.shape,dtype=v.dtype,
                          compression=compression, chunks=chunk_shape,maxshape=None)
        g[k][:] = v[:]

        if dimensions is not None:
            try:
                dims = dimensions[k]
                for i, description in enumerate(dims):
                    g[k].dims[i] = description
            except:
                pass

    for k in attributes:
        v = attributes[k]
        g.attrs[k] = v

    f.close()


#TODO: Use this for masks, no compression and no chunking default for small metadata datasets
def update_hdf5_mask(filename,group_name,datasets={},attributes={},dimensions=None,
                     compression="lzf",chunk_shape=None):
    update_hdf5(filename,group_name,datasets,attributes,dimensions,compression,chunk_shape)


def h5meta_info_volume_matched(sample):
    with h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5","r") as h5meta:
        vm_shifts  = h5meta["volume_matching_shifts"][:]
        Nz, Ny, Nx = h5meta['voxels'].shape
        Nz -= np.sum(vm_shifts)

        subvolume_dimensions =  h5meta['subvolume_dimensions'][:]
        subvolume_nzs = subvolume_dimensions[:,0] - np.append(vm_shifts,0)
        voxel_size    = h5meta["voxels"].attrs["voxelsize"]

        return ((Nz,Ny,Nx), subvolume_nzs, voxel_size)

def block_info(h5meta_filename,block_size=0, n_blocks=0,z_offset=0):
    print(f"Opening {h5meta_filename}")
    with h5py.File(h5meta_filename, 'r') as h5meta:
        vm_shifts  = h5meta["volume_matching_shifts"][:]
        Nz, Ny, Nx = h5meta['voxels'].shape
        Nz -= np.sum(vm_shifts)
        Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1

        subvolume_dimensions =  h5meta['subvolume_dimensions'][:]
        subvolume_nzs = subvolume_dimensions[:,0] - np.append(vm_shifts,0)

        if block_size == 0:
            # If block_size is 0, let each block be exactly a full subvolume
            blocks_are_subvolumes = True

            # Do either n_blocks subvolumes, or if n_blocks == 0: all remaining after offset
            if n_blocks == 0:
                n_blocks = len(subvolume_nzs)-z_offset

        else:
            blocks_are_subvolumes = False
            if n_blocks == 0:
                n_blocks = Nz // block_size + (Nz % block_size > 0)


        return {
            'dimensions' : (Nz,Ny,Nx,Nr),
            'voxel_size' :  h5meta["voxels"].attrs["voxelsize"],
            'n_blocks' : n_blocks,
            'block_size' : block_size,
            'blocks_are_subvolumes' : blocks_are_subvolumes,
            'subvolume_dimensions' : subvolume_dimensions,
            'subvolume_nzs' : subvolume_nzs,
            'subvolume_starts' : np.concatenate([[0],np.cumsum(subvolume_nzs)[:-1]])
        }


def load_block(sample, scale, offset, block_size, mask_name, mask_scale, field_names, field_scale):
    '''
    Loads a block of data from disk into memory.
    '''
    NA = np.newaxis
    Nfields = len(field_names)

    h5meta = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    Nz, Ny, Nx = h5meta['voxels'].shape
    fNz, fNy, fNx = Nz // field_scale, Ny // field_scale, Nx // field_scale
    Nz -= np.sum(h5meta["volume_matching_shifts"][:])
    h5meta.close()
#    print(block_size,Nz,offset)
    block_size       = min(block_size, Nz-offset)

    voxels = np.zeros((block_size,Ny,Nx),    dtype=np.uint16)
    fields = np.zeros((Nfields,block_size//field_scale,fNy,fNx), dtype=np.uint16)

    if mask_name is not None:
        for i in tqdm.tqdm(range(1),f"Loading {mask_name} mask from {hdf5_root}/masks/{mask_scale}x/{sample}.h5", leave=True):
            with h5py.File(f"{hdf5_root}/masks/{mask_scale}x/{sample}.h5","r") as h5mask:
                mask = h5mask[mask_name]["mask"][offset//mask_scale:offset//mask_scale + block_size//mask_scale]

    #TODO: Make voxel & field scale command line parameters
    for i in tqdm.tqdm(range(1),f"Loading {voxels.shape} voxels from {binary_root}/voxels/{scale}x/{sample}.uint16", leave=True):
        load_slice(voxels, f'{binary_root}/voxels/{scale}x/{sample}.uint16', (offset, 0, 0), (block_size, Ny, Nx)) # TODO: Don't use 3 different methods for load/store

    for i in tqdm.tqdm(range(Nfields),f"Loading {binary_root}/fields/implant-{field_names}/{field_scale}x/{sample}.npy",leave=True):
        fi = np.load(f"{binary_root}/fields/implant-{field_names[i]}/{field_scale}x/{sample}.npy", mmap_mode='r')
        fields[i,:] = fi[offset//field_scale:offset//field_scale + block_size//field_scale]

    if mask_name is not None:
        nz, ny, nx = (block_size//mask_scale), Ny//mask_scale, Nx//mask_scale
        mask_1x = np.broadcast_to(mask[:,NA,:,NA,:,NA],(nz,mask_scale, ny,mask_scale, nx,mask_scale))
        mask_1x = mask_1x.reshape(nz*mask_scale,ny*mask_scale,nx*mask_scale)
        voxels[:nz*mask_scale] *= mask_1x               # block_size may not be divisible by mask_scale
        voxels[nz*mask_scale:] *= mask_1x[-1][NA,...]  # Remainder gets last line of mask

#    plt.imshow(voxels[:,voxels.shape[1]//2,:]); plt.show()
#    plt.imshow(fields[0,:,fields[0].shape[1]//2,:]); plt.show()
    return voxels, fields

def row_normalize(A,r):
    na = np.newaxis
    return A/(r[:,na]+(r==0)[:,na])

def to_int(x,dtype):
    vmin, vmax = x.min(), x.max()
    # Ensure everything is float32, to ensure float32 computations
    int_max = np.float32(np.iinfo(dtype).max - 1)
    factor = np.float32(vmax - vmin + (vmin==vmax))
    vmin, vmax = np.float32(vmin), np.float32(vmax)
    result = x.astype(np.float32)
    result -= vmin
    result /= factor
    result *= int_max
    result = np.floor(result).astype(dtype)
    result += 1
    return result
