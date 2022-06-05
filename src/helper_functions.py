#!/usr/bin/env python3
import h5py, numpy as np, pybind_kernels.histograms as histograms
from config.paths import hdf5_root, binary_root
from tqdm import tqdm

def update_hdf5(filename,group_name,datasets,attributes,dimensions=None,
                compression="lzf",chunk_shape=None):

    f = h5py.File(filename,'a')

    g = f.require_group(group_name)
    
    for k in datasets:
        v = datasets[k]
        g.require_dataset(k,shape=v.shape,dtype=v.dtype,
                          compression=compression, chunks=chunk_shape)
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


def block_info(h5meta_filename,block_size, n_blocks,z_offset):
    print(f"Opening {h5meta_filename}")
    with h5py.File(h5meta_filename, 'r') as dm:
        vm_shifts  = dm["volume_matching_shifts"][:]
        Nz, Ny, Nx = dm['voxels'].shape
        Nz -= np.sum(vm_shifts)
        Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1


        subvolume_dimensions =  dm['subvolume_dimensions'][:]                
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


        return {'dimensions':(Nz,Ny,Nx,Nr),
                'n_blocks': n_blocks,
                'block_size': block_size,
                'blocks_are_subvolumes': blocks_are_subvolumes,
                'subvolume_dimensions': subvolume_dimensions,
                'subvolume_nzs': subvolume_nzs,
                'subvolume_starts': np.concatenate([[0],np.cumsum(subvolume_nzs)[:-1]])
        }


def load_block(sample, offset, block_size, mask_name, mask_scale, field_names):
    '''
    Loads a block of data from disk into memory.
    '''
    Nfields = len(field_names)

    dm = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    Nz, Ny, Nx = dm['voxels'].shape
    Nz -= np.sum(dm["volume_matching_shifts"][:])
    dm.close()
    print(block_size,Nz,offset)   
    block_size       = min(block_size, Nz-offset)

    voxels = np.zeros((block_size,Ny,Nx),    dtype=np.uint16)
    fields = np.zeros((Nfields,block_size//2,Ny//2,Nx//2), dtype=np.uint16)    

    if mask_name is not None:
        for i in tqdm(range(1),f"Loading {mask_name} mask from {hdf5_root}/masks/{mask_scale}x/{sample}.h5", leave=True):
            with h5py.File(f"{hdf5_root}/masks/{mask_scale}x/{sample}.h5","r") as h5mask:
                mask = h5mask[mask_name]["mask"][offset//mask_scale:offset//mask_scale + block_size//mask_scale]
            
    #TODO: Make voxel & field scale command line parameters
    for i in tqdm(range(1),f"Loading {voxels.shape} voxels from {binary_root}/voxels/1x/{sample}.uint16", leave=True):    
        histograms.load_slice(voxels, f'{binary_root}/voxels/1x/{sample}.uint16', (offset, 0, 0), (Nz, Ny, Nx)) # TODO: Don't use 3 different methods for load/store

    for i in tqdm(range(Nfields),f"Loading {binary_root}/fields/implant-{field_names}/2x/{sample}.npy",leave=True):
        fi = np.load(f"{binary_root}/fields/implant-{field_names[i]}/2x/{sample}.npy", mmap_mode='r')
        fields[i,:] = fi[offset//2:offset//2 + block_size//2]

    if mask_name is not None:
        nz, ny, nx = (block_size//mask_scale), Ny//mask_scale, Nx//mask_scale
        mask_1x = np.broadcast_to(mask[:,NA,:,NA,:,NA],(nz,mask_scale, ny,mask_scale, nx,mask_scale))
        mask_1x = mask_1x.reshape(nz*mask_scale,ny*mask_scale,nx*mask_scale)
        print(f"{voxels.shape}, {mask_1x.shape}")
        voxels[:nz*mask_scale] *= mask_1x               # block_size may not be divisible by mask_scale
        voxels[nz*mask_scale:] *= mask_1x[-1][NA,...]  # Remainder gets last line of mask
        
    return voxels, fields
    
