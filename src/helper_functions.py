#!/usr/bin/env python3
import h5py, numpy as np

def block_info(h5meta_filename,block_size, n_blocks):
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
