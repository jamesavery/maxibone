import numpy as np
import os, sys, h5py
#from scipy.ndimage.morphology import distance_transform_edt as edt
from edt import edt
from config.paths import hdf5_root, commandline_args

NA = np.newaxis

sample, scale, chunk_size, padding = commandline_args({"sample":"<required>","scale":1,'chunk_size':200, 'padding':28})

h5implant = h5py.File(f"{hdf5_root}/processed/implant/{scale}x/{sample}.h5",'r')
output    = h5py.File(f"{hdf5_root}/processed/implant/{scale}x/{sample}-edt.h5",'w')

(Nz,Ny,Nx) = h5implant['voxels'].shape

output_voxels = output.create_dataset("voxels",(Nz,Ny,Nx),dtype=np.float16)

print(output_voxels.shape)
for z0 in range(0,Nz,chunk_size):
    z1     = min(Nz,z0+chunk_size)
    Z0, Z1 = max(0,z0-padding), min(Nz,z1+padding)
    
    print(f"Loading voxels {Z0}:{Z1} ({z0}:{z1} with padding)")
    implant_mask = ~(h5implant['voxels'][Z0:Z1].astype(np.bool))
    print(implant_mask.mean())
    print(f"Calculating EDT")
    implant_edt = edt(implant_mask).astype(np.float16)
    print(f"Writing out output_voxels[{z0}:{z1}] = EDT[{z0-Z0}:{z1-Z0}]. EDT.shape = {implant_edt.shape}, implant_mask.shape = {implant_mask.shape}")
    output_voxels[z0:z1] = implant_edt[z0-Z0:z1-Z0]
    del implant_mask    
    del implant_edt

h5implant.close()    
output.close()
