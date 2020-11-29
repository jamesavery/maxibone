import numpy as np
import os
import sys
import h5py
import scipy.ndimage as ndi
import edt
from config.paths import hdf5_root, commandline_args

from esrf_read         import *;
from blockmap          import *
NA = np.newaxis

sample, scale, chunk_size, padding = commandline_args({"sample":"<required>","scale":1,'chunk_size':200, 'padding':23})

h5implant = h5py.File(f"{hdf5_root}/processed/implant/{scale}x/{sample}.h5",'r')
output    = h5py.File(f"{hdf5_root}/processed/implant/{scale}x/{sample}-edt.h5",'w')

(Nz,Ny,Nx) = h5implant['voxels'].shape

output_voxels = output.create_dataset("voxels",(Nz,Ny,Nx),dtype=np.float16)

print(output_voxels.shape)
for z in range(0,Nz,chunk_size):
    zend = min(Nz,z+chunk_size)
    top_padding = (z!=0)*padding
    bot_padding = (zend!=Nz)*padding
    
    print(f"Loading voxels {z-top_padding}:{zend+bot_padding} = ({z}-{top_padding}):({zend}+{bot_padding})")
    implant_mask = ~(h5implant['voxels'][z-top_padding:zend+bot_padding].astype(np.bool))
    print(implant_mask.mean())
    print(f"Calculating EDT")
    implant_edt = edt.edt(implant_mask).astype(np.float16)
    print(f"Writing out output_voxels[{z}:{zend}] = EDT[{top_padding}:{implant_edt.shape[0]-bot_padding}]. EDT.shape = {implant_edt.shape}, implant_mask.shape = {implant_mask.shape}")
    output_voxels[z:zend] = implant_edt[top_padding:implant_edt.shape[0]-bot_padding]
#    del implant_mask    
#    del implant_edt

h5implant.close()    
output.close()
