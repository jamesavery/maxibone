import jax.numpy as jp
import numpy as np
import jax
import os
import sys
import h5py
import scipy.ndimage as ndi
import edt

from esrf_read         import *;
from blockmap          import *
jax.config.update("jax_enable_x64", True)
NA = np.newaxis

sample, hdf5_root = sys.argv[1:] 


h5implant = h5py.File(f"{hdf5_root}/processed/implant/1x/{sample}.h5",'r')
h5data    = h5py.File(f"{hdf5_root}/hdf5-byte/msb/1x/{sample}.h5",'r')
output    = h5py.File(f"{hdf5_root}/processed/implant/1x/{sample}-edt.h5",'w')

subvolume_dimensions = h5data['subvolume_dimensions'][:]
n_subvolumes = len(subvolume_dimensions)
h5data.close()

for i in range(n_subvolumes):
    (nz,ny,nx) = subvolume_dimensions[i]
    print(f"Loading {(nz,ny,nx)} voxels in {sample} subvolume {i}")
    implant = ~h5implant[f"subvolume{i}"][:]
    print(f"Calculating EDT for {sample} subvolume {i}")
    implant_edt = edt.edt(implant).astype(np.float16)
    del implant
    print(f"Writing out EDT")
    output.create_dataset(f"subvolume{i}",data=implant_edt)
    del implant_edt


h5implant.close()    
output.close()
