import jax.numpy as jp
import numpy as np
import jax
import os
import sys
import h5py
import scipy.ndimage as ndi

from esrf_read         import *;
from blockmap          import *
from resample          import *
from static_thresholds import *
jax.config.update("jax_enable_x64", True)
NA = np.newaxis

def sphere(n):
    xs = np.linspace(-1,1,n)
    return (xs[:,NA,NA]**2 + xs[NA,:,NA]**2 + xs[NA,NA,:]**2) <= 1


sample, xml_root, hdf5_root = sys.argv[1:] 

xmlfiles     = readfile(f"{xml_root}/{sample}-xml.txt")
n_subvolumes = len(xmlfiles)
info_subvolumes = [esrf_read_xml(f"{xml_root}/{xml_filename.strip()}") for xml_filename in xmlfiles]

output = h5py.File(f"{hdf5_root}/processed/implant/1x/{sample}.h5",'w')

sph5 = np.array(sphere(5))

for i in range(n_subvolumes):
    info = info_subvolumes[i]
    [nz,ny,nx] = [int(info[s]) for s in ['sizez','sizey','sizex']]
    print(f"subvolume {i} dimension is {nz,ny,nx}")
    output.create_dataset(f"subvolume{i}",(nz,ny,nx),dtype=np.uint8,compression='lzf')

    print(f"Loading subvolume {i}")
    tomo    = esrf_full_tomogram(info)
    print(f"Thresholding subvolume {i}")
    implant = (tomo >= 1.2)
    del tomo
    print(f"Binary opening for subvolume {i}")
    implant = ndi.binary_opening(implant,sph5)
    print(f"Writing subvolume {i}")
    output[f'subvolume{i}'][:] = implant[:]
    del implant

output.close()
