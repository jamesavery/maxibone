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

sample, xml_root, hdf5_root, scale = sys.argv[1:] 

xmlfiles     = readfile(f"{xml_root}/{sample}-xml.txt")
n_subvolumes = len(xmlfiles)
info_subvolumes = [esrf_read_xml(f"{xml_root}/{xml_filename.strip()}") for xml_filename in xmlfiles]

input  = h5py.File(f"{hdf5_root}/processed/implant/{scale}x/{sample}.h5",'r')
output = h5py.File(f"{hdf5_root}/processed/implant/{scale}x/{sample}-dispersion.h5",'w')



def gauss_dispersion(tomo):
    Igauss_cu = ndimage.gaussian_filter(implant_cu.astype(cp.float16),sigma=6,mode='reflect')
    for i in range(nx//6):
        if((i%50==0)): print(i)
        Igauss_cu = ndimage.gaussian_filter((Igauss_cu*(implant_cu==0)+implant_cu),sigma=6,mode='reflect')
    



