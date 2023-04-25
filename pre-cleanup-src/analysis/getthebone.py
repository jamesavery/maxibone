import bohrium as bh
import numpy as np
import scipy.ndimage as nd
from clustering import * 
from distributions import *
from esrf_read import *
from bitmaps import *
from time import time
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: Where to stick this?
def as_numpy(A): return bh.interop_numpy.get_array(A)

t0 = time()
dataroot="/home/avery/nobackup/maxibone-data/"
sample = "HA_xc520_50kev_1_88mu_implant_770c_002_pag";
#sample_disk = "disk1/esrf_april_2013_implant_770/vol_float/";
sample_disk = "770c_data/"
sample_info = esrf_read_xml(dataroot+sample_disk+sample+"/"+sample+".xml");

classifiers = np.load(dataroot+"/classification/absorption/"+sample+".npz")
Pbone       = bh.array(classifiers["Pc"][2]);
t1 = time()

# Let's try to do the whole thing in one go
#tomo_chunk =  esrf_edfrange_to_npy(sample_info,region):
print("Reading tomogram")
tomo_chunk = esrf_full_tomogram_bh(sample_info)
(Nz,Ny,Nx) = tomo_chunk.shape;
#bh.flush()
t2 = time()
#TODO: air_limit should be calculated
air_limit = 1300;

# TODO: rhos x-axis is np.linspace(-4,12,2048) -> ((tomo+4)/16)*2048
#       bone is 3rd class
#       Do this nicer
print("Calculating Ibone")
print("Pbone.shape=",Pbone.shape)
print(bh.bhary.check(Pbone))
print("tomo_chunk.shape=",tomo_chunk.shape)
print(bh.bhary.check(tomo_chunk))

tomo_chunk_i = (((tomo_chunk+4)/16)*2048).astype(bh.uint16)
Ibone = Pbone[tomo_chunk_i];
del tomo_chunk
del tomo_chunk_i
bh.flush()
t3 = time()


ys = bh.linspace(0,Ny-1,Ny)[None,:,None];

print("bone_volume_weak")
bone_volume_weak = dilation_cross(Ibone>0.9,n=2);
bone_volume_weak[:,:air_limit] = False;

#bh.flush()
t4 = time()


# TODO: 25 should be calculated
print("bone_volume_holes")
bone_volume_holes = closing_cross(bone_volume_weak,n=25) # TODO: Add efficient packbits operation to bitmaps.py
t5 = time()

bone_volume_weak_packed = np.packbits(bone_volume_weak)
bone_volume_packed = np.packbits(bone_volume_holes)
#bh.flush()
t6 = time()
print("Writing out mask")
np.savez_compressed(dataroot+"/classification/absorption/"+sample+"-bonemask.npz",bone_volume_packed=bone_volume_packed,bone_volume_weak_packed=bone_volume_weak_packed)
t7 = time()

print("Load metadata:",t1-t0)
print("Load tomogram:",t2-t1)
print("Probability indexing:",t3-t2)
print("Initial mask:",t4-t3)
print("Morphological closure:",t5-t4)
print("Packing bits:",t6-t5)
print("Store mask:",t7-t6)
print("Total:",t7-t0)

plt.imshow(np.array(Ibone[15].copy()))
plt.savefig(dataroot+"/classification/absorption/"+sample+"-Ibone.png")
plt.imshow(np.array(bone_volume_weak[15].copy()))
plt.savefig(dataroot+"/classification/absorption/"+sample+"-bone_volume_weak.png")
plt.imshow(np.array(bone_volume_holes[15].copy()))
plt.savefig(dataroot+"/classification/absorption/"+sample+"-bone_volume_holes.png")
