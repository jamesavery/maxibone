import bohrium as bh
import h5py
import h5tomo 
import matplotlib.pyplot as plt
from esrf_read import *
import sys

i = int(sys.argv[1])

dataroot="/data/maxibone/bone_data/esrf_dental_implants_april_2013/";
with open(dataroot+"/xmlfiles-pag.txt") as f:
    xmlfiles = [x.rstrip('\n') for x in f.readlines()]

sample_info = esrf_read_xml(f"{dataroot}/{xmlfiles[i]}")
sample = sample_info['experiment']
print(f"Loading {sample}")

tomo=esrf_full_tomogram_bh(sample_info)

print("Squeezing into byte-sized voxels")
(Nz,Ny,Nx) = tomo.shape
X = bh.linspace(-1,1,Nx)[None,:]
Y = bh.linspace(-1,1,Ny)[:,None]
mask = X*X+Y*Y>=1

def normalize(A,value_range):
    vmin,vmax = value_range
    return (A-vmin)/(vmax-vmin+1)

tomo_i = (normalize(tomo,(-3,8))*255).astype(np.uint8)

for z in range(Nz):
    if(z%100 == 0):
        print(z)
    tomo_i[z][mask] = 0;
print(f"Saving to {dataroot}/bytefiles/tomo_i-{sample}.raw")

tomo_i.tofile(f"{dataroot}/bytefiles/tomo_i-{sample}.raw")
bh.flush()
