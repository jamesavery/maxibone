import numpy as np;
from scipy import ndimage as ndi;
import skimage.feature as feat;
import skimage.morphology as morf;
import skimage.filters as filt;
import nrrd;
from esrf_read import *;

import pyqtgraph as pg;
# 810c-pag-block.npy.npz
block=np.load("../data/810c-pag-block.npz")["block"];

test_scan_xml="/diskstation/data/xns/maxibone/esrf_dental_implants_2013/810c-repeat/HA_xc520_50kev_1_88mu_implant_810c_001repeat_pag.xml";

info = esrf_read_xml(test_scan_xml);

[osteocyte_length,osteocyte_width] = [15,7]; # All lengths are in micrometers
voxelsize = float(info['voxelsize']); 

print("1: Calculating threshold");
bone_threshold     = filt.threshold_otsu(block);
print("2: Applying threshold");
holes              = morf.closing(block<bone_threshold);
print("3: Skeletonizing");
skeleton           = morf.skeletonize_3d(holes);
print("4: Labeling");    
hole_id,num_holes  = ndi.label(holes);

print("5: Enumerating");
counts = np.zeros((num_holes+1),dtype=np.int64);
nzi    = np.array(np.nonzero(skeleton)).T;

for (x,y,z) in nzi:
    hid = hole_id[x,y,z];
    counts[hid] += 1;
    tally = np.bincount(counts);

vessel_cutoff=int(1.7*osteocyte_length/voxelsize);

print("7: Separating vessels and osteocytes.\n"
      +"    vessel_cutoff          = ",vessel_cutoff,"voxels (",vessel_cutoff*voxelsize," micrometers)\n"
      +"    osteocyte volume range = [200,6000]");


print("Compute volumes, total mass, and centres of mass.");


(tiny_osteocyte,huge_osteocyte) = (10,6000);
V = np.bincount(hole_id.flatten())*(voxelsize**3);

print("Long skeleton => vessel, short skeleton => possible osteocyte.");
# Osteocyte volume is in the range 500-5000 \mu m^3    
# hole_type = {bone:0, osteocyte:1, vessel:2, small uncertain soft tissue: 3, large uncertain soft tissue:4}            
hole_type = np.zeros((num_holes+1),dtype=np.int8);
for hid in range(1,num_holes+1):
    hole_type[hid] = 1+(counts[hid]>=vessel_cutoff);
    if(hole_type[hid] == 1 and V[hid] < tiny_osteocyte): hole_type[hid] = 3;
    if(hole_type[hid] == 1 and V[hid] > huge_osteocyte): hole_type[hid] = 4;        
    
hole_type_id = hole_type[hole_id];


def center_of_mass(b):
    (nx,ny,nz) = b.shape;
    (ix,iy,iz) = (np.arange(nx),np.arange(ny),np.arange(nz));
    vol = np.sum(b);
    cx  = np.sum(b*ix[:,None,None])/vol;
    cy  = np.sum(b*iy[None,:,None])/vol;
    cz  = np.sum(b*iz[None,None,:])/vol;

    return (cx,cy,cz);
    
def moment_of_inertia(rho,x=None):
    (nx,ny,nz) = rho.shape;
    (ix,iy,iz) = (np.arange(nx),np.arange(ny),np.arange(nz));

    if(x==None):
        x = center_of_mass(rho);
    
    # Change to x-coordinate system
    (rx,ry,rz) = (ix-x[0],iy-x[1],iz-x[2]);

    I = np.zeros((3,3),dtype=np.float);
    I[0,0] = np.sum( rho*(ry[None,:,None]**2 + rz[None,None,:]**2) );
    I[1,1] = np.sum( rho*(rx[:,None,None]**2 + rz[None,None,:]**2) );
    I[2,2] = np.sum( rho*(rx[:,None,None]**2 + ry[None,:,None]**2) );
    I[0,1] = I[1,0] = -np.sum( rho*rx[:,None,None]*ry[None,:,None]);
    I[0,2] = I[2,0] = -np.sum( rho*rx[:,None,None]*rz[None,None,:]);
    I[1,2] = I[2,1] = -np.sum( rho*ry[None,:,None]*rz[None,None,:]);

    return I;

#TODO: Closed-form solutions to eigensystem instead of NumPy
def principal_axes(I):
    lam,Ix = np.linalg.eigh(I);
#TODO: Sanity checks    
    return Ix;

def blob_principal_axes(osteocytes,cm,diameter):
    (start,end) = (int(np.round(x-diameter)), int(np.round(x+diameter+1)));

    rho = osteocytes[start[0]:end[0],start[1]:end[1],start[2]:end[2]];
    I   = moment_of_inertia(rho);
    return principal_axes(I);

def osteocytes_principal_axes(osteocyte_image,osteocyte_centres):
    osteocyte_length = 7;         # TODO: Calculate from voxelsize. TODO2: .py-file with biological sizes
    fn  = np.vectorize(lambda cm: blob_principal_axes(osteocyte_image,cm,1.5*osteocyte_length));

    return fn(osteocyte_centres);

def osteocyte_centres(block,osteocyte_nz, osteocyte_ids, num_osteocytes):
    vols = np.zeros((num_osteocytes+1));
    cms  = np.zeros((num_osteocytes+1),3);

    for (x,y,z) in osteocyte_nz:
        oid = osteocyte_ids[x,y,z];
        vols[oid] += block[x,y,z];
        cms[oid]  += (x,y,z);

    return cms;
