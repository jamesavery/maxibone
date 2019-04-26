#!/usr/bin/python3
# What is air, and what is bone?
# (C) James Avery for the MAXIBONE project, 2018
import numpy as np;
from scipy import ndimage as ndi;
import skimage.feature as feat;
import skimage.morphology as morf;
import skimage.filters as filt;
from esrf_read import *;


def segment_cortical_block(block,info):
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

    


#    global vol, mass, cm;
  



        


    return {
        'hole_id':hole_id,
        'hole_type_id':   hole_type_id,
        'osteocyte_mask': hole_type_id==1,
        'vessel_mask'   : hole_type_id==2,
        'bone_mask'     : hole_type_id==0,
#        'n_osteocytes'  : n_osteocytes,
#        'n_vessels'     : n_vessels,
        'volumes': V,
        'skeleton_ix'   : nzi,
        'skeleton'      : skeleton
    };


    





