from esrf_read import *
import h5py
import numpy as np
import numpy.ma as ma
from h5tomo import *

def frame_histogram(frame,i,bin_edges):
#    print("Calculating histogram for frame",i)        
    count =  np.histogram(frame.compressed(),bins=bin_edges)[0];
#    print("Completed histogram for frame",i)
    return count

from time import time;
import os
dataroot = sys.argv[1];        

with open(dataroot+"/xmlfiles-pag.txt") as f:
    xmlfiles = [x.rstrip('\n') for x in f.readlines()]

    for i in range(36,len(xmlfiles)):
        t0 = time();
        xml = esrf_read_xml(dataroot+xmlfiles[i]);        
        print(i,": Converting sample",xml["experiment"]," to HDF5")
        esrf_to_hdf5(xml,dataroot)
        t1 = time();
        print(i,": Processing time was ",t1-t0,"s")
        
