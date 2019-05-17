import numpy as np;
import numpy.ma as ma;
import matplotlib.pyplot as plt;
from scipy import ndimage as ndi;
import scipy;
import skimage.feature as feat;
import skimage.morphology as morf;
import skimage.filters as filt;
import skimage.draw as drameasurew
import skimage.measure as measure
from joblib import Parallel, delayed
import multiprocessing;

from esrf_read import *;
import glob;
import sys;

num_cores = multiprocessing.cpu_count();

dataroot = sys.argv[1];
#TODO: Less greasy
nbins=2048
bin_edges = np.linspace(-4, 12, nbins+1);

with open(dataroot+"/xmlfiles.txt") as f:
    xmlfiles = [x.rstrip('\n') for x in f.readlines()]

# def frame_histogram(xml,i,bin_edges):
#     print("Reading data frame",i)                
#     meta,frame  = esrf_edf_n_to_npy(xml,i);
#     print("Calculating histogram for frame",i)        
#     return np.histogram(frame.compressed(),bins=bin_edges)[0];            

for i in range(len(xmlfiles)):
    print("\n\n",i,xmlfiles[i]);

    xml = esrf_read_xml(dataroot+"/"+xmlfiles[i]);

    hcounts,bin_edges = progressive_histogram(xml,nbins=nbins,bin_edges=bin_edges,num_cores=num_cores);

    print("Storing result")
    filename = os.path.splitext(os.path.basename(xmlfiles[i]))[0]+'.npz';
    np.savez_compressed(dataroot+"/histograms/"+filename,hcounts=hcounts,bin_edges=bin_edges,dataroot=dataroot,xmlfile=xmlfiles[i]);

