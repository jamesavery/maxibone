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

from resample import *;
from esrf_read import *;
import glob;
import sys;

#num_cores = multiprocessing.cpu_count();
num_cores = 4;

dataroot = sys.argv[1];
#TODO: Less greasy
nbins=2048
bin_edges = np.linspace(-4, 12, nbins+1);


def hist_count(im,bin_edges):
    return np.histogram(im,bins=bin_edges)[0];

def y_histogram(tomo,bin_edges):
    nbins = len(bin_edges)-1
    return np.array(
        Parallel(n_jobs=num_cores)(delayed(hist_count)(tomo[:,i,:].compressed(),bin_edges) for i in range(tomo.shape[1]))
    );


def progressive_y_histogram(xml,nbins=2048,bin_edges=np.array([]),num_cores=4):
    
    if(len(bin_edges)==0):
        bin_edges = np.linspace(float(xml["valmin"]), float(xml["valmax"]), nbins + 1);
        nbins = len(bin_edges)-1;

    nz     = int(xml["sizez"]);
    meta,frame  = esrf_edf_n_to_npy(xml,0);
    frames = np.ma.empty((4*num_cores, frame.shape[0], frame.shape[1]));
    ny = frame.shape[1];
    
    counts = np.zeros((ny,nbins),dtype=float);
    
    for i in range(0,nz,4*num_cores):
        chunk_length = min(4*num_cores,nz-i);
        for j in range(chunk_length):
            print("Reading data frame",i+j);
            _, frames[j] = esrf_edf_n_to_npy(xml,i+j);
        counts += y_histogram(frames[:chunk_length],bin_edges)
        
    return counts, bin_edges;


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

    hcounts,bin_edges = progressive_y_histogram(xml,nbins=nbins,bin_edges=bin_edges,num_cores=num_cores);

    print("Storing result")
    filename = os.path.splitext(os.path.basename(xmlfiles[i]))[0]+'.npz';
    np.savez_compressed(dataroot+"/y-histograms/"+filename,hcounts=hcounts,bin_edges=bin_edges,dataroot=dataroot,xmlfile=xmlfiles[i]);

