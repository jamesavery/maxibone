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


def radial_histogram(tomo,nr,bin_edges):
    Rtot=tomo.shape[1]/2
    nbins = len(bin_edges)-1
    hists = np.empty((nr,nbins),dtype=float)
    for i in range(n):
        r = i*Rtot/nr;
        R = (i+1)*Rtot/nr
        iR = cart_to_polar(tomo.data,np.ceil(R-r), np.ceil(2*np.pi*R),r=r,R=R);
        hists[i] = np.histogram(iR,bins=bin_edges)[0];
    return hists


def progressive_radial_histogram(xml,nbins=2048,bin_edges=np.array([]),num_cores=4):
    
    if(len(bin_edges)==0):
        bin_edges = np.linspace(float(xml["valmin"]), float(xml["valmax"]), nbins + 1);
        nbins = len(bin_edges)-1;

    nz     = int(xml["sizez"]);
    meta,frame  = esrf_edf_n_to_npy(xml,0);
    frames = np.ma.empty((4*num_cores, frame.shape[0], frame.shape[1]));
    nr = frame.shape[1]//2;
    
    counts = np.zeros((nr,nbins),dtype=float);
    
    for i in range(0,nz,4*num_cores):
        chunk_length = min(4*num_cores,nz-i);
        for j in range(chunk_length):
            print("Reading data frame",i+j);
            _, frames[j] = esrf_edf_n_to_npy(xml,i+j);
        counts += radial_histogram(frames[:chunk_length],nr,bin_edges)
        
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

    hcounts,bin_edges = progressive_radial_histogram(xml,nbins=nbins,bin_edges=bin_edges,num_cores=num_cores);

    print("Storing result")
    filename = os.path.splitext(os.path.basename(xmlfiles[i]))[0]+'.npz';
    np.savez_compressed(dataroot+"/radial-histograms/"+filename,hcounts=hcounts,bin_edges=bin_edges,dataroot=dataroot,xmlfile=xmlfiles[i]);

