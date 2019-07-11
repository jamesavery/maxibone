import numpy as np
import scipy.ndimage as nd
from clustering import * 
from distributions import gaussians, powers
from esrf_read import *
from bitmaps import *

# For generating a figure of the result
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


dataroot="/home/avery/nobackup/maxibone-data/"
sample = "HA_xc520_50kev_1_88mu_implant_770c_002_pag";
#sample_disk = "disk1/esrf_april_2013_implant_770/vol_float/";
sample_disk = "770c_data/"
sample_info = esrf_read_xml(dataroot+sample_disk+sample+"/"+sample+".xml");


# ******** HISTOGRAM ANALYSIS ********
# TODO: Adapt analysis to variation over y, z, r to produce conditional probabilities P(c|I,z), P(c|I,y), P(c|I,r)
# TODO: Combine conditional probabilities to improve class prediction P(c|I,r,y,z) (on-the-fly, don't compute 4D prob.)

# Read the data
z_histogram_file        = np.load(dataroot+"/z-histograms/"+sample+".npz")
y_histogram_file        = np.load(dataroot+"/y-histograms/"+sample+".npz")
radial_histogram_file   = np.load(dataroot+"/radial-histograms/"+sample+".npz")

z_hist = z_histogram_file["hcounts"]
z_bins = z_histogram_file["bin_edges"]

y_hist = y_histogram_file["hcounts"]
y_bins = y_histogram_file["bin_edges"]

r_hist = radial_histogram_file["hcounts"]
r_bins = radial_histogram_file["bin_edges"]

values = (r_bins[1:]+r_bins[:-1])/2


# Function definitions
def get_air(values,y_hist):
    # TODO: Automatically detect region with only air        
    hist_air   = np.sum(y_hist[:1100],axis=0);
    hist_sum   = np.sum(y_hist,axis=0);
    
    region = bit_and([values>-2,values<=0])
    values_air = values[region]    
    hist_air[np.logical_not(region)] = 0
    proportion = hist_air[region].max()/(hist_sum[region].max())    
    hist_air  /= proportion
    
    abcd_air = distributions_from_clusters(values_air,hist_air[region],1,distribution_function=powers, overshoot_penalty=1)
    
    return (hist_air, abcd_air)
    
def get_implant(values,y_hist):
    # TODO: Automatically detect region with only implant            
    hist_implant = np.sum(y_hist[1100:2100],axis=0)
    hist_sum     = np.sum(y_hist,axis=0)
        
    region         = bit_and([values>1,values<8])
    values_implant = values[region]
    hist_implant[np.logical_not(region)] = 0    
    
    proportion     = hist_implant[region].max()/hist_sum[region].max()
    hist_implant  /= proportion
    
    abcd_implant = distributions_from_clusters(values_implant,hist_implant[region],1,distribution_function=powers, overshoot_penalty=0.5)
    
    return (hist_implant, abcd_implant)


# Do the calculation
# K-Means
#hist_sum = np.sum(y_hist[1500:2800],axis=0)
hist_sum = np.sum(y_hist,axis=0)

# Get air distribution directly
(hist_air,abcd_air) = get_air(values,y_hist)
g_air = powers(values,abcd_air)[0]

# Get air distribution directly
(hist_implant,abcd_implant) = get_implant(values,y_hist)
g_implant = powers(values,abcd_implant)[0]
hist_opt = np.maximum(hist_sum - g_air - hist_implant, np.zeros(hist_sum.shape))

n_clusters = 2
# Optimize
abcd = distributions_from_clusters(values,hist_opt,
                                   n_clusters,
                                   overshoot_penalty=5,
                                   distribution_function=powers)

# TODO: Clean up
A = abcd[           0:  n_clusters]
B = abcd[  n_clusters:2*n_clusters]
C = abcd[2*n_clusters:3*n_clusters]
D = abcd[3*n_clusters:4*n_clusters]

# Hack - merge more nicely

A = np.array([abcd_air[0], A[0], A[1], abcd_implant[0]])
B = np.array([abcd_air[1], B[0], B[1], abcd_implant[1]])
C = np.array([abcd_air[2], C[0], C[1], abcd_implant[2]])
D = np.array([abcd_air[3], D[0], D[1], abcd_implant[3]])

abcd = np.concatenate([A,B,C,D])

rhos=powers(values,abcd)
rho_tot = np.sum(rhos,axis=0)

Pc=np.minimum(rhos/(hist_sum+(hist_sum==0)),np.ones(hist_sum.shape))
Prest=1-np.sum(Pc,axis=0)

np.savez(dataroot+"/classification/absorption/"+sample+".npz",distribution_type='powers',abcd=abcd,rhos=rhos,Pc=Pc,Prest=Prest)

#Also generate the graphics, to see how well it works?

colors = ['red','green','blue','yellow','orange','purple'];
n_clusters = len(rhos)
region = bit_and([values>-1,values<=6])
plt.figure(figsize=(30,5))
plt.plot(values[region],hist_sum[region])
plt.plot(values[region],rho_tot[region])
#plt.plot(values[region],g_mid[region])

#plt.plot(values[region],g_mid[region])
for i in range(n_clusters):
    plt.plot(values[region],rhos[i][region],color=colors[i])
    plt.axvline(x=C[i],color=colors[i])
plt.savefig(dataroot+"/classification/absorption/"+sample+"-match.png")
