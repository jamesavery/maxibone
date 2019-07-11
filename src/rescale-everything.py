from skimage.transform import rescale;
from esrf_read import *;
import numpy as np;
import glob;
import sys;

dataroot = sys.argv[1];
scales = [3,4,5,6,7,8];

with open(dataroot+"/xmlfiles-pag.txt") as f:
    xmlfiles = [x.rstrip('\n') for x in f.readlines()]

# Restart with nr. 40
xmlfiles = xmlfiles[39:]

for i in range(len(xmlfiles)):
    print("\n\n",i,xmlfiles[i]);

    xml      = esrf_read_xml(dataroot+"/"+xmlfiles[i]);
    filename = os.path.splitext(os.path.basename(xmlfiles[i]))[0];

    tomo       = esrf_full_tomogram(xml)
    vmin, vmax = tomo.min(), tomo.max()
    
    for s in scales:
        print("Scaling down by factor ",s)
        tomo_scaled = rescale(tomo.data,1/s,multichannel=False,anti_aliasing=True)
        np.savez(dataroot+"/downscaled/%s-%d.npz" % (filename,s),tomo=tomo_scaled,vmin=vmin,vmax=vmax,dataroot=dataroot,xmlfile=xmlfiles[i])
    
