from skimage.transform import rescale;
from esrf_read import *;
import numpy as np;
import glob;
import sys;
import h5py;

dataroot  = sys.argv[1];
chunk_max = 256
scales    = [2,3,4,5,6,7,8]

with open(dataroot+"/xmlfiles.txt") as f:
    xmlfiles = [x.rstrip('\n') for x in f.readlines()]

def process_histogram(xmlfile,chunk_max,scales):
    xml         = esrf_read_xml(dataroot+xmlfile)    
    filename    = os.path.splitext(os.path.basename(xmlfile))[0];

    nz,ny,nx    = int(xml["sizez"]), int(xml["sizey"]), int(xml["sizex"]);    
    meta,frame  = esrf_edf_n_to_npy(xml,0);
    frames      = np.ma.empty((chunk_max, ny, nx));

    output_files   = [];
    output_z       = np.zeros(len(scales),dtype=int)
    shape          = np.array([nz,ny,nx])

    print("chunk_shape = ",frames.shape)
    
# Create target files
    for s in scales:
        s_shape = np.round(shape/s).astype(int);

        # Create the file and allocate memory for buffer
        f = h5py.File("%s/downscaled/%s-%d.h5" % (dataroot,filename,s),"w");
        output_files.append(f)

        # Create HDF5 metadata
        grp_meta = f.create_group("metadata");
        for k in xml.keys():
            grp_meta.attrs[k] = np.string_(xml[k]);
        grp_meta.attrs["scale"] = np.string_(str(s))

        grp_vol = f.create_group("subvolume");
        print("s_shape = ",s_shape)
        voxels  = grp_vol.create_dataset("voxels", s_shape, dtype=np.float32,fletcher32=True,compression="lzf");

# Read and scale data            
    for i in range(0,nz,chunk_max):
        chunk_length = min(chunk_max,nz-i);
        print("Reading data frames %d to %d..." % (i,i+chunk_max));
        for j in range(chunk_length):
            _, frames[j] = esrf_edf_n_to_npy(xml,i+j);

        print("Rescaling...")

        for i in range(len(scales)):
            s = scales[i]            
            print("1/",s)
            s_chunk_length = int(np.round(chunk_length/s))
            voxels = output_files[i]["subvolume"]["voxels"];

            scaled_chunk = rescale(frames[:chunk_length],1/s,multichannel=False,anti_aliasing=True)
            print("scaled_chunk: ",scaled_chunk.shape)
            voxels[output_z[i]:output_z[i]+s_chunk_length] = scaled_chunk
            output_z[i] += s_chunk_length

# Close the output files
        for i in range(len(scales)):
            output_files[i].close()

#TODO Overlap chunks
   
    
for i in range(len(xmlfiles)):
    process_histogram(xmlfiles[i],chunk_max,scales)
