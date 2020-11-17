# Byte-per-voxel HDF5 files for complete multi-scan tomograms
# Format
# /subvolume_dimensions:  int(n,3).      For each of the n component scans, the sub-volume dimensions (nz,ny,nx)
# /subvolume_range:     float(n,2).      For each of the n component scane, the value range (vmin,vmax)
# /subvolume_metadata:  group            Attributes are info from ESRF XML-file describing original data
# /volume:              uint8(Nz,Ny,Nx). Nz = sum(scan_dimensions[:,0]), ny = minimum(subvolume_dimensions[:,1]), nx = minimum(subvolume_dimensions[:,2])
#import bohrium as bh
import numpy   as bh
import numpy   as np
import h5py
#import h5tomo 
import matplotlib.pyplot as plt
from esrf_read import *
import sys
import jax.numpy as jp
import jax
jax.config.update("jax_enable_x64", True)
NA = np.newaxis

sample     = sys.argv[1];
xml_root   = sys.argv[2];
hdf5_root  = sys.argv[3];


print(f"data_root={xml_root}")

def normalize(A,value_range,nbits=16,dtype=np.uint16):
    vmin,vmax = value_range
    return (((A-vmin)/(vmax-vmin))*(2**nbits-1)).astype(dtype)+1

subvolume_xmls     = readfile(f"{xml_root}/index/{sample}.txt")
subvolume_metadata = [esrf_read_xml(f"{xml_root}/{xml.strip()}") for xml in subvolume_xmls];

subvolume_dimensions = np.array([ (int(m['sizez']),  int(m['sizey']), int(m['sizex'])) for m in subvolume_metadata]);
subvolume_range      = np.array([(float(m['valmin']), float(m['valmax'])) for m in subvolume_metadata]);

global_vmin = np.min(subvolume_range[:,0])
global_vmax = np.max(subvolume_range[:,1])
(Nz,Ny,Nx)  = (np.sum(subvolume_dimensions[:,0]), np.min(subvolume_dimensions[:,1]), np.min(subvolume_dimensions[:,2]))

for i in range(len(subvolume_metadata)):
    print(f"{i} {sample}/{subvolume_metadata[i]['experiment']}: {subvolume_range[i]}")
print((global_vmin, global_vmax), (Nz,Ny,Nx))    
print(subvolume_dimensions)
print(subvolume_range)


#import re
#experiment_re = re.compile("_+([0-9a-zA-Z]+)_+(\d+)_pag$")
#re_match      = re.search(experiment_re, subvolume_metadata[0]['experiment'])
#assert(re_match)
#experiment = re_match.group(1)
experiment  = sample

#print(re_match.group(0))
#print(re_match.group(1))
#print(re_match.group(2))

msb_filename = f"{hdf5_root}/hdf5-byte/msb/1x/{experiment}.h5";
lsb_filename = f"{hdf5_root}/hdf5-byte/lsb/1x/{experiment}.h5";
print(f"Writing {msb_filename} and {lsb_filename}")
h5file_msb = h5py.File(msb_filename,"w");
h5file_lsb = h5py.File(lsb_filename,"w");

# Store metadata in both files for each subvolume scan
for h5file in [h5file_msb,h5file_lsb]:
    grp_meta = h5file.create_group("metadata");
    for i in range(len(subvolume_metadata)):
        subvolume_info = subvolume_metadata[i];
        grp_sub = grp_meta.create_group(f"subvolume{i}");
        for k in subvolume_info.keys():
            grp_sub.attrs[k] = np.string_(subvolume_info[k]);


    h5file.create_dataset("subvolume_dimensions",subvolume_dimensions.shape,dtype=np.uint16,data=subvolume_dimensions);
    h5file.create_dataset("subvolume_range",subvolume_range.shape,dtype=np.float32,data=subvolume_range);
    h5file.create_dataset("global_range",(2,),dtype=np.float32,data=np.array([global_vmin,global_vmax]));
    h5tomo     = h5file.create_dataset("voxels",(Nz,Ny,Nx),dtype=np.uint8,fletcher32=True,compression="lzf");

    h5tomo.dims[0].label = 'z';
    h5tomo.dims[1].label = 'y';
    h5tomo.dims[2].label = 'x';
    h5tomo.attrs['voxelsize'] = np.float(subvolume_info['voxelsize']);

z_offset = 0;
normalize_jit = jax.jit(normalize)
h5tomo_msb = h5file_msb['voxels']
h5tomo_lsb = h5file_lsb['voxels']

def cylinder_mask(Ny,Nx):
    ys = jp.linspace(-1,1,Ny)
    xs = jp.linspace(-1,1,Nx)
    return (xs[NA,:]**2 + ys[:,NA]**2) < 1 

mask = jp.array(cylinder_mask(Ny,Nx))

for i in range(len(subvolume_metadata)):
    subvolume_info = subvolume_metadata[i];
    (nz,ny,nx)     = subvolume_dimensions[i];
    (sy,sx)        = ((ny-Ny)//2+((ny-Ny)%2), (nx-Nx)//2+((nx-Nx)%2))
    (ey,ex)        = (ny-(ny-Ny)//2, nx-(nx-Nx)//2)
    print((sy,ey),(sx,ex))
    
    # print(f"Loading {subvolume_info['experiment']}")
    # tomo = normalize(esrf_full_tomogram_bh(subvolume_info), (global_vmin,global_vmax));
    # print(f"Writing {subvolume_info['experiment']}")    
    # h5tomo[z_offset:z_offset+nz] = tomo[:,sy:ey,sx:ex];
    # del tomo
    chunk_length = nz
    chunk = np.zeros((chunk_length,Ny,Nx),dtype=np.uint16);
    for z in range(0,nz,chunk_length):
        chunk_end = min(z+chunk_length,nz);
        print(f"Reading slice {z+z_offset}:{chunk_end+z_offset} ({i}-{z})");        
        for j in range(0,chunk_end-z):
#            slice_meta, slice_data = esrf_edf_n_to_bh(subvolume_info,z+j);
            slice_meta, slice_data = esrf_edf_n_to_npy(subvolume_info,z+j);
            slice_data = jp.array(slice_data[sy:ey,sx:ex].copy())
            chunk[j] = normalize_jit(slice_data,(global_vmin,global_vmax)) * ~mask[NA,:,:];
            del slice_data
            
        print(f"Writing {sample} MSB slice {z+z_offset}:{chunk_end+z_offset} ({i}-{z})");        
        h5tomo_msb[z_offset+z:z_offset+chunk_end] = ((chunk[:chunk_end-z]>>8)&0xff).astype(np.uint8) #.copy2numpy();
        print(f"Writing {sample} LSB slice {z+z_offset}:{chunk_end+z_offset} ({i}-{z})");        
        h5tomo_lsb[z_offset+z:z_offset+chunk_end] = (chunk[:chunk_end-z]&0xff).astype(np.uint8) #.copy2numpy();
    z_offset += nz;

h5file_msb.close()
h5file_lsb.close()
