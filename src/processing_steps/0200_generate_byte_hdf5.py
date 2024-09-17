# Byte-per-voxel HDF5 files for complete multi-scan tomograms
# Format
# /subvolume_dimensions:  int(n,3).      For each of the n component scans, the sub-volume dimensions (nz,ny,nx)
# /subvolume_range:     float(n,2).      For each of the n component scane, the value range (vmin,vmax)
# /subvolume_metadata:  group            Attributes are info from ESRF XML-file describing original data
# /voxels:              uint8(Nz,Ny,Nx). Nz = sum(scan_dimensions[:,0]), ny = minimum(subvolume_dimensions[:,1]), nx = minimum(subvolume_dimensions[:,2])
import h5py, sys, os.path, pathlib, tqdm
sys.path.append(sys.path[0]+"/../")
#import bohrium as bh # TODO: Get rid of Bohrium dependence without losing too much performance
from lib.py.esrf_read import *
import numpy   as np, matplotlib.pyplot as plt
from config.paths import hdf5_root as hdf5_root, esrf_implants_root
from lib.py.helpers import commandline_args, generate_cylinder_mask, normalize

NA = np.newaxis

sample, chunk_length, use_bohrium, xml_root, verbose  = commandline_args({"sample" : "<required>",
                                                                          "chunk_length" : 256,
                                                                          "use_bohrium" : True,
                                                                          "xml_root" : esrf_implants_root,
                                                                          "verbose" : 1})


if verbose >= 1: print(f"data_root={xml_root}")

    subvolume_xmls       = readfile(f"{xml_root}/index/{sample}.txt")

global_vmin = np.min(subvolume_range[:,0])
global_vmax = np.max(subvolume_range[:,1])
#TODO: Should we also enforce Nz % 32 == 0? Problem: 1) volume matching will ruin it anyway, and
#                                                    2) top or bottom can have important info (depending on orientation of scan)
(Nz,Ny,Nx)  = (np.sum(subvolume_dimensions[:,0]), np.min(subvolume_dimensions[:,1]&~31), np.min(subvolume_dimensions[:,2]&~31))

for i in range(len(subvolume_metadata)):
    if verbose >= 1: print(f"{i} {sample}/{subvolume_metadata[i]['experiment']}: {subvolume_range[i]}")
if verbose >= 1: print((global_vmin, global_vmax), (Nz,Ny,Nx))
if verbose >= 1: print(subvolume_dimensions)
if verbose >= 1: print(subvolume_range)


#import re
#sample_re = re.compile("_+([0-9a-zA-Z]+)_+(\d+)_pag$")
#re_match      = re.search(sample_re, subvolume_metadata[0]['experiment'])
#assert(re_match)
#sample = re_match.group(1)
sample  = sample

#print(re_match.group(0))
#print(re_match.group(1))
#print(re_match.group(2))

msb_filename = f"{hdf5_root}/hdf5-byte/msb/{sample}.h5";
lsb_filename = f"{hdf5_root}/hdf5-byte/lsb/{sample}.h5";

# Make sure directory exists
outdir = os.path.dirname(msb_filename)
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
outdir = os.path.dirname(lsb_filename)
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

if verbose >= 1: print(f"Writing {msb_filename} and {lsb_filename}")
h5file_msb = h5py.File(msb_filename,"w");
h5file_lsb = h5py.File(lsb_filename,"w");

    mask = np.array(generate_cylinder_mask(Ny, Nx))

for i in tqdm.tqdm(range(len(subvolume_metadata))):
    subvolume_info = subvolume_metadata[i];
    (nz,ny,nx)     = subvolume_dimensions[i];
    (sy,sx)        = ((ny-Ny)//2+((ny-Ny)%2), (nx-Nx)//2+((nx-Nx)%2))
    (ey,ex)        = (ny-(ny-Ny)//2, nx-(nx-Nx)//2)
    if verbose >= 1: print((sy,ey),(sx,ex))

    # if verbose >= 1: print(f"Loading {subvolume_info['experiment']}")
    # tomo = normalize(esrf_full_tomogram_bh(subvolume_info), (global_vmin,global_vmax));
    # if verbose >= 1: print(f"Writing {subvolume_info['experiment']}")
    # h5tomo[z_offset:z_offset+nz] = tomo[:,sy:ey,sx:ex];
    # del tomo
    chunk = np.zeros((chunk_length,Ny,Nx),dtype=np.uint16);
    for z in range(0,nz,chunk_length):
        chunk_end = min(z+chunk_length,nz);

        region = [[sx,sy,z],[ex,ey,chunk_end]]
        if verbose >= 1: print(f"Reading chunk {z+z_offset}:{chunk_end+z_offset} ({i}-{z}), region={region}");
        slab_data = esrf_edfrange_to_bh(subvolume_info,region)
        if verbose >= 1: print(f"Chunk shape: {slab_data.shape}")
        if verbose >= 1: print("Max value before masking:", slab_data.max())
        slab_data *= mask[NA,:,:]
        if verbose >= 1: print("Max value after masking:", slab_data.max())
        chunk[:chunk_end-z] = normalize(slab_data,(global_vmin,global_vmax))
        if verbose >= 1: print("Max value after normalizing:", chunk.max())

        # for j in range(0,chunk_end-z):
        #     slice_meta, slice_data = esrf_edf_n_to_npy(subvolume_info,z+j);
        #     slice_data = jp.array(slice_data[sy:ey,sx:ex].copy())
        #     chunk[j] = normalize(slice_data[sy:ey,sx:ex],(global_vmin,global_vmax)) * mask


        if verbose >= 1: print(f"Writing {sample} MSB slice {z+z_offset}:{chunk_end+z_offset} ({i}-{z})");
        chunk_msb = ((chunk[:chunk_end-z]>>8)&0xff).astype(np.uint8)
        if verbose >= 1: print("chunk_msb.max: ", chunk_msb.max())
        #chunk_msb = chunk_msb.copy2numpy()
        if verbose >= 1: print("chunk_msb.copy2numpy().max: ", chunk_msb.max())
        h5tomo_msb[z_offset+z:z_offset+chunk_end] = chunk_msb[:]

        if verbose >= 1: print(f"Writing {sample} LSB slice {z+z_offset}:{chunk_end+z_offset} ({i}-{z})");
        chunk_lsb = (chunk[:chunk_end-z]&0xff).astype(np.uint8)
        if verbose >= 1: print("chunk_lsb.max: ", chunk_lsb.max())
        #chunk_lsb = chunk_lsb.copy2numpy()
        if verbose >= 1: print("chunk_lsb.copy2numpy().max: ", chunk_lsb.max())
        h5tomo_lsb[z_offset+z:z_offset+chunk_end] = chunk_lsb[:]
        #np.flush()

    z_offset += nz;

h5file_msb.close()
h5file_lsb.close()
