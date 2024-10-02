#! /usr/bin/python3
'''
Byte-per-voxel HDF5 files for complete multi-scan tomograms.

Format:
- /subvolume_dimensions: int(n,3).         For each of the n component scans, the sub-volume dimensions (nz,ny,nx)
- /subvolume_range:      float(n,2).       For each of the n component scane, the value range (vmin,vmax)
- /subvolume_metadata:   group             Attributes are info from ESRF XML-file describing original data
- /voxels:               uint8(Nz,Ny,Nx).  Nz = sum(scan_dimensions[:,0]), ny = minimum(subvolume_dimensions[:,1]), nx = minimum(subvolume_dimensions[:,2])
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.paths import hdf5_root as hdf5_root, esrf_implants_root
import h5py
from lib.py.commandline_args import default_parser
from lib.py.esrf_read import *
from lib.py.helpers import generate_cylinder_mask, normalize
import numpy as np
import tqdm

NA = np.newaxis

if __name__ == "__main__":
    argparser = default_parser(description=__doc__)
    argparser.add_argument('--use_bohrium', action='store_true',
        help='Toggles whether to use Bohrium for processing.')
    argparser.add_argument('--xml_root', action='store', type=str, default=esrf_implants_root,
        help='The root directory of the ESRF 2013 dataset.')
    args = argparser.parse_args()

    if args.verbose >= 1: print(f"data_root={args.xml_root}")

    subvolume_xmls       = readfile(f"{args.xml_root}/index/{args.sample}.txt")
    subvolume_metadata   = [esrf_read_xml(f"{args.xml_root}/{xml.strip()}") for xml in subvolume_xmls]
    subvolume_dimensions = np.array([(int(m['sizez']), int(m['sizey']), int(m['sizex'])) for m in subvolume_metadata])
    subvolume_range      = np.array([(float(m['valmin']), float(m['valmax'])) for m in subvolume_metadata])

    global_vmin = np.min(subvolume_range[:,0])
    global_vmax = np.max(subvolume_range[:,1])
    # TODO Should we also enforce Nz % 32 == 0? Problem:
    # 1) volume matching will ruin it anyway
    # 2) top or bottom can have important info (depending on orientation of scan)
    (Nz,Ny,Nx)  = (np.sum(subvolume_dimensions[:,0]), np.min(subvolume_dimensions[:,1]&~31), np.min(subvolume_dimensions[:,2]&~31))

    if args.verbose >= 1:
        for i in range(len(subvolume_metadata)):
            print(f"{i} {args.sample}/{subvolume_metadata[i]['experiment']}: {subvolume_range[i]}")
        print((global_vmin, global_vmax), (Nz,Ny,Nx))
        print(subvolume_dimensions)
        print(subvolume_range)

    msb_filename = f"{hdf5_root}/hdf5-byte/msb/{args.sample}.h5"
    lsb_filename = f"{hdf5_root}/hdf5-byte/lsb/{args.sample}.h5"

    # Make sure directory exists
    outdir = os.path.dirname(msb_filename)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    outdir = os.path.dirname(lsb_filename)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    if args.verbose >= 1: print(f"Writing {msb_filename} and {lsb_filename}")
    h5file_msb = h5py.File(msb_filename,"w")
    h5file_lsb = h5py.File(lsb_filename,"w")

    # Store metadata in both files for each subvolume scan
    for h5file in [h5file_msb,h5file_lsb]:
        grp_meta = h5file.create_group("metadata")
        for i in range(len(subvolume_metadata)):
            subvolume_info = subvolume_metadata[i]
            grp_sub = grp_meta.create_group(f"subvolume{i}")
            for k in subvolume_info.keys():
                grp_sub.attrs[k] = np.string_(subvolume_info[k])

        h5file.create_dataset("subvolume_dimensions", subvolume_dimensions.shape, dtype=np.uint16, data=subvolume_dimensions)
        h5file.create_dataset("subvolume_range", subvolume_range.shape, dtype=np.float32, data=subvolume_range)
        h5file.create_dataset("global_range", (2, ), dtype=np.float32, data=np.array([global_vmin, global_vmax]))
        h5tomo = h5file.create_dataset("voxels", (Nz,Ny,Nx), dtype=np.uint8, fletcher32=True,  compression="lzf" if h5file==h5file_msb else None)

        h5tomo.dims[0].label = 'z'
        h5tomo.dims[1].label = 'y'
        h5tomo.dims[2].label = 'x'
        h5tomo.attrs['voxelsize'] = float(subvolume_info['voxelsize'])

    z_offset = 0
    h5tomo_msb = h5file_msb['voxels']
    h5tomo_lsb = h5file_lsb['voxels']
    mask = np.array(generate_cylinder_mask(Ny, Nx))

    for i in tqdm.tqdm(range(len(subvolume_metadata))):
        subvolume_info = subvolume_metadata[i]
        (nz, ny, nx)   = subvolume_dimensions[i]
        (sy, sx)       = ((ny-Ny)//2 + ((ny-Ny)%2), (nx-Nx)//2 + ((nx-Nx)%2))
        (ey, ex)       = (ny-(ny-Ny)//2, nx-(nx-Nx)//2)
        if args.verbose >= 1: print((sy,ey), (sx,ex))

        chunk = np.zeros((args.chunk_size, Ny, Nx), dtype=np.uint16)
        for z in range(0, nz, args.chunk_size):
            chunk_end = min(z+args.chunk_size, nz)

            region = [[sx, sy, z], [ex, ey, chunk_end]]
            if args.verbose >= 1: print(f"Reading chunk {z+z_offset}:{chunk_end+z_offset} ({i}-{z}), region={region}")

            slab_data = esrf_edfrange_to_bh(subvolume_info,region)
            if args.verbose >= 1: print(f"Chunk shape: {slab_data.shape}")
            if args.verbose >= 1: print("Max value before masking:", slab_data.max())

            slab_data *= mask[NA,:,:]
            if args.verbose >= 1: print("Max value after masking:", slab_data.max())

            chunk[:chunk_end-z] = normalize(slab_data, (global_vmin,global_vmax))
            if args.verbose >= 1: print("Max value after normalizing:", chunk.max())

            chunk_msb = ((chunk[:chunk_end-z] >> 8) & 0xff).astype(np.uint8)
            chunk_lsb = ( chunk[:chunk_end-z]       & 0xff).astype(np.uint8)

            if args.verbose >= 1:
                print(f"Writing {args.sample} MSB slice {z+z_offset}:{chunk_end+z_offset} ({i}-{z})")
                print("chunk_msb.max: ", chunk_msb.max())
                print("chunk_msb.copy2numpy().max: ", chunk_msb.max())
            h5tomo_msb[z_offset+z:z_offset+chunk_end] = chunk_msb[:]

            if args.verbose >= 1:
                print(f"Writing {args.sample} LSB slice {z+z_offset}:{chunk_end+z_offset} ({i}-{z})")
                print("chunk_lsb.max: ", chunk_lsb.max())
                print("chunk_lsb.copy2numpy().max: ", chunk_lsb.max())
            h5tomo_lsb[z_offset+z:z_offset+chunk_end] = chunk_lsb[:]

        z_offset += nz

    h5file_msb.close()
    h5file_lsb.close()
