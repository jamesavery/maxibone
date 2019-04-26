import h5py
import bohrium as bh
import numpy as np
from time import time
from uk_ndi import flat_nonzero, sparse_label

def unpackbits(packed,shape):
    n_valid = shape[0]*shape[1];
    return bh.unpackbits(packed)[:n_valid].reshape(shape);

def values(Fi,tomo):
    (mn,mx) = tomo['range'];
    return Fi*(mx-mn+1)/65535.0+mn

def load_sparse_filter(h5file,predicate, chunk_size=10):
    h5meta = h5file["metadata"];
    meta   = {'range': [float(h5meta.attrs["valmin"]), float(h5meta.attrs["valmax"])]};

    vol = h5file['subvolume'];    
    voxels_hi   = vol['voxels_hi'];
    voxels_lo   = vol['voxels_lo'];    
    (nz,ny,nx)= voxels_hi.shape;

    voxel_chunk = bh.empty((chunk_size,ny,nx),dtype=bh.uint16);

    Ts = [];
    for z in range(0,nz,chunk_size):
        chunk_length = min(chunk_size,nz-z);
        voxel_chunk [:chunk_length] = (voxels_hi[nz:nz+chunk_length].astype(bh.uint16)<<8);
        voxel_chunk[:chunk_length] |= voxels_lo[nz:nz+chunk_length];

        Ts.append( sparse_label(flat_nonzero(predicate(values(voxels_chunk))), voxel_chunk.shape) )

    return Ts;
    
def load_volume(h5file):

    t0 = time()
    meta = h5file["metadata"];
    valmin = float(meta.attrs["valmin"])
    valmax = float(meta.attrs["valmax"])
    voxelsize=float(meta.attrs["voxelsize"])

    vol = h5file['subvolume'];    
    voxels_hi   = vol['voxels_hi'];
    voxels_lo   = vol['voxels_lo'];    
    (nz,ny,nx)= voxels_hi.shape;
    
    voxels    = bh.empty((nz,ny,nx),dtype=bh.uint16);
    t1 = time()
    # NB: Change back if mask is not constant
    # print("Reading mask")    
    # packed_mask = vol['mask'][:];
    # t2 = time()
    # print("Unpacking mask")
    # mask        = bh.array([unpackbits(packed_mask[z],(ny,nx)) for z in range(nz)]);
    print("Reading mask")    
    packed_mask = vol['mask'][0];
    t2 = time()
    print("Unpacking mask")
    mask        = unpackbits(packed_mask,(ny,nx))
    t3 = time()
    
    print("Reading voxel most significant bits");    
    voxels[:]  = (voxels_hi[:].astype(bh.uint16)<<8);
    t4 = time()
    print("Reading voxel least significant bits");    
    voxels[:] |= voxels_lo[:]
    t5 = time();
    print("Total time: ",t5-t0,"; break-down: open(",t1-t0,"), read-mask(",t2-t1,"), unpack-mask(",t3-t2,"), read-voxels(",t4-t3,t5-t4,")");
    
    return {'voxels':voxels,
            'mask':  mask,
            'range': [valmin,valmax],
            'voxelsize': voxelsize
    };

def load_volume_region(h5file,region):

    meta = h5file["metadata"];
    valmin = float(meta.attrs["valmin"])
    valmax = float(meta.attrs["valmax"])
    voxelsize=float(meta.attrs["voxelsize"])

    vol         = h5file['subvolume'];
    voxels_hi   = vol['voxels_hi'];
    voxels_lo   = vol['voxels_lo'];    
    (nz,ny,nx)= voxels_hi.shape;
    ((sz,sy,sx),(ez,ey,ex));
    
    voxels = bh.empty((ez-sz,ey-sy,ex-sx),dtype=bh.uint16);
    packed_mask = vol['mask'][sz:ez];
    mask        = bh.array([unpackbits(packed_mask[z],(ny,nx))[sy:ey,sx:ex] for z in range(ez-sz)])
    
    voxels[:]  = voxels_lo[sz:ez,sy:ey,sx:ex];    
    voxels[:] |= (voxels_hi[sz:ez,sy:ey,sx:ex].astype(bh.uint16)<<8);    

    return {'voxels':voxels,
            'mask':  mask,
            'range': [valmin,valmax],
            'voxelsize': voxelsize
    };
    


def esrf_to_hdf5(info,output_dir,chunk_size=64):
    # Mirror the metadata from the ESRF XML file
    filename = output_dir+"/hdf5/"+info['experiment']+".h5";
#    os.unlink(filename)    
    f = h5py.File(filename,'w')    

    grp_meta = f.create_group("metadata");
    for k in info.keys():
        grp_meta.attrs[k] = np.string_(info[k]);

    grp_vol  = f.create_group("subvolume");
    grp_seg  = f.create_group("segments");
    
    (nx,ny,nz) = tuple([int(xml["size"+x]) for x in ["x","y","z"]]);
#    nz = 64;

    volhi = grp_vol.create_dataset("voxels_hi", (nz,ny,nx), dtype=np.uint8,fletcher32=True,compression="lzf");
    vollo = grp_vol.create_dataset("voxels_lo", (nz,ny,nx), dtype=np.uint8,fletcher32=True,compression="lzf");
    mask  = grp_vol.create_dataset("mask",(nz,int(np.ceil(ny*nx/8))), dtype=np.uint8,fletcher32=True,compression="lzf");

#    for i in [0,1,2]:
#        volume.dims[i].label = "zyx"[i];
    
    vmin, vmax = float(info["valmin"]), float(info["valmax"]);
    def normalize(A):
        return (A-vmin)/(vmax-vmin+1);

    # Load slices and fill out data
    frames = np.ma.empty((chunk_size, ny, nx),dtype=np.float32);
    for z in range(0,nz,chunk_size):
        chunk_length = min(chunk_size,nz-z);
        print("Reading Frame",z)
        frames[:chunk_length] = np.array([esrf_edf_n_to_npy(info,z+i)[1] for i in range(chunk_length)]);
 
        print("Writing Mask",z);
        mask[z:z+chunk_length] = np.array([np.packbits(frames[i]==0) for i in range(chunk_length)]);

        print("Writing Voxels",z);        
        normalized = normalize(frames[:chunk_length]);
        volume_u16 = (normalized*65535).astype(np.uint16);
        vollo[z:z+chunk_length] =  volume_u16       & 0xff;                
        volhi[z:z+chunk_length] = (volume_u16 >> 8) & 0xff;
    f.close()     

