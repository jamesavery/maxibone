#!/usr/bin/env python3
import h5py

def update_hdf5(filename,group_name,datasets,attributes,compression="lzf",chunksize=(64,64,64)):
    f = h5py.File(filename,'a')

    g = f.require_group(group_name)
    
    for k in datasets:
        v = datasets[k]
        g.require_dataset(k,shape=v.shape,dtype=v.dtype,compression=compression, chunks=chunksize)
        g[k][:] = v[:]

    for k in attributes:
        v = attributes[k]
        g.attrs[k] = v

    f.close()
