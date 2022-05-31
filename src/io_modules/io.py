#!/usr/bin/env python3

def update_hdf5(filename,group,datasets,attributes,compression="lzf",chunksize=(64,64,64)):
    f = h5py.File(filename,'a')

    g = f.require_group(f)
    
    for k,v in datasets:
        g.require_dataset(k,shape=v.shape,dtype=v.dtype,compression=compression, chunksize=chunksize)
        g[k][:] = v[:]

    for k,v in attributes:
        g.attr(k) = v

    f.close()
