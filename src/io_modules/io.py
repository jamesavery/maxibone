#!/usr/bin/env python3

def update_hdf5(filename,datasets,attributes,compression="lzf",chunksize=(64,64,64)):
    f = h5py.File(filename,'a')

    for k,v in datasets:
        f.require_dataset(k,shape=v.shape,dtype=v.dtype,compression=compression, chunksize=chunksize)
        f[k][:] = v[:]

    for k,v in attributes:
        f.attr(k) = v

    f.close()
