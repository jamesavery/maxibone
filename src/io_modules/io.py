#!/usr/bin/env python3
import h5py

def update_hdf5(filename,group_name,datasets,attributes,dimensions=None,
                compression="lzf",chunk_shape=(64,64,64)):

    f = h5py.File(filename,'a')

    g = f.require_group(group_name)
    
    for k in datasets:
        v = datasets[k]
        g.require_dataset(k,shape=v.shape,dtype=v.dtype,
                          compression=compression, chunks=chunk_shape)
        g[k][:] = v[:]

        if dimensions is not None:
            try:
                dims = dimensions[k]
                for i, description in enumerate(dims):
                    g[k].dims[i] = description
            except:
                pass

    for k in attributes:
        v = attributes[k]
        g.attrs[k] = v

    f.close()
