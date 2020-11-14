import numpy as np
import h5py as h5

#TODO: 3D blocks for greater volume-to-boundary ratio
#TODO: Parallelize
#TODO: Is overlap and merging needed?
def blockmap(fun, args, datain, dataout, chunk_size, boundary_size):
    nz = datain.shape[0];
    n_chunks        = nz // chunk_size;
    chunk_remainder = nz % chunk_size;

    # First and last chunks are special
    dataout[:chunk_size]       = fun(datain[:chunk_size+boundary_size], *args)[:chunk_size]; 
    dataout[-chunk_remainder:] = fun(datain[-chunk_remainder-boundary_size:], *args)[-chunk_remainder:];

    # The middle ones all have a top- and bottom-boundary that is discarded, and a middle good part
    for i in range(1,n_chunks):
        out = fun(datain[i*chunk_size-boundary_size:(i+1)*chunk_size+boundary_size],*args)
        dataout[i*chunk_size:(i+1)*chunk_size] = out[boundary_size:chunk_size+boundary_size]



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xs = np.linspace(0,2*np.pi,10000)
    ys = np.empty(10000)

    blockmap(np.sin,[],xs,ys,300,50)
    
    plt.plot(xs,ys)
