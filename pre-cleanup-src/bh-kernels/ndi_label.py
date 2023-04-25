def ndi_label(nonzeros,shape,dtype=bh.uint32):
    L = bh.zeros(shape,dtype=dtype);
    T = bh.arange(len(nonzeros)+1); # Initialize union-find equialence tree to "all separate"
    strides = bh.array(L.strides,dtype=bh.uint64);
    L[nonzeros] = nonzeros+1;

    kernel = read_kernel("ndi_label") % {
        'ndim': L.ndim,
        'total_size':L.size,
        'n_nonzeros':len(nonzeros),
        'label_t':ctype[L.dtype.name],
        'index_t':ctype[nonzeros.dtype.name]
        };

    uk.execute(kernel,[L,T,strides,nonzeros])

    return L,T
