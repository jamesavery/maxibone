import bohrium as bh;
import bohrium.user_kernel as uk;
import numpy as np;
import scipy.sparse as sp;

def uk_reduceat(A,segments,axis=0,operation="+",kernel_preamble=""):
    (Am,Ai,An) = axis_split(A,axis);
    
    values       = uk.make_behaving(A);
    segments     = uk.make_behaving(segments);
    num_segments = len(segments);

    O = bh.empty((Am,num_segments,An),dtype=A.dtype);

    kernel = read_kernel("reduceat") % {'num_points': values.shape[axis],
                                        'num_segments': num_segments,
                                        'element_t': ctype[values.dtype.name]};
    
    
