import bohrium as bh;
import bohrium.user_kernel as uk;
from time import time;
import sys

ctype = {
    'float32': 'float',
    'float64': 'double',
    'complex64': 'float complex',
    'complex128':'double complex',
    'int8':  'int8_t',
    'int16': 'int16_t',
    'int32': 'int32_t',
    'int64': 'int64_t',
    'uint8':  'uint8_t',
    'uint16': 'uint16_t',
    'uint32': 'uint32_t',
    'uint64': 'uint64_t',    
    'bool': 'uint8_t'
}


def read_kernel(name):
    with open("../src/bh-kernels/"+name+".c") as f:
        return f.read();

# Collect base-array indices of nonzero elements in ndarray 
def flat_nonzero(A):
    bitmap   = uk.make_behaving(A!=0);
    nz_count = bh.sum(bitmap);
    nonzeros = bh.empty((nz_count),dtype=bh.uint64); # TODO: Hvorfor bliver denne givet videre til np?
    kernel = read_kernel("collect-nonzero") % {"image_len":A.size};
    uk.execute(kernel, [nonzeros,bitmap]);
    return nonzeros;

def field_histogram(voxels, field, ranges=None,field_bins=1024, voxel_bins=4096):
    voxels = uk.make_behaving(voxels)
    field  = uk.make_behaving(field)

    assert(voxels.shape == field.shape)
    
    bins   = bh.zeros((field_bins,voxel_bins),dtype=bh.uint64)

    # TODO: Don't scan over array 4 times - perhaps roll into user kernel
    if ranges is None:
        vmin, vmax = voxels.min(), voxels.max()
        fmin, fmax = field.min(), field.max()
    else:
        ((vmin,vmax),(fmin,fmax)) = ranges
    
    kernel = read_kernel("field_histogram")
    args = {"image_length": voxels.size,
            "vmin":vmin, "vmax":vmax, "fmin":fmin, "fmax": fmax,
            "voxel_bins":voxel_bins, "field_bins": field_bins,
            "voxel_type":ctype[voxels.dtype.name], "field_type":ctype[field.dtype.name]}
    kernel = kernel % args;

    print(kernel)
    uk.execute(kernel, [bins, voxels,field])
    return bins


def axes_histogram(voxels, ranges=None, voxel_bins=256):
    voxels = uk.make_behaving(voxels)
    (Nz,Ny,Nx) = voxels.shape
    Nr = max(Nx//2,Ny//2)+1
    
    x_bins   = bh.zeros((Nx,voxel_bins),dtype=bh.uint64)
    y_bins   = bh.zeros((Ny,voxel_bins),dtype=bh.uint64)
    z_bins   = bh.zeros((Nz,voxel_bins),dtype=bh.uint64)
    r_bins   = bh.zeros((Nr,voxel_bins),dtype=bh.uint64)

    if ranges is None:
        vmin, vmax = voxels.min(), voxels.max()
    
    kernel = read_kernel("axes_histogram")
    args = {"image_length": voxels.size,
            "voxel_bins":voxel_bins,             
            "vmin":vmin, "vmax":vmax, 
            "voxel_type":ctype[voxels.dtype.name],
            "Nx": Nx, "Ny":Ny, "Nz":Nz, "Nr":Nr}

    kernel = kernel % args;

    uk.execute(kernel, [x_bins, y_bins, z_bins, r_bins, voxels])
    return x_bins, y_bins, z_bins, r_bins


def count_labeled(labeled_image,num_labels,include_zero=False):    
    labeled_image = uk.make_behaving(labeled_image,dtype=bh.uint32);
    counts        = bh.empty(num_labels+1,dtype=bh.uint64)
    
    kernel = read_kernel("count-labeled") % {'image_len':labeled_image.size,'num_labels':num_labels};    
    uk.execute(kernel, [counts,labeled_image])
    
    if(not include_zero):
        counts[0] = 0;
        
    return counts;

def collect_labeled(labeled_image,num_labels,counts,label_type=bh.uint32):    
    labeled_image      = uk.make_behaving(labeled_image,dtype=label_type);
    row_starts         = bh.cumsum(counts);
    n_nonzero_labels   = bh.sum(counts);
    
    O = bh.empty((n_nonzero_labels),dtype=bh.uint64);

    kernel = read_kernel("collect-labeled") % {'image_len':labeled_image.size,
                                               'num_labels':num_labels,
                                               'label_t': ctype[labeled_image.dtype.name]
    };    
    uk.execute(kernel, [O,row_starts,labeled_image]);
    
    return O, row_starts;

def select_segments(segment_indices, segment_starts, selected_segments):
    num_segments = len(segment_starts);

    segment_indices   = uk.make_behaving(segment_indices);
    segment_starts    = uk.make_behaving(segment_starts);
    selected_segments = uk.make_behaving(selected_segments);

    selected_starts = segment_starts[:-1];
    selected_ends   = segment_starts[1:];
    
    num_selected_segments = bh.sum(selected_segments);
    num_selected_indices  = bh.sum((selected_ends-selected_starts)[selected_segments]);

    print("num_selected_segments:",num_selected_segments, file=sys.stderr)
    print("num_selected_indices:",num_selected_indices, file=sys.stderr)    
    sys.stderr.flush();

    new_segment_starts = bh.zeros(num_selected_segments,dtype=bh.uint64);
    new_indices        = bh.zeros(num_selected_indices, dtype=bh.uint64);
    
    kernel = read_kernel("select_segments") % {'num_segments':num_segments};

    uk.execute(kernel,[new_indices,     new_segment_starts,
                       segment_indices, segment_starts,
                       selected_segments]);

    return new_indices, new_segment_starts


# LABELING
# Internal user kernel
def uk_label(nonzeros,shape,dtype=bh.uint32):

    nonzeros = uk.make_behaving(nonzeros);
    L = bh.zeros(shape,dtype=dtype);
    T = bh.arange(len(nonzeros)+1,dtype=dtype); # Initialize union-find equialence tree to "all separate"
#    L.reshape(-1)[nonzeros] = T;
    strides = bh.array(L.strides,dtype=bh.int64)//L.itemsize;

    kernel = read_kernel("ndi_label") % {
        'ndim': L.ndim,
        'total_size':L.size,
        'n_nonzeros':len(nonzeros),
        'label_t':ctype[L.dtype.name],
        'index_t':ctype[nonzeros.dtype.name]
        };

    uk.execute(kernel,[L,T,strides,nonzeros])

    return L,T


def uk_sparse_label(nonzeros,shape,label_type=bh.uint32):
    shape      = uk.make_behaving(bh.array(shape), bh.int64)
    nonzeros   = uk.make_behaving(nonzeros);
    n_nonzeros = bh.array([len(nonzeros)]);
    T          = bh.empty(len(nonzeros)+1, dtype=label_type);

    kernel = read_kernel("sparse_label") % {
        "ndim":    len(shape),
        "label_t": ctype[T.dtype.name],
        "index_t": ctype[nonzeros.dtype.name]
    };
    
    uk.execute(kernel,[T,shape,nonzeros,n_nonzeros])

    return T;
   


#TODO: Make fully compatible with ndimage.label
# * Add support for general structuring element
# * Add support for supplying output
# * Add funny return logic
def ndi_label(bitmap,dtype=bh.uint32):
    print("nonzeros")
    bitmap_flat = bitmap.reshape(-1);
    nz          = bh.nonzero(bitmap_flat)[0];

    print("uk_label")
    L,T = uk_label(nz,bitmap.shape,dtype);
    L_flat = L.reshape(-1);

    print("np.unique")
    labels, TI = np.unique(T.copy2numpy(),return_inverse=True);

    print("Repaint")
    L_flat[nz] = TI[1:];
    return L, len(labels)-1



# Label connected components in compressed bitmap -> sparse values, num_components
def sparse_label(nonzeros,shape,label_type=bh.uint32):
    print("uk_sparse_label")
    t0 = time()
    T = uk_sparse_label(nonzeros,shape,label_type);
    t1 = time()
    
    print("np.unique")
    labels, TI, counts = bh.unique(T,return_inverse=True,return_counts=True)
    counts[0] = 0;

    print("make_behaving")
    t2 = time()    
    values, num_components, counts = uk.make_behaving(TI[1:]), len(labels)-1, uk.make_behaving(counts)

    print("collect_labeled")
    t3 = time()
    indices, starts = collect_labeled(values, num_components, counts);
    t4 = time()
    
    print("uk_sparse_label:",t1-t0,
          "np.unique",t2-t1,
          "make_behaving:",t3-t2,
          "collect_labeled:",t4-t3)
    return (indices, starts), values[indices], num_components




