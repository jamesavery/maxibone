import bohrium as bh;
import bohrium.user_kernel as uk;

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
    with open(f"../src/bh-kernels/{name}.c") as f:
        return f.read();


def count_labeled(labeled_image,num_labels,include_zero=False):    
    labeled_image = uk.make_behaving(labeled_image,dtype=np.uint32);
    counts        = bh.empty(num_labels+1,dtype=bh.uint64)
    
    kernel = read_kernel("count-labeled") % {'image_len':labeled_image.size,'num_labels':num_labels};    
    uk.execute(kernel, [counts,labeled_image])
    
    if(not include_zero):
        counts[0] = 0;
        
    return counts;

def collect_labeled(labeled_image,num_labels,counts):    
    labeled_image      = uk.make_behaving(labeled_image,dtype=np.uint32);
    row_starts         = bh.cumsum(counts);
    n_nonzero_labels   = bh.sum(counts);
    
    O = bh.empty(n_nonzero_labels,dtype=bh.uint64);
        
    kernel = read_kernel("collect-labeled") % {'image_len':labeled_image.size,'num_labels':num_labels};    
    uk.execute(kernel, [O,row_starts,labeled_image]);
    
    return O, row_starts;

