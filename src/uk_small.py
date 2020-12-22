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
