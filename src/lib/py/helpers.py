#! /usr/bin/python3
'''
This file contains helper functions for loading and updating HDF5 files, generating cylinder masks, and parsing commandline arguments.
'''
import sys
sys.path.append(sys.path[0]+"/../")

from config.paths import binary_root, hdf5_root
import h5py
from lib.cpp.cpu.io import load_slice
from lib.cpp.cpu.general import normalized_convert
from lib.cpp.gpu.morphology import dilate_3d, erode_3d, dilate_3d_bitpacked, erode_3d_bitpacked
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import os
import pathlib
import scipy.signal as signal
import tqdm

def block_info(h5meta_filename, scale, block_size=0, n_blocks=0, z_offset=0):
    '''
    Returns information about the blocks in a volume-matched dataset. It is used for loading blocks in the `load_block` function.

    Parameters
    ----------
    `h5meta_filename` : str
        The path to the HDF5 file containing the metadata.
    `scale` : int
        The scale of the data.
    `block_size` : int
        The size of the blocks to load. If 0, the block size is the size of a subvolume.
    `n_blocks` : int
        The number of blocks to load. If 0, all blocks are loaded.
    `z_offset` : int
        The offset in the z-dimension to start loading blocks.

    Returns
    -------
    `info` : dict(str, Any)
        A dictionary containing the dimensions, voxel size, number of blocks, block size, whether blocks are subvolumes, subvolume dimensions, subvolume nzs, and subvolume starts.

    '''

    print(f"Opening {h5meta_filename}")
    with h5py.File(h5meta_filename, 'r') as h5meta:
        vm_shifts  = h5meta["volume_matching_shifts"][:]
        Nz, Ny, Nx = h5meta['voxels'].shape
        Nz -= np.sum(vm_shifts)
        Nz, Ny, Nx = Nz//scale, Ny//scale, Nx//scale
        Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1

        subvolume_dimensions =  h5meta['subvolume_dimensions'][:]
        subvolume_nzs = subvolume_dimensions[:,0] - np.append(vm_shifts,0)

        if block_size == 0:
            # If block_size is 0, let each block be exactly a full subvolume
            blocks_are_subvolumes = True

            # Do either n_blocks subvolumes, or if n_blocks == 0: all remaining after offset
            if n_blocks == 0:
                n_blocks = len(subvolume_nzs)-z_offset
        else:
            blocks_are_subvolumes = False
            if n_blocks == 0:
                n_blocks = Nz // block_size + (Nz % block_size > 0)

        return {
            'dimensions' : (Nz, Ny, Nx, Nr),
            'voxel_size' :  h5meta["voxels"].attrs["voxelsize"],
            'n_blocks' : n_blocks,
            'block_size' : block_size,
            'blocks_are_subvolumes' : blocks_are_subvolumes,
            'subvolume_dimensions' : subvolume_dimensions,
            'subvolume_nzs' : subvolume_nzs,
            'subvolume_starts' : np.concatenate([[0],np.cumsum(subvolume_nzs)[:-1]])
        }

def circle_center(p0, p1, p2):
    '''
    Calculate the center of a circle given three points.

    Parameters
    ----------
    `p0` : numpy.array[float]
        The first point.
    `p1` : numpy.array[float]
        The second point.
    `p2` : numpy.array[float]
        The third point.

    Returns
    -------
    `c` : numpy.array[float]
        The center of the circle.
    '''

    m1, m2 = (p0+p1) / 2, (p0+p2) / 2 # Midpoints
    (dx1,dy1), (dx2,dy2) = (p1-p0), (p2-p0) # Slopes of connecting lines
    n1, n2 = np.array([dy1,-dx1]).T, np.array([dy2,-dx2]).T # Normals perpendicular to connecting lines

    A = np.array([n1,-n2]).T # Solve m1 + t1*n1 == m2 + t2*n2   <=> t1*n1 - t2*n2 = m2-m1
    (t1, t2) = la.solve(A, m2-m1)
    c1, c2 = m1 + t1*n1, m2 + t2*n2  # Center of circle

    assert(np.allclose(c1, c2))

    return c1

def close_3d(image, r):
    '''
    Apply an closing operation to a 3D image with a spherical structuring element of radius `r`.
    If the image is of type `np.uint32`, this function assumes that the image is bitpacked and uses the bitpacked morphology operations.

    Parameters
    ----------
    `image` : numpy.array[Any]
        The image to apply the closing operation to.
    `r` : int
        The radius of the structuring element.

    Returns
    -------
    `image` : numpy.array[Any]
        The image after applying the closing operation.
    '''

    if image.dtype == np.uint32:
        return morph_3d(image, r, dilate_3d_bitpacked, erode_3d_bitpacked)
    else:
        return morph_3d(image, r, dilate_3d, erode_3d)

def commandline_args(defaults):
    '''
    Parse commandline arguments, and return them in the order of the keys in the defaults dictionary.
    If a key is not provided, the default value is used.
    If a key is not provided and has the value `"<required>"`, the script will print the helpstring and exit.
    The type of the default value is used to convert the commandline argument to the appropriate type.

    Example usage:
    ```
    defaults = {
        "input_file" : "<required>",
        "output_file" : "output.txt",
        "threshold" : 0.5
    }
    input_file, output_file, threshold = commandline_args(defaults)
    ```

    The above will parse the commandline arguments, and return the values in the order of the keys in the defaults dictionary.
    If the `input_file` is not provided, the script will print the helpstring and exit.

    The helpstring is generated from the keys in the defaults dictionary, and is printed if the script is called with `--help` or `-h`.

    Parameters
    ----------
    `defaults` : dict(str, str)
        A dictionary of key-value pairs, where the key is the name of the parameter, and the value is the default value or `"<required>"` if the parameter is required and cannot have a default value. If the default value is not a string, the type of the default value is used to convert the commandline argument to the appropriate type.

    Returns
    -------
    args : list[Any]
        A list of the values of the parameters, in the order of the keys in the defaults dictionary.
    '''

    keys = list(defaults.keys())

    helpstring = f"Syntax: {sys.argv[0]} "
    for k in keys:
        if (defaults[k] == "<required>"): helpstring += f" <{k}>"
        else:                             helpstring += f" [{k}:{defaults[k]}]"

    # Do we just want to know how to call this script?
    if (len(sys.argv) == 2):
        if (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
            print(helpstring, file=sys.stderr)
            sys.exit(1)

    # Fill in parameters from commandline and defaults, converting to appropriate types
    args = []
    for i in range(len(keys)):
        default = defaults[keys[i]]
        if (len(sys.argv) <= i+1):
            if(default == "<required>"):
                print(helpstring, file=sys.stderr)
                sys.exit(1)
            else:
                args.append(default)
        else:
            args.append(type(default)(sys.argv[i+1]))

    return args

def coordinate_image(shape, verbose=0):
    '''
    Generate a 3D image of the coordinates of each voxel in the image.
    The shape of the image is `(Nz,Ny,Nx)`, and the coordinates are `(z,y,x)`.
    The coordinates are broadcasted to the shape of the image.

    Parameters
    ----------
    `shape` : tuple(int,int,int)
        The shape of the image.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    `zyxs` : numpy.array[int]
        A 3D image of the coordinates of each voxel in the image.
    '''

    #NA = np.newaxis
    #Nz,Ny,Nx   = shape
    #zs, ys, xs = np.broadcast_to(np.arange(Nz)[:,NA,NA],shape),\
    #             np.broadcast_to(np.arange(Ny)[NA,:,NA],shape),\
    #             np.broadcast_to(np.arange(Nx)[NA,NA,:],shape);
    #zyxs = np.stack([zs,ys,xs],axis=-1)
    #del zs, ys, xs
    if verbose >= 1: print(f"Broadcasting coordinates for {shape} image")
    zyxs = np.moveaxis(np.indices(shape, np.uint16), 0, -1)
    if verbose >= 1: print(f"Done")
    return zyxs

def generate_cylinder_mask(nx):
    '''
    Generate a 2D mask of a cylinder with diameter `nx` pixels.

    Parameters
    ----------
    `nx` : int
        The diameter of the cylinder in pixels.

    Returns
    -------
    `mask` : numpy.array[bool]
        A 2D boolean mask of the cylinder.
    '''

    xs = np.linspace(-1, 1, nx)
    rs = np.sqrt(xs[np.newaxis,np.newaxis,:]**2 + xs[np.newaxis,:,np.newaxis]**2)
    return rs <= 1

def generate_cylinder_mask(ny, nx):
    '''
    Generate a 2D mask of a cylinder with diameter `nx` pixels in the x-dimension and `ny` pixels in the y-dimension.

    Parameters
    ----------
    `ny` : int
        The diameter of the cylinder in pixels in the y-dimension.
    `nx` : int
        The diameter of the cylinder in pixels in the x-dimension.

    Returns
    -------
    `mask` : numpy.array[bool]
        A 2D boolean mask of the cylinder.
    '''

    ys = np.linspace(-1, 1, ny)
    xs = np.linspace(-1, 1, nx)
    return (xs[np.newaxis,:]**2 + ys[:,np.newaxis]**2) < 1

def gramschmidt(u, v, w):
    '''
    Apply the Gram-Schmidt process to the vectors `u`, `v`, and `w`.

    Parameters
    ----------
    `u` : numpy.array[float]
        The first vector.
    `v` : numpy.array[float]
        The second vector.
    `w` : numpy.array[float]
        The third vector.

    Returns
    -------
    `Q` : numpy.array[float]
        The orthonormal basis of the vectors `u`, `v`, and `w`.
    '''

    vp = v - proj(v, u)
    wp = w - proj(w, u) - proj(w, vp)

    return np.array([u / la.norm(u), vp / la.norm(v), wp / la.norm(w)])

def highest_peaks(data, n, height=0.7):
    '''
    Find the `n` highest peaks in the data.

    Parameters
    ----------
    `data` : numpy.array[float]
        The data to find the peaks in.
    `n` : int
        The number of peaks to find.
    `height` : float
        The height of the peaks to find. Default is 0.7.

    Returns
    -------
    `peaks` : numpy.array[int]
        The indices of the `n` highest peaks in the data.
    '''

    peaks, info = signal.find_peaks(data, height=height*data.max())
    return peaks[np.argsort(info['peak_heights'])][:n]

def homogeneous_transform(xs, M, verbose=0):
    '''
    Apply a homogeneous transformation matrix `M` to the homogeneous coordinates `xs`.

    Parameters
    ----------
    `xs` : numpy.array[float]
        The homogeneous coordinates to transform.
    `M` : numpy.array[float]
        The homogeneous transformation matrix to apply.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    `hxs` : numpy.array[float]
        The transformed homogeneous coordinates.
    '''

    shape = np.array(xs.shape)
    assert(shape[-1] == 3)
    shape[-1] = 4
    hxs = np.empty(shape, dtype=xs.dtype)
    hxs[...,:3] = xs
    hxs[..., 3]  = 1

    if verbose >= 1: print(hxs.shape, M.shape)
    return hxs @ M.T

def hom_linear(A):
    '''
    Generate a homogeneous linear transformation matrix for the matrix `A`.

    Parameters
    ----------
    `A` : numpy.array[float]
        The matrix to generate the transformation matrix for.

    Returns
    -------
    `M` : numpy.array[float]
        The homogeneous linear transformation matrix for the matrix `A`.
    '''

    M = np.eye(4, dtype=float)
    M[:3,:3] = A
    return M

def hom_translate(x):
    '''
    Generate a homogeneous translation matrix for the vector `x`.

    Parameters
    ----------
    `x` : numpy.array[float]
        The vector to generate the translation matrix for.

    Returns
    -------
    `T` : numpy.array[float]
        The homogeneous translation matrix for the vector `x`.
    '''

    T = np.eye(4, dtype=float)
    T[0:3,3] = x
    return T

def h5meta_info_volume_matched(sample):
    '''
    Returns the volume-matched dimensions, subvolume dimensions, and voxel size for a sample.

    Parameters
    ----------
    `sample` : str
        The name of the sample to get the information for.

    Returns
    -------
    `((Nz,Ny,Nx), subvolume_nzs, voxel_size)` : tuple(tuple(int,int,int), numpy.array[int], numpy.array[float])
        A tuple containing the volume-matched dimensions, subvolume dimensions, and voxel size for the sample.
    '''

    with h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5", "r") as h5meta:
        vm_shifts  = h5meta["volume_matching_shifts"][:]
        Nz, Ny, Nx = h5meta['voxels'].shape
        Nz -= np.sum(vm_shifts)

        subvolume_dimensions = h5meta['subvolume_dimensions'][:]
        subvolume_nzs        = subvolume_dimensions[:,0] - np.append(vm_shifts,0)
        voxel_size           = h5meta["voxels"].attrs["voxelsize"]

        return ((Nz,Ny,Nx), subvolume_nzs, voxel_size)

def load_block(sample, scale, offset, block_size, mask_name, mask_scale, field_names, field_scale):
    '''
    Load a block of voxels and fields from the binary and HDF5 files.
    The block is loaded at the given offset and has the given size.
    The data itself is loaded from the binary files, and the mask and metadata are loaded from the HDF5 files.
    If a mask is provided, it is applied to the voxels.
    If the field and/or mask scales are different from the voxel scale, they are upscaled to the voxel scale.

    Parameters
    ----------
    `sample` : str
        The name of the sample to load the block from.
    `scale` : int
        The scale of the voxels.
    `offset` : int
        The offset in the z-dimension to start loading the block.
    `block_size` : int
        The size of the block to load.
    `mask_name` : str
        The name of the mask to apply to the voxels. If None, no mask is applied.
    `mask_scale` : int
        The scale of the mask.
    `field_names` : list[str]
        The names of the fields to load.
    `field_scale` : int
        The scale of the fields.

    Returns
    -------
    `voxels, fields` : tuple(numpy.array[uint16], numpy.array[uint16])
        A tuple containing the voxels and fields loaded from the binary files. Note that the fields have an additional dimension for the number of fields.
    '''

    NA = np.newaxis
    Nfields = len(field_names)
    h5meta = h5py.File(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', 'r')
    Nz1x, Ny1x, Nx1x = h5meta['voxels'].shape
    Nz1x -= np.sum(h5meta["volume_matching_shifts"][:])
    fNz, fNy, fNx = Nz1x // field_scale, Ny1x // field_scale, Nx1x // field_scale
    Nz, Ny, Nx = Nz1x // scale, Ny1x // scale, Nx1x // scale
    mNz, mNy, mNx = Nz1x // mask_scale, Ny1x // mask_scale, Nx1x // mask_scale
    mask_scale_relative = mask_scale // scale
    h5meta.close()
    block_size = min(block_size, Nz-offset)

    voxels = np.zeros((block_size, Ny, Nx), dtype=np.uint16)
    fields = np.zeros((Nfields, block_size//field_scale, fNy, fNx), dtype=np.uint16)

    if mask_name is not None:
        for i in tqdm.tqdm(range(1),f"Loading {mask_name} mask from {hdf5_root}/masks/{mask_scale}x/{sample}.h5"):
            with h5py.File(f"{hdf5_root}/masks/{mask_scale}x/{sample}.h5","r") as h5mask:
                mask = h5mask[mask_name]["mask"][offset//mask_scale_relative:offset//mask_scale_relative + block_size//mask_scale_relative]

    for i in tqdm.tqdm(range(1),f"Loading {voxels.shape} voxels from {binary_root}/voxels/{scale}x/{sample}.uint16", leave=True):
        # TODO: Don't use 3 different methods for load/store
        load_slice(voxels, f'{binary_root}/voxels/{scale}x/{sample}.uint16', (offset, 0, 0), (block_size, Ny, Nx))

    for i in tqdm.tqdm(range(Nfields),f"Loading {binary_root}/fields/implant-{field_names}/{field_scale}x/{sample}.npy",leave=True):
        fi = np.load(f"{binary_root}/fields/implant-{field_names[i]}/{field_scale}x/{sample}.npy", mmap_mode='r')
        fields[i,:] = fi[offset//field_scale:offset//field_scale + block_size//field_scale]

    if mask_name is not None:
        nz, ny, nx = (block_size//mask_scale_relative), Ny//mask_scale_relative, Nx//mask_scale_relative
        mask_1x = np.broadcast_to(mask[:,NA,:,NA,:,NA],(nz, mask_scale_relative, ny, mask_scale_relative, nx, mask_scale_relative))
        mask_1x = mask_1x.reshape(nz*mask_scale_relative, ny*mask_scale_relative, nx*mask_scale_relative)
        voxels[:nz*mask_scale_relative] *= mask_1x             # block_size may not be divisible by mask_scale_relative
        voxels[nz*mask_scale_relative:] *= mask_1x[-1][NA,...] # Remainder gets last line of mask

    return voxels, fields

def morph_3d(image, r, fa, fb):
    '''
    Applies two 3D spherical morphology operation (`fa` and `fb`) of radius `r` to the image `img`.
    It is a generic function used to build `open_3d` and `close_3d`.
    Each function is applied with spheres of max radius `rmin` (currently 16), and the remainder is applied with a sphere of radius `rrest`.
    This is due to the fact that `r//rmin` applications with radius `rmin` are faster than a single application with radius `r`.

    See `open_3d` and `close_3d` for examples of usage.

    Parameters
    ----------
    `image` : numpy.array[Any]
        The image to apply the morphological operation to.
    `r` : int
        The radius of the morphological operation.
    `fa` : function
        The first morphological operation to apply.
    `fb` : function
        The second morphological operation to apply.

    Returns
    -------
    `I1` : numpy.array[Any]
        The image after applying the morphological operation.
    '''

    I1 = image.copy().astype(image.dtype)
    I2 = np.empty(image.shape, dtype=image.dtype)
    rmin = 16
    rmins = r // rmin
    rrest = r % rmin
    for _ in range(rmins):
        fa(I1, rmin, I2)
        I1, I2 = I2, I1
    if rrest > 0:
        fa(I1, rrest, I2)
        I1, I2 = I2, I1

    for i in range(rmins):
        fb(I1, rmin, I2)
        I1, I2 = I2, I1
    if rrest > 0:
        fb(I1, rrest, I2)
        I1, I2 = I2, I1

    return I1

def normalize(A, value_range, nbits=16, dtype=np.uint16):
    '''
    Normalize an array `A` to the range `value_range` and convert to `dtype` with `nbits` bits.
    Afterwards, `1,...,2**(nbits)-1` correspond to `vmin,...,vmax` and `0` corresponds to a masked value.

    Parameters
    ----------
    `A` : np.ndarray
        The array to normalize.
    `value_range` : tuple(Any,Any)
        The range of values in `A` to normalize to.
    `nbits` : int
        The number of bits to use for the output.
    `dtype` : np.dtype
        The data type of the output.

    Returns
    -------
    `A_normed` : np.ndarray
        The normalized array.
    '''
    vmin, vmax = value_range
    return (A != 0) * ((((A-vmin) / (vmax-vmin)) * (2**nbits-1)).astype(dtype)+1)

def open_3d(image, r):
    '''
    Apply an opening operation to a 3D image with a spherical structuring element of radius `r`.
    If the image is of type `np.uint32`, this function assumes that the image is bitpacked and uses the bitpacked morphology operations.

    Parameters
    ----------
    `image` : numpy.array[Any]
        The image to apply the opening operation to.
    `r` : int
        The radius of the structuring element.

    Returns
    -------
    `image` : numpy.array[Any]
        The image after applying the opening operation.
    '''

    if image.dtype == np.uint32:
        return morph_3d(image, r, erode_3d_bitpacked, dilate_3d_bitpacked)
    else:
        return morph_3d(image, r, erode_3d, dilate_3d)

def plot_middle_planes(tomo, output_dir, prefix):
    '''
    Plot the middle planes of a 3D volume and save them to the output directory.
    For a 3D volume of shape `(nz,ny,nx)`, the middle planes are `(nz//2,:,:), (:,ny//2,:), (:,:,nx//2)`.
    The images are saved as PDF files with `interpolation='none'` to ensure the pixels are not interpolated.

    Parameters
    ----------
    `tomo` : numpy.array[Any]
        The 3D volume to plot the middle planes of.
    `output_dir` : str
        The directory to save the images to.
    `prefix` : str
        The prefix to add to the image filenames.

    Returns
    -------
    None
    '''
    assert len(tomo.shape) == 3
    nz, ny, nx = tomo.shape
    names = ['yx', 'zx', 'zy']
    planes = [tomo[nz//2,:,:], tomo[:,ny//2,:], tomo[:,:,nx//2]]

    for name, plane in zip(names, planes):
        plt.figure(figsize=(10,10))
        plt.imshow(plane, interpolation='none')
        plt.colorbar()
        plt.savefig(f"{output_dir}/{prefix}_{name}.pdf", bbox_inches='tight')
        plt.close()

def proj(u, v):
    '''
    Project the vector `u` onto the vector `v`.

    Parameters
    ----------
    `u` : numpy.array[float]
        The vector to project.
    `v` : numpy.array[float]
        The vector to project onto.

    Returns
    -------
    `proj` : numpy.array[float]
        The projection of `u` onto `v`.
    '''

    return (np.dot(u,v) / np.dot(v,v)) * v

def row_normalize(A, r):
    '''
    Normalize the rows of a matrix `A` by the vector `r`.

    Parameters
    ----------
    `A` : numpy.array[float]
        The matrix to normalize.
    `r` : numpy.array[float]
        The vector to normalize the rows by.

    Returns
    -------
    `A_normed` : numpy.array[float]
        The normalized matrix.
    '''

    na = np.newaxis
    return A / (r[:,na] + (r==0)[:,na])

def sphere(n):
    '''
    Generate a 3D boolean mask of a sphere with diameter `n` pixels.

    Parameters
    ----------
    `n` : int
        The diameter of the sphere in pixels.

    Returns
    -------
    `mask` : numpy.array[bool]
        A 3D boolean mask of the sphere.
    '''

    NA = np.newaxis
    xs = np.linspace(-1, 1, n)
    return (xs[:,NA,NA]**2 + xs[NA,:,NA]**2 + xs[NA,NA,:]**2) <= 1

def to_int(x, dtype):
    '''
    Convert a numpy array to an integer type.
    This is a wrapper around the C++ implementation in `lib/cpp/*/general.cc`.
    If the dimensions of the array are 3, the C++ implementation is used, otherwise the pure Python implementation is used.

    Parameters
    ----------
    `x` : numpy.array[Any]
        The array to convert.
    `dtype` : numpy.dtype
        The integer type to convert the array to.

    Returns
    -------
    `result` : numpy.array[dtype]
        The converted array.
    '''

    if len(x.shape) != 3:
        return to_int_py(x,dtype)

    result = np.empty(x.shape, dtype=dtype)
    normalized_convert(x, result)
    return result

def to_int_py(x, dtype):
    '''
    Convert a numpy array to an integer type.
    This is a pure Python implementation, which is slower than the C++ implementation in `to_int`.
    It is used as a fallback when the dimensions of the array are not 3.

    Parameters
    ----------
    `x` : numpy.array[Any]
        The array to convert.
    `dtype` : numpy.dtype
        The integer type to convert the array to.

    Returns
    -------
    `result` : numpy.array[dtype]
        The converted array.
    '''

    vmin, vmax = x.min(), x.max()
    # Ensure everything is float32, to ensure float32 computations
    int_max = np.float32(np.iinfo(dtype).max - 1)
    factor = np.float32(vmax - vmin + (vmin==vmax))
    vmin, vmax = np.float32(vmin), np.float32(vmax)
    result = x.astype(np.float32)
    result -= vmin
    result /= factor
    result *= int_max
    result = np.floor(result).astype(dtype)
    result += 1
    return result

def update_hdf5(filename, group_name, datasets={}, attributes={}, dimensions=None,
                compression=None, chunk_shape=None):
    '''
    Update an HDF5 file with new datasets and attributes.

    Parameters
    ----------
    `filename` : str
        The path to the HDF5 file to update.
    `group_name` : str
        The name of the group in the HDF5 file to update.
    `datasets` : dict[str, numpy.array[Any]]
        A dictionary of dataset names and numpy arrays to update or add to the HDF5 file.
    `attributes` : dict[str, Any]
        A dictionary of attribute names and values to update or add to the HDF5 file.
    `dimensions` : dict[str, list[str]]
        A dictionary of dataset names and lists of dimension descriptions to update or add to the HDF5 file.
    `compression` : str
        The compression algorithm to use for the datasets. Default is no compression.
    `chunk_shape` : tuple[int]
        The shape of the chunks to use for the datasets. Default is no chunking.

    Returns
    -------
    None
    '''

    # Create the file and directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    f = h5py.File(filename,'a')

    # Create the group if it doesn't exist
    if((group_name is not None) and (group_name != "/")):
        g = f.require_group(group_name)
    else:
       g = f

    # Update datasets
    for k in datasets:
        v = datasets[k]
        if(k in g): del g[k]
        g.create_dataset(k, shape=v.shape, dtype=v.dtype,
                          compression=compression, chunks=chunk_shape, maxshape=None)
        g[k][:] = v[:]

        # Update dimensions if provided
        if dimensions is not None:
            try:
                dims = dimensions[k]
                for i, description in enumerate(dims):
                    g[k].dims[i] = description
            except:
                pass

    # Update attributes
    for k in attributes:
        v = attributes[k]
        g.attrs[k] = v

    f.close()

def update_hdf5_mask(filename, group_name, datasets={}, attributes={}, dimensions=None,
                     compression="lzf", chunk_shape=None):
    '''
    Update an HDF5 file with new datasets and attributes.
    This function is specifically for updating masks, and uses lzf compression and no chunking by default.

    Parameters
    ----------
    `filename` : str
        The path to the HDF5 file to update.
    `group_name` : str
        The name of the group in the HDF5 file to update.
    `datasets` : dict[str, numpy.array[Any]]
        A dictionary of dataset names and numpy arrays to update or add to the HDF5 file.
    `attributes` : dict[str, Any]
        A dictionary of attribute names and values to update or add to the HDF5 file.
    `dimensions` : dict[str, list[str]]
        A dictionary of dataset names and lists of dimension descriptions to update or add to the HDF5 file.
    `compression` : str
        The compression algorithm to use for the datasets. Default is lzf.
    `chunk_shape` : tuple[int]
        The shape of the chunks to use for the datasets. Default is no chunking.

    Returns
    -------
    None
    '''

    update_hdf5(filename, group_name, datasets, attributes, dimensions, compression, chunk_shape)

def zyx_to_UVWp_transform(cm, voxel_size, UVW, w0, cp, UVWp):
    '''
    Generate the transformation matrix from zyx to UVWp coordinates.

    Parameters
    ----------
    `cm` : numpy.array[float]
        The center of mass of the volume.
    `voxel_size` : float
        The size of the voxels in micrometers.
    `UVW` : numpy.array[float]
        The orientation of the volume.
    `w0` : float
        The width of the volume.
    `cp` : numpy.array[float]
        The center of the volume.
    `UVWp` : numpy.array[float]
        The orientation of the volume.

    Returns
    -------
    `M` : numpy.array[float]
        The transformation matrix from zyx to UVWp coordinates.
    '''

    Tcm   = hom_translate(-cm * voxel_size)
    Muvw  = hom_linear(UVW)
    TW0   = hom_translate((0, 0, -w0 * voxel_size))
    Tcp   = hom_translate(-cp)
    Muvwp = hom_linear(UVWp)

    return Muvwp @ Tcp @ TW0 @ Muvw @ Tcm