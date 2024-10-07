#! /usr/bin/python3
'''
This file contains helper functions for loading and updating HDF5 files, generating cylinder masks, and parsing commandline arguments.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent.parent}')

from config.paths import binary_root, hdf5_root
import datetime
from functools import partial
import h5py
import lib.cpp.gpu.bitpacking as lib_bitpacking
import lib.cpp.cpu.connected_components as lib_connected_components
import lib.cpp.cpu.io as lib_io
import lib.cpp.cpu.general as lib_general
import lib.cpp.gpu.morphology as lib_morphology
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import numpy as np
import numpy.linalg as la
import psutil
import scipy.ndimage as ndi
import scipy.signal as signal
import tqdm

def bitpack_decode(src, dst=None, block_size=32, verbose=0):
    '''
    Decode a 3D image with bitpacking.
    It is done in blocks of size `block_size`, to ensure that the target device has enough memory.

    Parameters
    ----------
    `src` : numpy.array[uint32]
        The image to decode.
    `dst` : numpy.array[uint8]
        The decoded image. If None, a new array is created and returned.
    `block_size` : int
        The size of the z dimension of the blocks to decode. Default is 32.
    `verbose` : int
        The verbosity level. Default is 0. If 1, a progress bar is shown.

    Returns
    -------
    `None`
    '''

    nz, ny, nx = src.shape

    assert src.dtype == np.uint32

    if dst is None:
        dst = np.empty((nz, ny, nx*32), dtype=np.uint8)

    blocks = (nz + block_size - 1) // block_size
    if verbose >= 1:
        iterator = tqdm.tqdm(range(blocks), desc="Bitpack decoding", unit="block")
    else:
        iterator = range(blocks)
    for i in iterator:
        start, end = i*block_size, (i+1)*block_size
        end = min(end, nz)
        lib_bitpacking.decode(src[start:end], dst[start:end], verbose)

    return dst

def bitpack_encode(src, dst=None, block_size=32, verbose=0):
    '''
    Encode a 3D image with bitpacking.
    It is done in blocks of size `block_size`, to ensure that the target device has enough memory.

    Parameters
    ----------
    `src` : numpy.array[uint8]
        The image to encode.
    `dst` : numpy.array[uint32]
        The encoded image. If None, a new array is created and returned.
    `block_size` : int
        The size of the z dimension of the blocks to encode. Default is 32.
    `verbose` : int
        The verbosity level. Default is 0. If 1, a progress bar is shown.

    Returns
    -------
    `dst` : numpy.array[uint32]
        The encoded image.
    '''

    nz, ny, nx = src.shape

    assert src.dtype == np.uint8 or src.dtype == bool
    assert nx % 32 == 0

    if src.dtype == bool:
        src = src.astype(np.uint8)

    if dst is None:
        dst = np.empty((nz, ny, nx//32), dtype=np.uint32)

    blocks = (nz + block_size - 1) // block_size
    if verbose >= 1:
        iterator = tqdm.tqdm(range(blocks), desc="Bitpack encoding", unit="block")
    else:
        iterator = range(blocks)
    for i in iterator:
        start, end = i*block_size, (i+1)*block_size
        end = min(end, nz)
        lib_bitpacking.encode(src[start:end], dst[start:end], verbose)

    return dst

def chunk_info(h5meta_filename, scale, chunk_size=0, n_chunks=0, z_offset=0, verbose=0):
    '''
    Returns information about the chunks in a volume-matched dataset. It is used for loading chunks in the `load_chunk` function.

    Parameters
    ----------
    `h5meta_filename` : str
        The path to the HDF5 file containing the metadata.
    `scale` : int
        The scale of the data.
    `chunk_size` : int
        The size of the chunks to load. If 0, the chunk size is the size of a subvolume.
    `n_chunks` : int
        The number of chunks to load. If 0, all chunks are loaded.
    `z_offset` : int
        The offset in the z-dimension to start loading chunks.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    `info` : dict(str, Any)
        A dictionary containing the dimensions, voxel size, number of chunks, chunk size, whether chunks are subvolumes, subvolume dimensions, subvolume nzs, and subvolume starts.

    '''

    if verbose >= 2: print(f"Opening {h5meta_filename}")
    with h5py.File(h5meta_filename, 'r') as h5meta:
        vm_shifts  = h5meta["volume_matching_shifts"][:]
        Nz, Ny, Nx = h5meta['voxels'].shape
        Nz -= np.sum(vm_shifts)
        Nz, Ny, Nx = Nz//scale, Ny//scale, Nx//scale
        Nr = int(np.sqrt((Nx//2)**2 + (Ny//2)**2))+1

        subvolume_dimensions =  h5meta['subvolume_dimensions'][:]
        subvolume_nzs = subvolume_dimensions[:,0] - np.append(vm_shifts,0)

        if chunk_size == 0:
            # If chunk_size is 0, let each chunk be exactly a full subvolume
            chunks_are_subvolumes = True

            # Do either n_chunks subvolumes, or if n_chunks == 0: all remaining after offset
            if n_chunks == 0:
                n_chunks = len(subvolume_nzs)-z_offset
        else:
            chunks_are_subvolumes = False
            if n_chunks == 0:
                n_chunks = Nz // chunk_size + (Nz % chunk_size > 0)

        return {
            'dimensions' : (Nz, Ny, Nx, Nr),
            'voxel_size' :  h5meta["voxels"].attrs["voxelsize"],
            'n_chunks' : n_chunks,
            'chunk_size' : chunk_size,
            'chunks_are_subvolumes' : chunks_are_subvolumes,
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

    assert (np.allclose(c1, c2))

    return c1

def close_3d(image, r, verbose=0):
    '''
    Apply an closing operation to a 3D image with a spherical structuring element of radius `r`.
    If the image is of type `np.uint32`, this function assumes that the image is bitpacked and uses the bitpacked morphology operations.

    Parameters
    ----------
    `image` : numpy.array[Any]
        The image to apply the closing operation to.
    `r` : int
        The radius of the structuring element.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    `image` : numpy.array[Any]
        The image after applying the closing operation.
    '''

    if image.dtype == np.uint32:
        return morph_3d(image, r, [lib_morphology.dilate_3d_sphere_bitpacked, lib_morphology.erode_3d_sphere_bitpacked], verbose)
    else:
        return morph_3d(image, r, [lib_morphology.dilate_3d_sphere, lib_morphology.erode_3d_sphere], verbose)

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

    # Old, slow way:
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

def dilate_3d(image, r, verbose=0):
    '''
    Apply an dilation operation to a 3D image with a spherical structuring element of radius `r`.
    If the image is of type `np.uint32`, this function assumes that the image is bitpacked and uses the bitpacked morphology operations.

    Parameters
    ----------
    `image` : numpy.array[Any]
        The image to apply the dilation operation to.
    `r` : int
        The radius of the structuring element.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    `image` : numpy.array[Any]
        The image after applying the dilation operation.
    '''

    if image.dtype == np.uint32:
        return morph_3d(image, r, [lib_morphology.dilate_3d_sphere_bitpacked], verbose)
    else:
        return morph_3d(image, r, [lib_morphology.dilate_3d_sphere], verbose)

def erode_3d(image, r, verbose=0):
    '''
    Apply an erosion operation to a 3D image with a spherical structuring element of radius `r`.
    If the image is of type `np.uint32`, this function assumes that the image is bitpacked and uses the bitpacked morphology operations.

    Parameters
    ----------
    `image` : numpy.array[Any]
        The image to apply the erosion operation to.
    `r` : int
        The radius of the structuring element.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    `image` : numpy.array[Any]
        The image after applying the erosion operation.
    '''

    if image.dtype == np.uint32:
        return morph_3d(image, r, [lib_morphology.erode_3d_sphere_bitpacked], verbose)
    else:
        return morph_3d(image, r, [lib_morphology.erode_3d], verbose)

def gauss_kernel(sigma):
    '''
    Create a 1D Gaussian kernel with a given sigma.
    It mimics the kernel created in `scipy.ndimage.gaussian_filter1d`.

    Parameters
    ----------
    `sigma` : float
        The standard deviation of the Gaussian.

    Returns
    -------
    `kernel` : numpy.array[float]
        The 1D Gaussian kernel.
    '''

    radius = round(4.0 * sigma) # stolen from the default scipy parameters
    # Deprecated:
    #kernel = ndi.filters._gaussian_kernel1d(sigma_voxels, 0, radius).astype(internal_type)

    if False:
        # Create a 1D Gaussian
        x = np.arange(-radius, radius + 1)
        kernel = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

        return kernel
    else:
        # Stolen from ndimage. Makes it easier to compare with scipy.
        sigma2 = sigma * sigma
        x = np.arange(-radius, radius+1)
        phi_x = np.exp(-0.5 / sigma2 * x ** 2)
        phi_x = phi_x / phi_x.sum()

        return phi_x

def generate_cylinder_mask(ny, nx=None):
    '''
    Generate a 2D mask of a cylinder with diameter `nx` pixels in the x-dimension and `ny` pixels in the y-dimension.
    If `nx` is not provided, it is set to `ny`.

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

    if nx is None: nx = ny

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

def label_and_store_chunk(i, chunk, chunk_prefix):
    '''
    Label a chunk and write it to disk.

    Parameters
    ----------
    `i` : int
        The index of the chunk.
    `chunk` : np.ndarray[uint16]
        The chunk to label.
    `chunk_prefix` : str
        The prefix to use for the filename.

    Returns
    -------
    `n_features` : int
        The number of features found in the chunk.
    '''

    label, n_features = ndi.label(chunk, output=np.int64)
    label.tofile(f'{chunk_prefix}{i}.int64')
    del label

    return n_features

def largest_cc_of(sample_name, scale, mask, mask_name, plotting, plotting_dir, verbose=0):
    '''
    Find the largest connected component of a mask.
    The output is a binary mask with only the largest connected component.

    Parameters
    ----------
    `sample_name` : str
        The sample name.
    `scale` : int
        The scale of the sample.
    `mask` : np.ndarray[bool]
        The mask to find the largest connected component of.
    `mask_name` : str
        The name of the mask.
    `plotting` : bool
        Whether to plot the middle planes of the mask.
    `plotting_dir` : str
        The directory to plot the middle planes to.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    `largest_component` : np.ndarray[bool]
        The filtered largest connected component of the mask.
    '''

    nz, ny, nx = mask.shape
    flat_size = nz * ny * nx
    # Times 10 because input is uint8 (1 byte) and output is int64 (8 bytes)
    should_be_on_disk = flat_size*9 > psutil.virtual_memory().available
    layer_size = ny * nx
    n_cores = mp.cpu_count() // 2 # Only count physical cores, assumes hyperthreading
    available_memory = 1024**3 * 4 * n_cores # 1 GB per core
    memory_per_core = available_memory // n_cores
    elements_per_core = memory_per_core // 8 # 8 bytes per element
    layers_per_core = elements_per_core // layer_size
    # Align n_chunks to power of 2
    n_chunks = max(1, int(2**np.ceil(np.log2(nz // layers_per_core)))) if nz > layers_per_core else 1
    layers_per_chunk = nz // n_chunks
    intermediate_folder = f"/tmp/maxibone/labels_bone_region_{mask_name}/{scale}x"
    os.makedirs(intermediate_folder, exist_ok=True)

    # If the mask is smaller than 1 GB, just label it in one go
    if layers_per_chunk == 0 or layers_per_chunk >= nz:
        label, n_features = ndi.label(mask, output=np.int64)
        bincnts           = np.bincount(label[label > 0], minlength=n_features+1)
        largest_cc_ix     = np.argmax(bincnts)

        return (label == largest_cc_ix)
    elif should_be_on_disk:
        start = datetime.datetime.now()
        with ThreadPool(n_cores) as pool:
            label_chunk_partial = partial(label_and_store_chunk, chunk_prefix=f"{intermediate_folder}/{sample_name}_")
            chunks = [mask[i*layers_per_chunk:(i+1)*layers_per_chunk] for i in range(n_chunks-1)]
            chunks.append(mask[(n_chunks-1) * layers_per_chunk:])
            n_labels = pool.starmap(label_chunk_partial, enumerate(chunks))
            # Free memory
            for chunk in chunks:
                del chunk
            del chunks
        end = datetime.datetime.now()
        # load uint16, threshold (uint16 > uint8), label (int64), write int64
        total_bytes_processed = flat_size*2 + flat_size*2 + flat_size*8 + flat_size*8
        gb_per_second = total_bytes_processed / (end-start).total_seconds() / 1024**3
        if verbose >= 1:
            print (f'Loading and labelling {mask_name} took {end-start}. (throughput: {gb_per_second:.02f} GB/s)')

        np.array(n_labels, dtype=np.int64).tofile(f"{intermediate_folder}/{sample_name}_n_labels.int64")

        largest_component = np.zeros((nz, ny, nx), dtype=bool)
        lib_connected_components.largest_connected_component(largest_component, f"{intermediate_folder}/{sample_name}_", n_labels, (nz,ny,nx), (layers_per_chunk,ny,nx), verbose)

        return largest_component
    else: # label in chunks, but in memory
        def label_worker(dataset, start, end):
            '''
            Label a chunk of the dataset - inplace.

            Parameters
            ----------
            `dataset` : np.ndarray[bool]
                The dataset to label.
            `start` : int
                The start `z` index of the chunk.
            `end` : int
                The end `z` index of the chunk.

            Returns
            -------
            `n_features` : int
                The number of features found in the chunk.
            '''

            chunk = dataset[start:end]
            n_features = ndi.label(chunk, output=chunk)
            return n_features

        start = datetime.datetime.now()

        with ThreadPool(n_cores) as pool:
            labels = mask.astype(np.int64)
            starts = [i*layers_per_chunk for i in range(n_chunks)]
            ends = [(i+1)*layers_per_chunk for i in range(n_chunks-1)] + [nz]
            n_labels = pool.starmap(label_worker, zip([labels]*n_chunks, starts, ends))

        end = datetime.datetime.now()

        total_bytes_processed = flat_size*1 + flat_size*8
        gb_per_second = total_bytes_processed / (end-start).total_seconds() / 1024**3
        if verbose >= 1:
            print (f'Labeling {mask_name} took {end-start}. (throughput: {gb_per_second:.02f} GB/s)')

        final_n_labels = lib_connected_components.merge_labeled_chunks(labels, np.array(n_labels), (n_chunks, layers_per_chunk, ny, nx), verbose)

        if plotting:
            plot_middle_planes(labels, plotting_dir, 'labeled', verbose=verbose)
            plot_middle_planes(labels > 0, plotting_dir, 'labeled_binary', verbose=verbose)

        # TODO use lib_general.bincount at some point.
        bincounts = np.bincount(labels[labels > 0], minlength=final_n_labels+1)
        largest_cc = np.argmax(bincounts)

        return (labels == largest_cc)


def load_chunk(sample, scale, offset, chunk_size, mask_name, mask_scale, field_names, field_scale, verbose = 0):
    '''
    Load a chunk of voxels and fields from the binary and HDF5 files.
    The chunk is loaded at the given offset and has the given size.
    The data itself is loaded from the binary files, and the mask and metadata are loaded from the HDF5 files.
    If a mask is provided, it is applied to the voxels.
    If the field and/or mask scales are different from the voxel scale, they are upscaled to the voxel scale.

    Parameters
    ----------
    `sample` : str
        The name of the sample to load the chunk from.
    `scale` : int
        The scale of the voxels.
    `offset` : int
        The offset in the z-dimension to start loading the chunk.
    `chunk_size` : int
        The size of the chunk to load.
    `mask_name` : str
        The name of the mask to apply to the voxels. If None, no mask is applied.
    `mask_scale` : int
        The scale of the mask.
    `field_names` : list[str]
        The names of the fields to load.
    `field_scale` : int
        The scale of the fields.
    `verbose` : int
        The verbosity level. Default is 0.

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
    chunk_size = min(chunk_size, Nz-offset)

    voxels = np.zeros((chunk_size, Ny, Nx), dtype=np.uint16)
    fields = np.zeros((Nfields, chunk_size//field_scale, fNy, fNx), dtype=np.uint16)

    if mask_name is not None:
        mask_iter = tqdm.tqdm(range(1), f"Loading {mask_name} mask from {hdf5_root}/masks/{mask_scale}x/{sample}.h5") if verbose >= 2 else range(1)
        for i in mask_iter:
            with h5py.File(f"{hdf5_root}/masks/{mask_scale}x/{sample}.h5","r") as h5mask:
                mask = h5mask[mask_name]["mask"][offset//mask_scale_relative:offset//mask_scale_relative + chunk_size//mask_scale_relative]

    voxels_iter = tqdm.tqdm(range(1),f"Loading {voxels.shape} voxels from {binary_root}/voxels/{scale}x/{sample}.uint16", leave=True) if verbose >= 2 else range(1)
    for i in voxels_iter:
        # TODO: Don't use 3 different methods for load/store
        lib_io.load_slice(voxels, f'{binary_root}/voxels/{scale}x/{sample}.uint16', (offset, 0, 0), (chunk_size, Ny, Nx))

    fields_iter = tqdm.tqdm(range(Nfields),f"Loading {binary_root}/fields/implant-{field_names}/{field_scale}x/{sample}.npy",leave=True) if verbose >= 2 else range(Nfields)
    for i in fields_iter:
        fi = np.load(f"{binary_root}/fields/implant-{field_names[i]}/{field_scale}x/{sample}.npy", mmap_mode='r')
        fields[i,:] = fi[offset//field_scale:offset//field_scale + chunk_size//field_scale]

    if mask_name is not None:
        nz, ny, nx = (chunk_size//mask_scale_relative), Ny//mask_scale_relative, Nx//mask_scale_relative
        mask_1x = np.broadcast_to(mask[:,NA,:,NA,:,NA],(nz, mask_scale_relative, ny, mask_scale_relative, nx, mask_scale_relative))
        mask_1x = mask_1x.reshape(nz*mask_scale_relative, ny*mask_scale_relative, nx*mask_scale_relative)
        voxels[:nz*mask_scale_relative] *= mask_1x             # chunk_size may not be divisible by mask_scale_relative
        voxels[nz*mask_scale_relative:] *= mask_1x[-1][NA,...] # Remainder gets last line of mask

    return voxels, fields

def morph_3d(image, r, fs, verbose=0):
    '''
    Applies consecutive 3D spherical morphology operation (`fs`) of radius `r` to the image `img`.
    It is a generic function used to build `open_3d` and `close_3d`.
    Each function is applied with spheres of max radius `rmin` (currently 16), and the remainder is applied with a sphere of radius `rrest`.
    This is due to the fact that `r//rmin` applications with radius `rmin` are faster than a single application with radius `r`.
    The reasoning behind multiple functions is that intermediate buffers can be reused.

    See `open_3d` and `close_3d` for examples of usage.

    Parameters
    ----------
    `image` : numpy.array[Any]
        The image to apply the morphological operation to.
    `r` : int
        The radius of the morphological operation.
    `fs` : list[function]
        The morphological operations to apply.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    `I` : numpy.array[Any]
        The image after applying the morphological operation.
    '''

    # Allocate temporary arrays
    I1 = image.copy().astype(image.dtype)
    I2 = np.empty(image.shape, dtype=image.dtype)

    # Determine number of applications of fa and fb
    rmin = 16
    rmins = r // rmin
    rrest = r % rmin

    # Apply the functions
    for f in fs:
        rep_rng = range(rmins + (rrest > 0))
        rep_desc = f'{rmins} times with radius 16' if rrest > 0 else ''
        rep_desc = f'{rep_desc} and once with {rrest}' if rrest > 0 else rep_desc
        rep_iter = tqdm.tqdm(rep_rng, desc=f"Applying {f.__name__} {rep_desc}", unit="iteration") if verbose >= 2 else rep_rng
        for i in rep_iter:
            this_r = rmin if i < rmins else rrest
            f(I1, this_r, I2)
            I1, I2 = I2, I1

    # Ensure temporary array is deallocated
    del I2

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

def open_3d(image, r, verbose=0):
    '''
    Apply an opening operation to a 3D image with a spherical structuring element of radius `r`.
    If the image is of type `np.uint32`, this function assumes that the image is bitpacked and uses the bitpacked morphology operations.

    Parameters
    ----------
    `image` : numpy.array[Any]
        The image to apply the opening operation to.
    `r` : int
        The radius of the structuring element.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    `image` : numpy.array[Any]
        The image after applying the opening operation.
    '''

    if image.dtype == np.uint32:
        return morph_3d(image, r, [lib_morphology.erode_3d_sphere_bitpacked, lib_morphology.dilate_3d_sphere_bitpacked], verbose)
    else:
        return morph_3d(image, r, [lib_morphology.erode_3d_sphere, lib_morphology.dilate_3d_sphere], verbose)

def plot_middle_planes(tomo, output_dir, prefix, plane_func=lambda x: x, verbose=0):
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
    `plane_func` : function
        The function to apply to the planes before plotting. Default is the identity function.
    `verbose` : int
        The verbosity level. Default is 0.

    Returns
    -------
    None
    '''

    assert len(tomo.shape) == 3

    if verbose >= 2: print(f"Plotting middle planes {prefix} of {tomo.shape} volume to {output_dir}")

    nz, ny, nx = tomo.shape
    names = ['yx', 'zx', 'zy']
    planes = [tomo[nz//2,:,:], tomo[:,ny//2,:], tomo[:,:,nx//2]]
    planes = [plane_func(plane) for plane in planes]

    for name, plane in zip(names, planes):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ims = ax.imshow(plane, interpolation='none')
        fig.colorbar(ims, ax=ax)
        fig.savefig(f"{output_dir}/{prefix}_{name}.pdf", bbox_inches='tight')
        plt.close(fig)

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

def row_normalize(A, r=None):
    '''
    Normalize the rows of a matrix `A` by the vector `r`.
    If `r` is not provided, the rows are normalized by the maximum value in each row.

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

    if r is None:
        return A / (1 + np.max(A, axis=1))[:,na]
    else:
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
    lib_general.normalized_convert(x, result)

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