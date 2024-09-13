#! /usr/bin/python3
'''
This module provides functions to resample 3D images.
'''
import importlib
cupy_available = importlib.util.find_spec("cupy") is not None
if cupy_available:
    import cupy as np
else:
    import numpy as np
NA = np.newaxis

def downsample2x(V, verbose=False):
    '''
    Downsample a 3D image by a factor of 2 in each dimension.
    The image is assumed to be a 3D array with dimensions (Nz, Ny, Nx).
    The output is a 3D array with dimensions (Nz//2, Ny//2, Nx//2).

    Parameters
    ----------
    `V` : numpy.array[Any]
        The input 3D image.
    `verbose` : bool
        Print debug information.

    Returns
    -------
    `result`: numpy.array[Any]
        The downsampled 3D image.
    '''

    if verbose:
        print(f"Rescaling from {Nz, Ny, Nx} to {nz, ny, nx}", flush=True)
        print("Extracting S1")
    S1 = V[0:(2*nz):2].astype(np.float32)
    if verbose:
        print("Extracting S2", flush=True)

    if verbose:
        print(S1.shape, S2.shape)
        print(S1[0:(2*ny):2, 0:(2*nx):2].shape)
        print("Averaging", flush=True)

    if verbose:
        print("Storing")
def downsample3x(V, verbose=False):
    '''
    Downsample a 3D image by a factor of 3 in each dimension.
    The image is assumed to be a 3D array with dimensions (Nz, Ny, Nx).
    The output is a 3D array with dimensions (Nz//3, Ny//3, Nx//3).

    Parameters
    ----------
    `V` : numpy.array[Any]
        The input 3D image.
    `verbose` : bool
        Print debug information.

    Returns
    -------
    `result`: numpy.array[Any]
        The downsampled 3D image.
    '''


    if verbose:
        print(f"Rescaling from {Nz, Ny, Nx} to {nz, ny, nx}", flush=True)
        print("Extracting S1")

    if verbose:
        print("Extracting S2", flush=True)

    if verbose:
        print("Extracting S3", flush=True)

    if verbose:
        print(S1.shape, S2.shape, S3.shape)
        print(S1[0:(2*ny):2, 0:(2*nx):2].shape)
        print("Averaging", flush=True)
    '''
    Sample an image at the given coordinates.
    The image is assumed to be a 2D array with dimensions (Ny, Nx).

    Parameters
    ----------
    `image` : numpy.array[Any]
        The input 2D image.
    `xs` : numpy.array[float]
        The x-coordinates of the samples.
    `ys` : numpy.array[float]
        The y-coordinates of the samples.

    Returns
    -------
    `I_polar`: numpy.array[Any]
        The sampled image.
    '''


    ny,nx = image.shape[-2:]

    LD = (iminus*nx+jminus).astype(np.uint64) # x-,y-
    LU = (iplus*nx +jminus).astype(np.uint64) # x-,y+
    RD = (iminus*nx+jplus).astype(np.uint64)  # x+,y-
    RU = (iplus*nx +jplus).astype(np.uint64)  # x+,y+

    I = image.reshape((-1,nx*ny))

    I_polar = (1-xminus)*(1-yminus)*I[:,LD] \
             +(1-xminus)*yplus     *I[:,LU] \
             +xplus     *yplus     *I[:,RD] \
             +xplus*(1-yminus)     *I[:,RU]

    return I_polar.reshape((-1,xs.shape[0],xs.shape[1]))

    '''
    Convert a 2D image to polar coordinates.
    The image is assumed to be a 2D array with dimensions (Ny, Nx).

    Parameters
    ----------
    `image` : numpy.array[Any]
        The input 2D image.
    `nr` : int
        The number of radial samples.
    `ntheta` : int
        The number of angular samples.
    `r` : float
        The inner radius of the polar image.
    `R` : float
        The outer radius of the polar image.

    Returns
    -------
    `I_polar`: numpy.array[Any]
        The polar image.
    '''

        R = midx

    rs = np.linspace(r,R,nr,endpoint=False)
    thetas = np.linspace(0,2*np.pi,ntheta,endpoint=False)

    xs = rs[:,None]*np.cos(thetas)[None,:]+midx
    ys = rs[:,None]*np.sin(thetas)[None,:]+midy
    return sample(image,xs,ys)


def polar_to_cart(polar_image, nx, ny, verbose=False):
    '''
    Convert a polar image to cartesian coordinates.
    The image is assumed to be a 2D array with dimensions (Nr, Ntheta).

    Parameters
    ----------
    `polar_image` : numpy.array[Any]
        The input polar image.
    `nx` : int
        The number of x samples.
    `ny` : int
        The number of y samples.

    Returns
    -------
    `I_cart`: numpy.array[Any]
        The cartesian image
    '''

    R = polar_image.shape[1]
    if verbose:
        print(nx, ny, R)
        print(xs.min(), xs.max())
        print(ys.min(), ys.max())

    rs     = np.sqrt(xs[None,:]**2 + ys[:,None]**2)
    invalid = rs>=R
    thetas = np.arctan2(xs[None,:],ys[:,None])

    rs     = ma.masked_array(rs,    mask=invalid)
    thetas = ma.masked_array(thetas,mask=invalid)

    if verbose:
        print(rs.min(), rs.max())
        print(thetas.min(), thetas.max())


