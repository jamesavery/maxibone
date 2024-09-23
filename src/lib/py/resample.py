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

import numpy.ma as ma

def downsample2x(V, verbose=0):
    '''
    Downsample a 3D image by a factor of 2 in each dimension.
    The image is assumed to be a 3D array with dimensions (Nz, Ny, Nx).
    The output is a 3D array with dimensions (Nz//2, Ny//2, Nx//2).

    Parameters
    ----------
    `V` : numpy.array[Any]
        The input 3D image.
    `verbose` : int
        The verbosity level. This function prints debug information if verbose >= 2 (which is the debug level).

    Returns
    -------
    `result`: numpy.array[Any]
        The downsampled 3D image.
    '''

    NA = np.newaxis

    (Nz, Ny, Nx) = V.shape
    (nz, ny, nx) = (Nz//2, Ny//2, Nx//2)
    xs = np.linspace(-1, 1, nx)[:,NA]
    ys = np.linspace(-1, 1, ny)[NA,:]
    cylinder_mask = (xs*xs + ys*ys) <= 1

    if verbose >= 2:
        print(f"Rescaling from {Nz, Ny, Nx} to {nz, ny, nx}", flush=True)
        print("Extracting S1")

    S1 = V[0:(2*nz):2].astype(np.float32)

    if verbose >= 2:
        print("Extracting S2", flush=True)

    S2 = V[1:(2 * nz+1):2].astype(np.float32)

    if verbose >= 2:
        print(S1.shape, S2.shape)
        print(S1[0:(2*ny):2, 0:(2*nx):2].shape)
        print("Averaging", flush=True)

    s1 = S1[:, 0:2*ny:2, 0:2*nx:2] + S1[:, 0:2*ny:2, 1:(2*nx+1):2] + S1[:, 1:(2*ny+1):2, 0:(2*nx):2] + S1[:, 1:(2*ny+1):2, 1:(2*nx+1):2]
    s2 = S2[:, 0:2*ny:2, 0:2*nx:2] + S2[:, 0:2*ny:2, 1:(2*nx+1):2] + S2[:, 1:(2*ny+1):2, 0:(2*nx):2] + S2[:, 1:(2*ny+1):2, 1:(2*nx+1):2]

    if verbose >= 2:
        print("Storing")

    return (cylinder_mask * (s1+s2) / 8).astype(V.dtype)

def downsample3x(V, verbose=0):
    '''
    Downsample a 3D image by a factor of 3 in each dimension.
    The image is assumed to be a 3D array with dimensions (Nz, Ny, Nx).
    The output is a 3D array with dimensions (Nz//3, Ny//3, Nx//3).

    Parameters
    ----------
    `V` : numpy.array[Any]
        The input 3D image.
    `verbose` : int
        The verbosity level. This function prints debug information if verbose >= 2 (which is the debug level).

    Returns
    -------
    `result`: numpy.array[Any]
        The downsampled 3D image.
    '''

    (Nz, Ny, Nx) = V.shape
    (nz, ny, nx) = (Nz//3, Ny//3, Nx//3)
    xs = np.linspace(-1, 1, nx)[:,np.newaxis]
    ys = np.linspace(-1, 1, ny)[np.newaxis,:]
    cylinder_mask = (xs*xs + ys*ys) <= 1

    if verbose >= 2:
        print(f"Rescaling from {Nz, Ny, Nx} to {nz, ny, nx}", flush=True)
        print("Extracting S1")

    S1 = V[0:(3 * nz):3].astype(np.float32)

    if verbose >= 2:
        print("Extracting S2", flush=True)

    S2 = V[1:(3 * nz+1):3].astype(np.float32)

    if verbose >= 2:
        print("Extracting S3", flush=True)

    S3 = V[2:(3 * nz+2):3].astype(np.float32)

    if verbose >= 2:
        print(S1.shape, S2.shape, S3.shape)
        print(S1[0:(2*ny):2, 0:(2*nx):2].shape)
        print("Averaging", flush=True)

    s1 = S1[:,0:(3*ny+0):3, 0:3*nx:3] +S1[:,0:(3*ny+0):3, 1:(3*nx+1):3]+ S1[:,0:(3*ny+0):3, 2:(3*nx+2):3] + \
         S1[:,1:(3*ny+1):3, 0:3*nx:3] +S1[:,1:(3*ny+1):3, 1:(3*nx+1):3]+ S1[:,1:(3*ny+1):3, 2:(3*nx+2):3] + \
         S1[:,2:(3*ny+2):3, 0:3*nx:3] +S1[:,2:(3*ny+2):3, 1:(3*nx+1):3]+ S1[:,2:(3*ny+2):3, 2:(3*nx+2):3]

    s2 = S2[:,0:(3*ny+0):3, 0:3*nx:3] +S2[:,0:(3*ny+0):3, 1:(3*nx+1):3]+ S2[:,0:(3*ny+0):3, 2:(3*nx+2):3] + \
         S2[:,1:(3*ny+1):3, 0:3*nx:3] +S2[:,1:(3*ny+1):3, 1:(3*nx+1):3]+ S2[:,1:(3*ny+1):3, 2:(3*nx+2):3] + \
         S2[:,2:(3*ny+2):3, 0:3*nx:3] +S2[:,2:(3*ny+2):3, 1:(3*nx+1):3]+ S2[:,2:(3*ny+2):3, 2:(3*nx+2):3]

    s3 = S3[:,0:(3*ny+0):3, 0:3*nx:3] +S3[:,0:(3*ny+0):3, 1:(3*nx+1):3]+ S3[:,0:(3*ny+0):3, 2:(3*nx+2):3] + \
         S3[:,1:(3*ny+1):3, 0:3*nx:3] +S3[:,1:(3*ny+1):3, 1:(3*nx+1):3]+ S3[:,1:(3*ny+1):3, 2:(3*nx+2):3] + \
         S3[:,2:(3*ny+2):3, 0:3*nx:3] +S3[:,2:(3*ny+2):3, 1:(3*nx+1):3]+ S3[:,2:(3*ny+2):3, 2:(3*nx+2):3]

    return (cylinder_mask * (s1 + s2 + s3) / 27).astype(V.dtype)

def sample(image, xs, ys):
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

    yminus, iminus= np.modf(ys - 0.5)
    yplus,  iplus = np.modf(ys + 0.5)
    xminus, jminus= np.modf(xs - 0.5)
    xplus,  jplus = np.modf(xs + 0.5)

    ny, nx = image.shape[-2:]

    LD = (iminus*nx + jminus).astype(np.uint64) # x-,y-
    LU = (iplus*nx  + jminus).astype(np.uint64) # x-,y+
    RD = (iminus*nx +  jplus).astype(np.uint64) # x+,y-
    RU = (iplus*nx  +  jplus).astype(np.uint64) # x+,y+

    I = image.reshape((-1, nx*ny))

    I_polar = (1-xminus) * (1-yminus) * I[:,LD] \
             +(1-xminus) * yplus      * I[:,LU] \
             +xplus      * yplus      * I[:,RD] \
             +xplus      * (1-yminus) * I[:,RU]

    return I_polar.reshape((-1, xs.shape[0], xs.shape[1]))

def cart_to_polar(image, nr, ntheta, r=0, R=None):
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

    midy, midx = np.array(image.shape[-2:]) / 2
    assert (midx == midy)
    if (R == None):
        R = midx

    rs = np.linspace(r, R, nr, endpoint=False)
    thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)

    xs = rs[:,None] * np.cos(thetas)[None,:] + midx
    ys = rs[:,None] * np.sin(thetas)[None,:] + midy

    return sample(image, xs, ys)

def polar_to_cart(polar_image, nx, ny, verbose=0):
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
    `verbose` : int
        The verbosity level. This function prints debug information if verbose >= 2 (which is the debug level).

    Returns
    -------
    `I_cart`: numpy.array[Any]
        The cartesian image
    '''

    R = polar_image.shape[1]
    xs = np.arange(nx) + 0.5 - R
    ys = np.arange(ny) + 0.5 - R

    if verbose >= 2:
        print(nx, ny, R)
        print(xs.min(), xs.max())
        print(ys.min(), ys.max())

    rs      = np.sqrt(xs[None,:]**2 + ys[:,None]**2)
    invalid = rs >= R
    thetas  = np.arctan2(xs[None,:], ys[:,None])

    rs     = ma.masked_array(rs,     mask=invalid)
    thetas = ma.masked_array(thetas, mask=invalid)

    if verbose >= 2:
        print(rs.min(), rs.max())
        print(thetas.min(), thetas.max())

    return sample(polar_image, rs, thetas)
