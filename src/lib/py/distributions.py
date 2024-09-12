'''
This module contains functions for generating Gaussian, exponential, power, and Lorentzian distributions.
'''
import numpy as np
na = np.newaxis

def gaussian(xs, a, b, c):
    '''
    Generates a gaussian distribution. with height `a`, exponent `b`, and centre `c`

    Parameters
    ----------
    `xs` : numpy.array[float]
        The x values to evaluate the gaussian on.
    `a` : float
        The height of the gaussian.
    `b` : float
        The exponent of the gaussian.
    `c` : float
        The centre of the gaussian.

    Returns
    -------
    `G` : numpy.array[float]
        An array of Gaussian function values, I.e., `G[i] = a*np.exp(-b*(xs[i]-c)**2)`
    '''

    return (a*a) * np.exp(-(b*b) * ((xs-c)**2))


def gaussians(xs, abcd):
    '''
    Gaussian distribution with heights `a`, exponents `b`, and centres `c`, implemented with array-operations.

    Parameters
    ----------
    `xs` : numpy.array
        The x values to evaluate the gaussian on.
    `a`, `b`, `c` : numpy arrays
        Arrays of `a`, `b`, and `c` values defining `n` different Gaussians.

    Returns
    -------
    `G` : `n`x`m` numpy array of Gaussian function values.
        I.e., `G[i,j] = a[i]*np.exp(-b[i]*(xs[j]-c[i])**2)`
    '''

    n = len(abcd) // 4
    A = abcd[0  :  n,na]**2 # a^2 to enforce positive values
    B = abcd[n  :2*n,na]**2 # b^2 to enforce positive values
    C = abcd[2*n:3*n,na]    # center is allowed to be negative
    X = xs[na,:]
    G = A * np.exp(-B * ((X-C)**2))
    return G


def exponentials(xs, abcd):
    '''
    Exponential distributions with heights `a**2`, exponents `b**2`, and centres `c`, implemented with array-operations.

    Parameters
    ----------
    `xs` : numpy.array
        Array of `m` `x`-values to evaluate on.
    `a`, `b`, `c` : numpy arrays
        Arrays of `a`, `b`, `c` values defining `n` different exponentials.

    Returns
    -------
    `E` : `n`x`m` numpy array of exponential function values.
        I.e., `E[i,j] = a[i]*np.exp(-b[i]*(xs[j]-c[i]))`.
    '''

    n       = len(abcd) // 4
    A, B, C = abcd[:n,na]**2, abcd[n:2*n,na]**2, abcd[2*n:,na]
    X       = xs[na,:]
    E       = A * np.exp(-B * np.abs(X-C))
    return E


def lorentzians(xs, abcd):
    '''
    Lorentzian distributions with heights `a`, exponents `b`, and centres `c`, implemented with array-operations.

    Parameters
    ----------
    `xs` : numpy.array
        Array of `m` `x`-values to evaluate on.
    `a`, `b`, `c` : numpy arrays
        Arrays of `a`, `b`, `c` values defining `n` different Lorentzians.

    Returns
    -------
    `L` : `n`x`m` numpy array of Lorentzian function values.
        I.e., `L[i,j] = a[i]*b[i]**2 / ((xs[j]-c[i])**2 + b[i]**2)`.
    '''

    n            = len(abcd) // 4
    A, gamma2, C = abcd[:n,na]**2, abcd[n:2*n,na]**2, abcd[2*n:,na]
    X            = xs[na,:]
    L            = A * gamma2 / (((X-C)**2) + gamma2)
    return L

def powers(xs, abcd):
    '''
    Sum of exponential functions with arbitrary power `f(x) = a**2 * e**(-b**2 (x-c)**d)`.

    Parameters
    ----------
    `xs` : numpy.array
        Array of `m` `x`-values to evaluate on.
    `a`, `b`, `c`, `d` : numpy arrays
        Arrays of `a`, `b`, `c`, `d` values defining `n` different power functions.

    Returns
    -------
    `P` : `n`x`m` numpy array of power function values.
        I.e., `P[i,j] = a[i]**2 * np.exp(-b[i]**2 * np.abs(xs[j]-c[i])**d)`.
    '''

    n          = len(abcd) // 4
    A, B, C, D = abcd[:n,na], abcd[n:2*n,na], abcd[2*n:3*n,na], abcd[3*n:,na]
    X          = xs[na,:]
    P          = (A*A) * np.exp(-(B*B) * np.abs(X-C)**(D*D))
    return P


def b_from_width(f, w):
    '''
    Returns the `b` parameter for a given width `w` of a Gaussian, exponential, power, or Lorentzian distribution.

    Parameters
    ----------
    `f` : function
        The distribution function to calculate the `b` parameter for. Should be one of `gaussians`, `exponentials`, `powers`, or `lorentzians`.
    `w` : float
        The width of the distribution.

    Returns
    -------
    `b`: float
        The calculated `b` parameter.
    '''

    if f == gaussians:
        return 6 / w
    if f == exponentials:
        return np.sqrt(2 / w)
    if f == powers:
        return np.sqrt(2 / w)
    if f == lorentzians:
        return w / 4
