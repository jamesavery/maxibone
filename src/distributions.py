import numpy as np

def gaussian(xs, a, b, c):
    """Gaussian distribution with height a, exponent b, and centre c"""
    return (a*a)*np.exp(-(b*b)*((xs-c)**2));


def gaussians(xs, aa, bb, cc):
    """Gaussian distribution with heights aa, exponents bb, and centres cc, implemented with array-operations.
       Input:
          xs:          Array of m x-values to evaluate on
          aa, bb, cc:  Arrays of a,b,c values defining n different Gaussians
       Return:
          G: n x m Gaussian function values, i.e., G[i,j] = aa[i]*np.exp(-bb[i]*(xs[j]-cc[i])**2)
    """
    A, B, C = aa[:,None]**2, bb[:,None]**2, cc[:,None]
    X       = xs[None,:]
    G       =  A*np.exp(-B*((X-C)**2));
    return G


def exponentials(xs, aa, bb, cc):
    """Exponential distributions with heights aa, exponents bb, and centres cc, implemented with array-operations.
       Input:
          xs:          Array of m x-values to evaluate on
          aa, bb, cc:  Arrays of a,b,c values defining n different Gaussians
       Return:
          E: n x m Exponential function values, i.e., E[i,j] = aa[i]*np.exp(-bb[i]*(xs[j]-cc[i]))
    """    
    A, B, C = aa[:,None]**2, bb[:,None]**2, cc[:,None]
    X       = xs[None,:]
    E       =  A*np.exp(-B*np.abs(X-C));
    return E


def lorentzians(xs, aa, bb, cc):
    """Lorentzian distributions with heights aa, exponents bb, and centres cc, implemented with array-operations.
       Input:
          xs:          Array of m x-values to evaluate on
          aa, bb, cc:  Arrays of a,b,c values defining n different Gaussians
       Return:
          L: n x m Lorentzian function values, i.e., L[i,j] = aa[i] * bb[i]**2 / ((xs[j]-cc[i]])**2 + bb[i]**2)
    """    
    A, gamma2, C = aa[:,None]**2, bb[:,None]**2, cc[:,None]
    X       = xs[None,:]
    L       =  A*gamma2/(((X-C)**2) + gamma2)
    return L

