import numpy as np

def gaussian(xs, a, b, c):
    """Gaussian distribution with height a, exponent b, and centre c"""
    return (a*a)*np.exp(-(b*b)*((xs-c)**2));


def gaussians(xs, abcd):
    """Gaussian distribution with heights a, exponents b, and centres c, implemented with array-operations.
       Input:
          xs:       Array of m x-values to evaluate on
          a, b, c:  Arrays of a,b,c values defining n different Gaussians
       Return:
          G: n x m Gaussian function values, i.e., G[i,j] = aa[i]*np.exp(-b[i]*(xs[j]-c[i])**2)
    """
    n = len(abcd) // 4
    A = abcd[0  :  n,None]**2    # a^2 to enforce positive values
    B = abcd[n  :2*n,None]**2    # b^2 to enforce positive values
    C = abcd[2*n:3*n,None]       # center is allowed to be negative
    
    X       = xs[None,:]
    G       = A*np.exp(-B*((X-C)**2));
    return G


def exponentials(xs, abcd):
    """Exponential distributions with heights a^2, exponents b^2, and centres c, implemented with array-operations.
       Input:
          xs:       Array of m x-values to evaluate on
          a, b, c:  Arrays of a,b,c values defining n different exponentials
       Return:
          E: n x m Exponential function values, i.e., E[i,j] = a[i]*np.exp(-b[i]*(xs[j]-c[i]))
    """
    n = len(abcd) // 4
    A, B, C = abcd[:n,None]**2, abcd[n:2*n,None]**2, abcd[2*n:,None]
    
    X       = xs[None,:]
    E       =  A*np.exp(-B*np.abs(X-C));
    return E


def lorentzians(xs, abcd):
    """Lorentzian distributions with heights a, exponents b, and centres c, implemented with array-operations.
       Input:
          xs:       Array of m x-values to evaluate on
          a, b, c:  Arrays of a,b,c values defining n different Lorentzians
       Return:
          L: n x m Lorentzian function values, i.e., L[i,j] = a[i] * b[i]**2 / ((xs[j]-c[i]])**2 + b[i]**2)
    """
    n = len(abcd) // 4
    
    A, gamma2, C = abcd[:n,None]**2, abcd[n:2*n,None]**2, abcd[2*n:,None]
    X       = xs[None,:]
    L       = A*gamma2/(((X-C)**2) + gamma2)
    return L

def powers(xs, abcd):
    """ Sum of exponential functions with arbitrary power $f(x) = a^2 * e^{-b^2 (x-c)^d }$.
       Input:
          xs:          Array of m x-values to evaluate on
          a, b, c:  Arrays of a,b,c values defining n different Gaussians
       Return:
          E: n x m Function values, i.e., E[i,j] = a[i]*np.exp(-b[i]*(xs[j]-c[i]))
    """    
    n = len(abcd) // 4;

    A, B, C, D = abcd[:n,None]**2, abcd[n:2*n,None]**2, abcd[2*n:3*n,None], abcd[3*n:,None]**2;
    X          = xs[None,:]
    P          = A*np.exp(-(B*np.abs(X-C))**D)
    return P


def b_from_width(f,w):
    if f==gaussians:
        return 6/w;
    if f==exponentials:
        return np.sqrt(2/w);
    if f==powers:
        return np.sqrt(2/w)    
    if f==lorentzians:
        return w/4;


