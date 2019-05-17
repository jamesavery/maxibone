import numpy as np
import sklearn.cluster
from scipy.optimize import minimize
from maxibone.distributions import *

def points_from_histogram(xs,hist, N_points=10000):
    """Draws approximately N_points from xs, distributed 
    by the probability distribution defined by hist

    Input:
      xs:        Array of n values (can be any type)
      hist:      Array of n integers that count occurrences from xs (hist[i] is the number of events with value xs[i]
      N_points:  The number of points to create.
    Returns:
      points:  Array of N_points values from xs with same distribution of values as original (unknown) points.
    """
    
    coarseness = np.sum(hist) / N_points;
    
    points = np.array([], dtype=float);
    for i in range(len(xs)):
        x = xs[i];
        n = hist[i];                                # Number of voxels with x=xs[i]
        new_points = np.ones(int(n/coarseness))*x;  # Add n/coarseness copies of x
        points     = np.concatenate([points,new_points]);

    return points;

def cluster_peaks(xs,hist,mins,maxs):
    n_clusters     = len(mins)
    cluster_ranges = np.array([np.logical_and(xs>=mins[i], xs<=maxs[i]) for i in range(n_clusters)])
    cluster_starts = np.array([np.argmax(xs == mins[i]) for i in range(n_clusters)])
    local_maxima   = np.array([np.argmax(hist[cluster_ranges[i]]) for i in range(n_clusters) ])
    global_maxima  = cluster_starts + local_maxima

    return global_maxima


def kMeans1D(points, n_clusters, initial_clusters='k-means++'):
    """Computes k-means for a array of 1D points, sorted in ascending order
    Input:
      points:           An array of single value points
      n_clusters:       The number of desired clusters 
      initial_clusters: initial cluster centroid guess (n_clusters x 1 array) or 'k-means++' if first run

    Returns tuple (centroids,mins,maxs), where:
      centroids:  The centroid for each cluster
      mins, maxs: The range of the clusters; Cluster i runs from mins[i] to maxs[i] < mins[i+1]
    """
    X = points.reshape(-1,1)
    if (not isinstance(initial_clusters,str)):
        initial_clusters = initial_clusters.reshape(-1,1)
        n_init = 1
    else:
        n_init = 10
        
    kmeans_instance = sklearn.cluster.KMeans(algorithm='auto', copy_x=True, init=initial_clusters, max_iter=300,
                                    n_clusters=n_clusters, n_init=n_init, n_jobs=1,
                                    precompute_distances='auto', random_state=None, tol=0.0001, verbose=0)
    
    kmeans_instance.fit(X) #Compute k-means clustering, i.e. train

    centroids = kmeans_instance.cluster_centers_.squeeze()
    labels    = kmeans_instance.predict(X) #Predict the closest cluster two which each sample in X belongs, i.e. labels

    # Relabel so that 1D centroids are in ascending order
    pi        = np.argsort(centroids) # Index permutation that yields sorted centers
    centroids = centroids[pi];        # Betyder: centroids[i] = centroids[pi[i]]

    mins=np.array([np.min(points[labels==pi[i]]) for i in range(n_clusters)])
    maxs=np.array([np.max(points[labels==pi[i]]) for i in range(n_clusters)])
    
    return (centroids, mins, maxs)

    
def distance_squared(fapprox,fexact):
    return np.sum((fexact-fapprox)**2);

def distance_pos_squared(fapprox,fexact,penalty):
    diff      = fexact-fapprox;
    overshoot = -diff[diff<0];

    return np.sqrt(np.sum(diff**2)) + penalty*np.sqrt(np.sum(overshoot**2))

def fitfuncs_loops(x_gauss_para, *params):
    xs      = params[0];        
    fexact  = params[1];
    penalty = params[2];

    n = len(x_gauss_para) // 3;    
#    a=[0,0,0,0]                 # Pas på! Virker kun for n_clusters = 4
    a = np.zeros(n);
    b = np.zeros(n);
    c = np.zeros(n);

    for i in range(0, n):
        a[i] = x_gauss_para[    i] 
        b[i] = x_gauss_para[  n+i]
        c[i] = x_gauss_para[2*n+i]
        fapprox = np.zeros(len(xs))

    for i in range(len(xs)):         # Meget langsomt -- se vektoriseret version nedenfor 
        for j in range(n): 
            fapprox[i] = fapprox[i] + gaussian(xs[i],a[j],b[j],c[j])
            
    return distance_pos_squared(fapprox,fexact,penalty)


# Cirka 100 gange hurtigere
def fitfuncs_vectorized(x_gauss_para, *params):
    xs      = params[0];        
    fexact  = params[1];
    penalty = params[2];

    if(len(params) < 4):
        distribution_func = gaussians;
    else:
        distribution_func = params[3];

    n_clusters = len(x_gauss_para) // 3;
    aa = x_gauss_para[0           :  n_clusters]; # Peak values
    bb = x_gauss_para[n_clusters  :2*n_clusters]; # Exponents
    cc = x_gauss_para[2*n_clusters:3*n_clusters]; # Centers    

    fapprox = np.sum(distribution_func(xs,aa,bb,cc), axis=0);
    
    return distance_pos_squared(fapprox,fexact,penalty)

def fitfuncs_vectorized_flat(x_gauss_para, *params):
    xs      = params[0];        
    fexact  = params[1];
    penalty = params[2];

    if(len(params) < 4):
        distribution_func = gaussians;
    else:
        distribution_func = params[3];

    fapprox = np.sum(distribution_func(xs,x_gauss_para), axis=0);
    
    return distance_pos_squared(fapprox,fexact,penalty)


def distributions_from_clusters(xs, rho, n_clusters,
                                N_points=10000,
                                overshoot_penalty=10,
                                distribution_function=gaussians,
                                fitfuncs=fitfuncs_vectorized_flat,
                                initial_clusters='k-means++'
):

    # First generate N_points from distribution and run k-means 
    points              = points_from_histogram(xs, rho, N_points);
    centers, mins, maxs = kMeans1D(points,n_clusters,initial_clusters=initial_clusters);
    maxima              = cluster_peaks(xs,rho,mins,maxs)
    
    # Get initial guess for optimization from clusters
    centroid_values= np.interp(centers, xs, rho);             # Exact value of rho[c] for each center c
    peak_values    = rho[maxima]
    cluster_widths = np.minimum(maxs-centers, centers-mins); # Distribution needs to be essentially zero at this width

    params = (xs,rho,overshoot_penalty,distribution_function);

#    a_guess = np.sqrt(centroid_values)
#    c_guess = centers;
    a_guess = np.sqrt(peak_values)
    b_guess = b_from_width(distribution_function,cluster_widths)
    c_guess = xs[maxima];
    d_guess = np.ones(n_clusters)

#    return np.concatenate([a_guess,b_guess,c_guess,d_guess])
    
    res = minimize(fitfuncs,
                   x0=np.concatenate([a_guess, b_guess, c_guess, d_guess]), 
                   args=params);

    a = res.x[0           :  n_clusters]; # Peak values
    b = res.x[n_clusters  :2*n_clusters]; # Exponents
    c = res.x[2*n_clusters:3*n_clusters]; # Centers
    d = res.x[3*n_clusters:4*n_clusters]; # Powers, if applicable

    pi          = np.argsort(c) # Index permutation that yields sorted centers
    a, b, c, d = a[pi], b[pi], c[pi], d[pi];
    
    
    return np.concatenate([a,b,c,d])

