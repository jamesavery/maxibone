import numpy as np
import sklearn.cluster
from scipy.optimize import minimize
from maxibone.distributions import *

def points_from_histogram(xs,hist, N_points=10000):
    """Draws approximately N_points from xs, distributed by the probability distribution defined by hist

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
    

def kMeans1D(points, n_clusters):
    """Computes k-means for a array of 1D points, sorted in ascending order
    Input:
      points:     An array of single value points
      n_clusters: The number of desired clusters 

    Returns tuple (centroids,mins,maxs), where:
      centroids:  The centroid for each cluster
      mins, maxs: The range of the clusters; Cluster i runs from mins[i] to maxs[i] < mins[i+1]
    """
    X = points.reshape(-1,1)
    kmeans_instance = sklearn.cluster.KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                                    n_clusters=n_clusters, n_init=10, n_jobs=1,
                                    precompute_distances='auto', random_state=None, tol=0.0001, verbose=0)
    
    kmeans_instance.fit(X) #Compute k-means clustering, i.e. train

    centroids = kmeans_instance.cluster_centers_.squeeze()    
    labels    = kmeans_instance.predict(X) #Predict the closest cluster two which each sample in X belongs, i.e. labels

    # Relabel so that 1D centroids are in ascending order
    pi        = np.argsort(centroids) # Index permutation that yields sorted centers
    centroids = centroids[pi];

    mins=np.array([np.min(points[labels==pi[i]]) for i in range(n_clusters)])
    maxs=np.array([np.max(points[labels==pi[i]]) for i in range(n_clusters)])

    return (centroids, mins, maxs)
    
    #JA: What was this for?
    # # Assign each value to the nearest centroid and reshape it to the original image shape
    # input_data_shape = np.choose(labels, centroids).reshape(points.shape)
    # centres=np.array([np.average(points[labels==i]) for i in range(n_clusters)])    
    

    
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
        
    a=[0,0,0,0]                 # Pas på! Virker kun for n_clusters = 4
    b=[0,0,0,0]
    c=[0,0,0,0]
    n = len(x_gauss_para) // 3;
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
        dist_func = gaussians;
    else:
        dist_func = params[3];

    n_clusters = len(x_gauss_para) // 3;
    aa = x_gauss_para[0           :  n_clusters]; # Peak values
    bb = x_gauss_para[n_clusters  :2*n_clusters]; # Exponents
    cc = x_gauss_para[2*n_clusters:3*n_clusters]; # Centers    

    fapprox = np.sum(dist_func(xs,aa,bb,cc), axis=0);
    
    return distance_pos_squared(fapprox,fexact,penalty)


def distributions_from_clusters(xs, rho, n_clusters,
                                N_points=10000, overshoot_penalty=10,
                                distribution_function=gaussians,
                                fitfuncs=fitfuncs_vectorized):

    # First generate N_points from distribution and run k-means 
    points              = points_from_histogram(xs, rho, N_points);
    centers, mins, maxs = kMeans1D(points,n_clusters);

    # Get initial guess for optimization from clusters
    peak_values   = np.interp(centers, xs, rho);             # Exact value of rho[c] for each center c
    cluster_widths = np.minimum(maxs-centers, centers-mins); # Distribution needs to be essentially zero at this width

    params = (xs,rho,overshoot_penalty,distribution_function);
    
    res = minimize(fitfuncs,
                   x0=np.concatenate([np.sqrt(peak_values), np.sqrt(9/cluster_widths), centers]),#np.sqrt(6/cluster_widths), centers]),
                   args=params);

    aa = res.x[0           :  n_clusters]; # Peak values
    bb = res.x[n_clusters  :2*n_clusters]; # Exponents
    cc = res.x[2*n_clusters:3*n_clusters]; # Centers    

    pi        = np.argsort(cc) # Index permutation that yields sorted centers
    aa, bb, cc = aa[pi], bb[pi], cc[pi];
    
    
    return aa,bb,cc

