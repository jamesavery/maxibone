import sys
sys.path.append(sys.path[0]+"/../")
import numpy as np, matplotlib.pyplot as plt, numpy.linalg as la
from piecewise_cubic import piecewisecubic_matrix, piecewisecubic
from config.paths import commandline_args, hdf5_root as hdf5_root

hist_path = f"{hdf5_root}/processed/histograms/"
sample, material_id, axis, n_segments = commandline_args({"sample":"<required>",
                                        "material_id": 1,
                                        "axis":"x",
                                        "n_segments": 8})

f_hist   = np.load(f"{hist_path}/{sample}/bins.npz")
f_labels = np.load(f"{hist_path}/{sample}/bins_relabeled.npz")

def material_points(labs,material_id):
    mask = labs==material_id    
    xs, ys = np.argwhere(mask).astype(float).T    
    return xs,ys

def smooth_fun(xs,ys,n_segments):
    borders = np.linspace(xs.min(), xs.max()+1,n_segments)    

    A, b = piecewisecubic_matrix(xs,ys,borders)
    coefs, residuals, rank, sing = la.lstsq(A,b,rcond=None)    
    pc = coefs, borders

    return pc

# Extract voxel values (as ys) for each coordinate value (as xs)
xs,ys = material_points(f_labels[f'{axis}_bins'],material_id)
# Compute smooth piecewise cubic least-squares approximation
pc = smooth_fun(xs,ys,n_segments)

# We can now evaluate the smooth function for any x-values.
Ys = piecewisecubic(pc,xs)

plt.scatter(xs,ys)
plt.plot(xs,Ys,'C1')
plt.show()
