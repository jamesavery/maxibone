import sys
sys.path.append(sys.path[0]+"/../")
import numpy as np, matplotlib.pyplot as plt, numpy.linalg as la
from piecewise_cubic import piecewisecubic_matrix, piecewisecubic
from config.paths import commandline_args, hdf5_root as hdf5_root

hist_path = f"{hdf5_root}/processed/histograms/"
sample, suffix, material_id, axis, n_segments = commandline_args({"sample":"<required>",
                                                          "suffix":"<required>",
                                        "material_id": 1,
                                        "axis":"x",
                                        "n_segments": 8})

f_hist   = np.load(f"{hist_path}/{sample}/bins.npz")
f_labels = np.load(f"{hist_path}/{sample}/bins_relabeled.npz")

def material_points(labs,material_id):
    mask = labs==material_id    
    xs, ys = np.argwhere(mask).astype(float).T    
    return xs,ys


# Extract voxel values (as ys) for each coordinate value (as xs)
xs,ys = material_points(f_labels[f'{axis}_bins'],material_id)

plt.imshow(f_labels[f'{axis}_bins'])
plt.show()

plt.scatter(xs,ys)
plt.show()
# Compute smooth piecewise cubic least-squares approximation
pc = smooth_fun(xs,ys,n_segments) # Computes smooth piecewise cubic function (represented in pc)

# We can now evaluate the smooth function for any x-values.
Ys = piecewisecubic(pc,xs)  # Evaluates piecewise-cubic pc in (arbitrary) xs

plt.scatter(xs,ys)
plt.plot(xs,Ys,'C1')
plt.show()
