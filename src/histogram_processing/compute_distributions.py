import numpy as np, matplotlib.pyplot as plt, numpy.linalg as la
from piecewise_cubic import piecewisecubic_matrix, piecewisecubic
from config.paths import hdf5_root_fast as hdf5_root

hist_path = f"{hdf5_root}/processed/histograms/"
sample    = "770c_pag"


f_hist   = np.load(f"{hist_path}/{sample}/bins.npz")
f_labels = np.load(f"{hist_path}/{sample}/bins_relabeled.npz")


plt.imshow(f_labels['x_bins'])
plt.show()
