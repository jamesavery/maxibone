import h5py, sys, os.path, pathlib, numpy as np, scipy.ndimage as ndi, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root, commandline_args
from pybind_kernels.geometry import center_of_mass, inertia_matrix, integrate_axes, sample_plane
from pybind_kernels.histograms import load_slice
import matplotlib.pyplot as plt

NA = np.newaxis

sample, scale = commandline_args({"sample":"<required>","scale":8})

implant = np.load(f"{binary_root}/masks/implant/{scale}x/{sample}.npz")["implant_mask"]

cm = center_of_mass(implant)
IM = intertia_matrix(implant,cm)


