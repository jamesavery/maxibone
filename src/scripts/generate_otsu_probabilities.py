import sys
sys.path.append(sys.path[0]+"/../")
import cv2
import h5py
import pybind_kernels.histograms as histograms
import numpy as np
from config.paths import binary_root, hdf5_root_fast as hdf5_root, commandline_args
from tqdm import tqdm
import matplotlib.pyplot as plt
from histogram_processing.piecewise_cubic import piecewisecubic, piecewisecubic_matrix, smooth_fun
from PIL import Image
import os
import helper_functions

def apply_otsu(bins, name=None):
    # Set up buffers
    P0 = np.zeros(bins.shape, dtype=np.float32)
    P1 = P0.copy()
    n_rows = bins.shape[0]
    threshes = np.empty(n_rows, dtype=np.uint64)

    # Compute otsu thresholds
    histograms.otsu(bins, threshes, 1)

    # Ignore insignificant rows
    threshes[bins.max(axis=1) < 100] = 0

    ## Leading and trailing 0s and/or invalid floats should not be included in cubic fit
    is_valid = lambda x: not (np.isnan(x) or np.isinf(x) or x == 0)
    start = next((i for i, thresh in enumerate(threshes) if is_valid(thresh)), 0)
    end = next((i for i, thresh in reversed(list(enumerate(threshes))) if is_valid(thresh)), 0)

    # Fit and apply piecewise cubic to smooth the thresholds
    xs = np.arange(start, end, dtype=float)
    pc = smooth_fun(xs, threshes[start:end], 6)
    new_threshes_linear = piecewisecubic(pc, np.arange(n_rows+1), extrapolation='linear')
    if debug:
        new_threshes_cubic = piecewisecubic(pc, np.arange(n_rows+1), extrapolation='cubic')
        new_threshes_constant = piecewisecubic(pc, np.arange(n_rows+1), extrapolation='constant')

    # Extract the two new probabilities from the new thresholds
    for i, row in enumerate(bins):
        ma = max(1,np.float32(row.max()))
        P0[i,:int(new_threshes_linear[i])] = row[:int(new_threshes_linear[i])].astype(np.float32) / ma
        P1[i,int(new_threshes_linear[i]):] = row[int(new_threshes_linear[i]):].astype(np.float32) / ma

    # Save control images
    if debug:
        # Plot the two extracted probabilities
        plt.imshow(P0)
        plt.savefig(f'{debug_output}/P_otsu_c0_{name}.png')
        plt.clf()
        plt.imshow(P1)
        plt.savefig(f'{debug_output}/P_otsu_c1_{name}.png')
        plt.clf()

        # Plot the thresholds on top of the original image
        display_cubic = ((bins / bins.max()) * 255).astype(np.uint8)
        display_cubic = cv2.cvtColor(display_cubic, cv2.COLOR_GRAY2RGB)
        for i, thresh in enumerate(threshes):
            display_cubic[i,int(thresh)-2:int(thresh)+2] = (255,128,0) # otsu is orange
            display_cubic[i,int(new_threshes_cubic[i])-2:int(new_threshes_cubic[i])+2] = (255,0,0) # cubic is red
            display_cubic[i,int(new_threshes_linear[i])-2:int(new_threshes_linear[i])+2] = (0,255,0) # linear is green
            display_cubic[i,int(new_threshes_constant[i])-2:int(new_threshes_constant[i])+2] = (64,128,255) # constant is blue
        Image.fromarray(display_cubic).save(f'{debug_output}/P_otsu_thresholds_{name}.png')

    return name, P0, P1, pc, (start,end), threshes, new_threshes_linear

def extract_probabilities(labeled, axes_names, field_names):
    Ps = [apply_otsu(labeled[f'{name}_bins'], f'{name}_bins') for name in tqdm(axes_names, desc='Computing from axes')]
    for name in tqdm(field_names, desc='Computing from fields'):
        idx = list(labeled['field_names']).index(name)
        bins = labeled['field_bins'][idx]
        Ps.append(apply_otsu(bins, f'field_bins_{name}'))
    return Ps

def save_probabilities(Ps, output_path, value_ranges):
    group_name = 'otsu_seperation'
    helper_functions.update_hdf5(
        output_path,
        group_name = group_name,
        datasets = { 'value_ranges' : value_ranges },
        attributes = {}
    )
    for name, P0, P1, pc, valid_range, threshes, new_threshes in Ps:
        helper_functions.update_hdf5(
            output_path,
            group_name = group_name,
            datasets = {
                f'{name}_c0': P0,
                f'{name}_c1': P1,
                f'{name}_pc_coefs': pc[0],
                f'{name}_pc_borders': pc[1],
                f'{name}_threshes': threshes,
                f'{name}_new_threshes': new_threshes
            },
            attributes = {
                f'{name}_valid_start': valid_range[0],
                f'{name}_valid_end': valid_range[1],
            }
        )

if __name__ == '__main__':
    sample, subbins, debug_output = commandline_args({'sample':'<required>', 'subbins': '<required>', 'debug_output': None})
    output_folder = f'{hdf5_root}/processed/probabilities/'
    debug = True
    debug_output = output_folder if not debug_output else debug_output
    if not os.path.exists(debug_output):
        os.mkdir(debug_output)
    bins = np.load(f'{hdf5_root}/processed/histograms/{sample}/bins-{subbins}.npz')
    axes_names = [name.split('_bins')[0] for name in bins.keys() if '_bins' in name and 'field' not in name]
    field_names = bins['field_names']
    Ps = extract_probabilities(bins, axes_names, field_names)
    save_probabilities(Ps, f'{hdf5_root}/processed/probabilities/{sample}-{subbins}.h5', bins['value_ranges'])