#! /usr/bin/python3
'''
This script computes the probabilities of the two distributions in the bins of the histograms using Otsu's method for each row.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')

from config.paths import hdf5_root as hdf5_root
from lib.cpp.cpu.label import otsu
from lib.py.helpers import commandline_args, update_hdf5
from lib.py.piecewise_cubic import piecewisecubic, smooth_fun
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from PIL import Image
from tqdm import tqdm

def apply_otsu(bins, name=None):
    '''
    Apply Otsu's method to separate the two distributions in the bins of the histogram.
    The method is applied to each row of the histogram.

    Parameters
    ----------
    `bins` : numpy.array[uint64]
        The histogram to separate.
    `name` : str
        The name of the histogram. Used for prefixing the output files.

    Returns
    -------
    `name` : str
        The name of the histogram.
    `P0` : numpy.array[float32]
        The probability of the first distribution.
    `P1` : numpy.array[float32]
        The probability of the second distribution.
    `pc` : tuple
        The coefficients of the piecewise cubic function.
    `valid_range` : tuple
        The range of rows where the cubic fit is valid.
    `threshes` : numpy.array[uint64]
        The Otsu thresholds.
    `new_threshes_linear` : numpy.array[uint64]
        The new thresholds after applying the piecewise cubic function.
    '''

    # Set up buffers
    P0 = np.zeros(bins.shape, dtype=np.float32)
    P1 = P0.copy()
    n_rows = bins.shape[0]
    threshes = np.empty(n_rows, dtype=np.uint64)

    # Compute otsu thresholds
    otsu(bins, threshes, 1)

    # Ignore insignificant rows
    threshes[bins.sum(axis=1) < 100] = 0

    ## Leading and trailing 0s and/or invalid floats should not be included in cubic fit
    is_valid = lambda x: not (np.isnan(x) or np.isinf(x) or x == 0)
    # TODO udvid til at kunne håndtere at der er færre end 2 distributioner
    start = next((i for i, thresh in enumerate(threshes) if is_valid(thresh)), 0)
    end = next((i for i, thresh in reversed(list(enumerate(threshes))) if is_valid(thresh)), 0)

    # Fit and apply piecewise cubic to smooth the thresholds
    xs = np.arange(start, end, dtype=float)
    pc = smooth_fun(xs, threshes[start:end], 12)
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
        NA = np.newaxis

        # Plot the two extracted probabilities
        plt.imshow(P0/(P0.max(axis=1)[:,NA]+1))
        plt.savefig(f'{debug_output}/{name}_P_otsu_P0.png', bbox_inches='tight')
        plt.clf()
        plt.imshow(P1/(P1.max(axis=1)[:,NA]+1))
        plt.savefig(f'{debug_output}/{name}_P_otsu_P1.png', bbox_inches='tight')
        plt.clf()

        # Plot the thresholds on top of the original image
        display_cubic = np.empty(bins.shape+(3,),dtype=np.uint8)
        display_cubic[:,:] = ((bins / (bins.max(axis=1)[:,NA]+1)) * 255)[:,:,NA]
        for i, thresh in enumerate(threshes):
            display_cubic[i,int(thresh)-2:int(thresh)+2] = (255,128,0) # otsu is orange
            display_cubic[i,int(new_threshes_cubic[i])-2:int(new_threshes_cubic[i])+2] = (255,0,0) # cubic is red
            display_cubic[i,int(new_threshes_linear[i])-2:int(new_threshes_linear[i])+2] = (0,255,0) # linear is green
            display_cubic[i,int(new_threshes_constant[i])-2:int(new_threshes_constant[i])+2] = (64,128,255) # constant is blue
        Image.fromarray(display_cubic).save(f'{debug_output}/P_otsu_thresholds_{name}.png')

    return name, P0, P1, pc, (start, end), threshes, new_threshes_linear

def extract_probabilities(labeled, axes_names, field_names):
    '''
    Extract the probabilities for each row of the histograms.

    Parameters
    ----------
    `labeled` : dict
        The labeled histograms.
    `axes_names` : list
        The names of the axes histograms.
    `field_names` : list
        The names of the field histograms.

    Returns
    -------
    `Ps` : list
        A list of the probabilities for each row of the histograms. Each element is a tuple containing the output of `apply_otsu`.
    '''

    Ps = [apply_otsu(labeled[f'{name}_bins'], f'{name}_bins') for name in tqdm(axes_names, desc='Computing from axes')]
    for name in tqdm(field_names, desc='Computing from fields'):
        idx = list(labeled['field_names']).index(name)
        bins = labeled['field_bins'][idx]
        Ps.append(apply_otsu(bins, f'field_bins_{name}'))

    return Ps

def save_probabilities(Ps, sample, subbins, value_ranges):
    '''
    Save the probabilities to an HDF5 file.

    Parameters
    ----------
    `Ps` : list
        The probabilities to save.
    `sample` : str
        The name of the sample.
    `subbins` : str
        The number of subbins.
    `value_ranges` : dict
        The value ranges of the histograms

    Returns
    -------
    `None`
    '''

    output_path = f'{hdf5_root}/processed/probabilities/{sample}.h5'

    update_hdf5(
        output_path,
        group_name = f'otsu_separation/{subbins}',
        datasets = { 'value_ranges' : value_ranges },
        attributes = {}
    )

    for name, P0, P1, pc, valid_range, threshes, new_threshes in Ps:
        update_hdf5(
            output_path,
            group_name = f'otsu_separation/{subbins}/{name}',
            datasets = {
                f'P0': P0,
                f'P1': P1,
                f'pc_coefs': pc[0],
                f'pc_borders': pc[1],
                f'threshes': threshes,
                f'new_threshes': new_threshes
            },
            attributes = {
                f'valid_start': valid_range[0],
                f'valid_end': valid_range[1],
            }
        )

if __name__ == '__main__':
    sample, subbins, debug_output = commandline_args({'sample':'<required>', 'subbins': '<required>', 'debug_output': None})
    output_folder = f'{hdf5_root}/processed/probabilities/'
    debug = True
    debug_output = f'{output_folder}/{sample}' if not debug_output else debug_output

    pathlib.Path(debug_output).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    bins = np.load(f'{hdf5_root}/processed/histograms/{sample}/bins-{subbins}.npz')
    axes_names = [name.split('_bins')[0] for name in bins.keys() if '_bins' in name and 'field' not in name]
    field_names = bins['field_names']
    Ps = extract_probabilities(bins, axes_names, field_names)
    save_probabilities(Ps, sample, subbins, bins['value_ranges'])
