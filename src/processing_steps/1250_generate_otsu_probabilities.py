#! /usr/bin/python3
'''
This script computes the probabilities of the two distributions in the bins of the histograms using Otsu's method for each row.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.paths import hdf5_root, get_plotting_dir
from lib.cpp.cpu.label import otsu
from lib.py.commandline_args import default_parser
from lib.py.helpers import update_hdf5
from lib.py.piecewise_cubic import piecewisecubic, smooth_fun
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

def apply_otsu(bins, name, plotting, plotting_dir):
    '''
    Apply Otsu's method to separate the two distributions in the bins of the histogram.
    The method is applied to each row of the histogram.

    Parameters
    ----------
    `bins` : numpy.array[uint64]
        The histogram to separate.
    `name` : str
        The name of the histogram. Used for prefixing the output files.
    `plotting` : bool
        Whether to plot debug images.
    `plotting_dir` : str
        The output folder for debug images.

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
    if plotting:
        new_threshes_cubic = piecewisecubic(pc, np.arange(n_rows+1), extrapolation='cubic')
        new_threshes_constant = piecewisecubic(pc, np.arange(n_rows+1), extrapolation='constant')

    # Extract the two new probabilities from the new thresholds
    for i, row in enumerate(bins):
        ma = max(1,np.float32(row.max()))
        P0[i,:int(new_threshes_linear[i])] = row[:int(new_threshes_linear[i])].astype(np.float32) / ma
        P1[i,int(new_threshes_linear[i]):] = row[int(new_threshes_linear[i]):].astype(np.float32) / ma

    # Save control images
    if plotting:
        NA = np.newaxis

        # Plot the two extracted probabilities
        plt.imshow(P0/(P0.max(axis=1)[:,NA]+1))
        plt.savefig(f'{plotting_dir}/{name}_P_otsu_P0.pdf', bbox_inches='tight')
        plt.close()

        plt.imshow(P1/(P1.max(axis=1)[:,NA]+1))
        plt.savefig(f'{plotting_dir}/{name}_P_otsu_P1.pdf', bbox_inches='tight')
        plt.close()

        # Plot the thresholds on top of the original image
        display_cubic = np.empty(bins.shape+(3,),dtype=np.uint8)
        display_cubic[:,:] = ((bins / (bins.max(axis=1)[:,NA]+1)) * 255)[:,:,NA]
        for i, thresh in enumerate(threshes):
            display_cubic[i,int(thresh)-2:int(thresh)+2] = (255,128,0) # otsu is orange
            display_cubic[i,int(new_threshes_cubic[i])-2:int(new_threshes_cubic[i])+2] = (255,0,0) # cubic is red
            display_cubic[i,int(new_threshes_linear[i])-2:int(new_threshes_linear[i])+2] = (0,255,0) # linear is green
            display_cubic[i,int(new_threshes_constant[i])-2:int(new_threshes_constant[i])+2] = (64,128,255) # constant is blue
        Image.fromarray(display_cubic).save(f'{plotting_dir}/P_otsu_thresholds_{name}.png')

    return name, P0, P1, pc, (start, end), threshes, new_threshes_linear

def extract_probabilities(labeled, axes_names, field_names, plotting, plotting_dir, verbose):
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
    `plotting` : bool
        Whether to plot debug images.
    `plotting_dir` : str
        The output folder for debug images.
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `Ps` : list
        A list of the probabilities for each row of the histograms. Each element is a tuple containing the output of `apply_otsu`.
    '''

    axes_rng = tqdm(axes_names, desc='Computing from axes') if verbose >= 1 else axes_names
    Ps = [apply_otsu(labeled[f'{name}_bins'], f'{name}_bins', plotting, plotting_dir) for name in axes_rng]
    field_rng = tqdm(field_names, desc='Computing from fields') if verbose >= 1 else field_names
    for name in field_rng:
        idx = list(labeled['field_names']).index(name)
        bins = labeled['field_bins'][idx]
        Ps.append(apply_otsu(bins, f'field_bins_{name}', plotting, plotting_dir))

    return Ps

def save_probabilities(output_dir, Ps, sample, subbins, value_ranges, verbose):
    '''
    Save the probabilities to an HDF5 file.

    Parameters
    ----------
    `output_dir` : str
        The output directory.
    `Ps` : list
        The probabilities to save.
    `sample` : str
        The name of the sample.
    `subbins` : str
        The number of subbins.
    `value_ranges` : dict
        The value ranges of the histograms
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `None`
    '''

    output_path = f'{output_dir}/{sample}.h5'
    if verbose >= 1: print(f'Saving probabilities to {output_path}')

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
    argparser = default_parser(__doc__)
    argparser.add_argument('subbins', action='store', type=str, default=None, nargs='?',
        help='The name of the subbins to process. Subbins is the postfix to the bins file, if one exists, e.g. "-bone_region". The default is None.')
    argparser.add_argument('--debug-output', action='store', type=str, default=None,
        help='The output folder for debug images.')
    args = argparser.parse_args()

    output_dir = f'{hdf5_root}/processed/probabilities/'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    plotting_dir = get_plotting_dir(args.sample, args.sample_scale)
    if args.plotting:
        pathlib.Path(plotting_dir).mkdir(parents=True, exist_ok=True)

    subbins_str = f'-{args.subbins}' if args.subbins is not None else ''
    input_path = f'{hdf5_root}/processed/histograms/{args.sample}/bins{subbins_str}.npz'
    if args.verbose >= 1: print(f'Loading bins from {input_path}')
    bins = np.load(input_path)
    axes_names = [name.split('_bins')[0] for name in bins.keys() if '_bins' in name and 'field' not in name]
    field_names = bins['field_names']
    Ps = extract_probabilities(bins, axes_names, field_names, args.plotting, plotting_dir, args.verbose)
    save_probabilities(output_dir, Ps, args.sample, args.subbins, bins['value_ranges'], args.verbose)
