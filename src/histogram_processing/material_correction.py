import argparse
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d

def parse_args():
    parser = argparse.ArgumentParser(description="""Corrects the material labeling made by find_ridges, so that the labels correspond across histograms.
    
Example command for running: 
python src/histogram_processing/material_correction.py -o $BONE_DATA/processed/histograms/770c_pag/ $BONE_DATA/processed/histograms/770c_pag/bins-bone_region3.npz $BONE_DATA/processed/histograms/770c_pag/bins-bone_region3_labeled.npz
""")

    parser.add_argument('histograms',
        help='Specifies the histograms file in npz format. Used for computing the different materials.')
    parser.add_argument('labels',
        help='Specifies the labels file in npz format.')
    parser.add_argument('-o', '--output', default='output', type=str,
        help='Specifies the folder to put the resulting images in. If a filename is specified, that will be used.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse the args
    args = parse_args()

    # Load the data
    bins = np.load(args.histograms)
    labs = np.load(args.labels)
    vals = dict()
    for name in bins['axis_names']:
        vals[name] = bins[f'{name}_bins']
    for i, name in enumerate(bins['field_names']):
        vals[name] = bins['field_bins'][i]

    # Compute the material peaks
    sums = np.mean([value.sum(axis=0) for _, value in vals.items()], axis=0)
    smoothed = gaussian_filter1d(sums, 3)
    peaks, _ = signal.find_peaks(smoothed, .01*sums.max())

    # Relabel each histogram in the labeled dictionary.
    relabeled = dict()
    for key, value in labs.items():
        relabeled[key] = np.zeros_like(value)
        for i in set(value.flatten())-{0}:
            # Filter so only the old label remains
            tmp = value.copy()
            tmp[value != i] = 0

            # Compute the mean value
            mean_val = np.argwhere(tmp)[:,-1].mean()

            # Adjust the label to the closest peak # TODO handling material clashing between two lines?
            new_label = np.abs(peaks - mean_val).argmin() + 1
            relabeled[key][tmp != 0] = new_label

    # Save the relabeling
    np.savez(f'{args.output}/{Path(args.histograms).stem}_relabeled.npz', **relabeled)