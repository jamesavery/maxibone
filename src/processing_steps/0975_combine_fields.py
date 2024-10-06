#! /usr/bin/python3
'''
This script combines the Gaussian and EDT fields into a single field.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.paths import binary_root, hdf5_root, get_plotting_dir
from lib.py.commandline_args import default_parser
from lib.py.helpers import generate_cylinder_mask, plot_middle_planes, to_int
import numpy as np

if __name__ == '__main__':
    args = default_parser(__doc__).parse_args()

    field_dir = f"{binary_root}/fields"
    edt_path = f"{field_dir}/implant-edt/{args.sample_scale}x"
    gauss_path = f"{field_dir}/implant-gauss/{args.sample_scale}x"
    output_dir = f"{field_dir}/implant-gauss+edt/{args.sample_scale}x"
    plotting_dir = get_plotting_dir(args.sample, args.sample_scale)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.plotting:
        pathlib.Path(plotting_dir).mkdir(parents=True, exist_ok=True)

    if args.verbose >= 1: print(f"Loading EDT field from {hdf5_root}/masks/{args.sample_scale}x/{args.sample}.npy")
    edt_field = np.load(f"{edt_path}/{args.sample}.npy")

    nz, ny, nx = edt_field.shape

    if args.verbose >= 1: print(f"Loading Gaussian field from {hdf5_root}/masks/{args.sample_scale}x/{args.sample}.npy")
    gauss_field = np.load(f"{gauss_path}/{args.sample}.npy")

    cylinder_mask = generate_cylinder_mask(nx)

    if args.verbose >= 1: print(f"Finding max values")
    edt_max = edt_field.max()
    gauss_max = gauss_field.max()

    if args.verbose >= 1: print(f"Combining fields")
    combined = gauss_field.astype(np.float32) / gauss_max
    del gauss_field
    combined -= edt_field.astype(np.float32) / edt_max
    del edt_field
    combined += np.abs(combined.min())
    combined /= combined.max()
    combined *= cylinder_mask

    if args.plotting:
        plot_middle_planes(combined, plotting_dir, f'gauss+edt', verbose=args.verbose)

    if args.verbose >= 1: print(f"Converting to uint16")
    combined = to_int(combined, np.uint16)

    if args.verbose >= 1: print(f"Applying cylinder mask")
    combined *= cylinder_mask

    if args.verbose >= 1: print(f"Writing combined field to {output_dir}/{args.sample}.npy")
    np.save(f"{output_dir}/{args.sample}.npy", combined)