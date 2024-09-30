#! /usr/bin/python3
'''
This script computes the Euclidean Distance Transform (EDT) of the implant mask.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')

from config.paths import binary_root, hdf5_root
import datetime
import edt
import h5py
from lib.py.commandline_args import default_parser
from lib.py.helpers import generate_cylinder_mask, plot_middle_planes, to_int
import multiprocessing as mp
import numpy as np
import os

if __name__ == '__main__':
    args = default_parser(__doc__).parse_args()

    output_dir = f"{binary_root}/fields/implant-edt/{args.sample_scale}x"
    os.makedirs(output_dir, exist_ok=True)
    image_output_dir = f"{hdf5_root}/processed/field-edt/{args.sample_scale}x/{args.sample}"
    if args.verbose >= 2:
        os.makedirs(image_output_dir, exist_ok=True)

    if args.verbose >= 1: print(f"Loading implant_solid mask from {hdf5_root}/masks/{args.sample_scale}x/{args.sample}.h5")
    with h5py.File(f"{hdf5_root}/masks/{args.sample_scale}x/{args.sample}.h5","r") as f:
        implant_solid = f['implant_solid/mask']
        nz,ny,nx = implant_solid.shape
        implant_mask = implant_solid[:]

    if args.verbose >= 1: print(f"Negating mask")
    implant_mask = ~implant_mask

    if args.verbose >= 1: print(f"Computing EDT")
    cylinder_mask = generate_cylinder_mask(nx)
    hyperthreading = True
    n_cores = mp.cpu_count() // (2 if hyperthreading else 1) # Only count physical cores
    fedt = edt.edt(implant_mask, parallel=n_cores)
    del implant_mask

    if args.verbose >= 1: print(f"Applying cylinder mask")
    fedt *= cylinder_mask

    if args.verbose >= 2:
        plot_middle_planes(fedt, image_output_dir, f'{args.sample}-edt', verbose=args.verbose)

    if args.verbose >= 1: print(f"Converting to uint16")
    start = datetime.datetime.now()
    fedt = to_int(fedt, np.uint16)
    end = datetime.datetime.now()
    if args.verbose >= 1: print (f"Conversion took {end-start}")

    if args.verbose >= 1: print(f"Applying cylinder mask")
    fedt *= cylinder_mask

    if args.verbose >= 1: print(f"Writing EDT-field to {output_dir}/{args.sample}.npy")
    np.save(f'{output_dir}/{args.sample}.npy', fedt)