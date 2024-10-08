#! /usr/bin/python3
'''
This script computes the connected components of the segmented blood mask.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.constants import *
from config.paths import hdf5_root, binary_root, get_plotting_dir
from lib.cpp.cpu_seq.io import load_slice
from lib.py.commandline_args import add_volume, default_parser
from lib.py.helpers import chunk_info, largest_cc_of, plot_middle_planes, update_hdf5
import numpy as np
import tqdm

if __name__ == '__main__':
    argparser = default_parser(__doc__, 0)
    argparser = add_volume(argparser, 'field', 2, 'gauss+edt')
    argparser.add_argument('-m', '--material', action='store', type=int, default=0,
        help='The material to segment. Default is 0, which should be soft tissue.')
    args = argparser.parse_args()

    scales = [32, 16, 8, 4, 2, 1] if args.sample_scale <= 0 else [args.sample_scale]
    bi = chunk_info(f'{hdf5_root}/hdf5-byte/msb/{args.sample}.h5', 1, verbose=args.verbose)
    Nz, Ny, Nx, _ = bi["dimensions"]

    scales_iter = tqdm.tqdm(scales, desc= 'Computing connected components') if args.verbose >= 1 else scales
    for scale in scales_iter:
        data = f'{binary_root}/segmented/{args.field}/P{args.material}/{scale}x/{args.sample}.uint16'
        output_dir = f'{hdf5_root}/masks/{scale}x'
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        plotting_dir = get_plotting_dir(args.sample, scale)
        if args.plotting:
            pathlib.Path(plotting_dir).mkdir(parents=True, exist_ok=True)

        nz, ny, nx = Nz // scale, Ny // scale, Nx // scale
        voxel_size = bi["voxel_size"]*scale

        segmented_u16 = np.zeros((nz,ny,nx), dtype=np.uint16)
        load_slice(segmented_u16, data, (0,0,0), segmented_u16.shape)
        if args.plotting:
            plot_middle_planes(segmented_u16, plotting_dir, f'{args.field}_segmented', verbose=args.verbose)
        segmented = (segmented_u16 > 0).astype(np.uint8)
        del segmented_u16

        mask = largest_cc_of(args.sample, scale, segmented, 'blood', args.plotting, plotting_dir, args.verbose)
        del segmented

        if args.plotting:
            plot_middle_planes(mask, plotting_dir, f'{args.field}_mask', verbose=args.verbose)

        update_hdf5(f"{output_dir}/{args.sample}.h5",
                    group_name=f"blood",
                    datasets={'mask':mask},
                    attributes={
                        'scale': scale,
                        'voxel_size': voxel_size,
                        'sample': args.sample,
                        'name': "blood_mask"
                    })
        del mask
