#! /usr/bin/python3
'''
This script combines the Gaussian and EDT fields into a single field.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')
from config.paths import binary_root, hdf5_root
from lib.py.helpers import commandline_args, generate_cylinder_mask, plot_middle_planes, to_int
import numpy as np
import os

if __name__ == '__main__':
    sample, scale, verbose = commandline_args({
        "sample" : "<required>",
        "scale" : 1,
        "verbose" : 2
    })

    field_dir = f"{binary_root}/fields"
    edt_path = f"{field_dir}/implant-edt/{scale}x"
    gauss_path = f"{field_dir}/implant-gauss/{scale}x"
    output_dir = f"{field_dir}/implant-gauss+edt/{scale}x"
    image_output_dir = f"{hdf5_root}/processed/field-gauss+edt/{scale}x/{sample}"

    os.makedirs(output_dir, exist_ok=True)

    if verbose >= 1: print(f"Loading EDT field from {hdf5_root}/masks/{scale}x/{sample}.npy")
    edt_field = np.load(f"{edt_path}/{sample}.npy")

    nz, ny, nx = edt_field.shape

    if verbose >= 1: print(f"Loading Gaussian field from {hdf5_root}/masks/{scale}x/{sample}.npy")
    gauss_field = np.load(f"{gauss_path}/{sample}.npy")

    cylinder_mask = generate_cylinder_mask(nx)

    if verbose >= 1: print(f"Finding max values")
    edt_max = edt_field.max()
    gauss_max = gauss_field.max()

    if verbose >= 1: print(f"Combining fields")
    combined = gauss_field.astype(np.float32) / gauss_max
    del gauss_field
    combined -= edt_field.astype(np.float32) / edt_max
    del edt_field
    combined += np.abs(combined.min())
    combined /= combined.max()
    combined *= cylinder_mask

    if verbose >= 2:
        plot_middle_planes(combined, image_output_dir, f'{sample}-gauss+edt')

    if verbose >= 1: print(f"Converting to uint16")
    combined = to_int(combined, np.uint16)

    if verbose >= 1: print(f"Applying cylinder mask")
    combined *= cylinder_mask

    if verbose >= 1: print(f"Writing combined field to {output_dir}/{sample}.npy")
    np.save(f"{output_dir}/{sample}.npy", combined)