import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(sys.path[0]+"/../")
from config.paths import binary_root, hdf5_root
from lib.py.helpers import commandline_args, generate_cylinder_mask, to_int
import multiprocessing as mp
import numpy as np
import os
from PIL import Image

if __name__ == '__main__':
    sample, scale, verbose = \
        commandline_args({
            "sample" : "<required>",
            "scale" : 1,
            "verbose" : 1
        })

    field_dir = f"{binary_root}/fields"
    edt_path = f"{field_dir}/implant-edt/{scale}x"
    gauss_path = f"{field_dir}/implant-gauss/{scale}x"
    output_dir = f"{field_dir}/implant-gauss+edt/{scale}x"

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
    combined = gauss_field / gauss_max
    del gauss_field
    combined -= edt_field / edt_max
    del edt_field
    combined += np.abs(combined.min())
    combined /= combined.max()
    combined *= cylinder_mask

    if verbose >= 2:
        print(f"Debug: Writing PNGs of result slices to {output_dir}")
        Image.fromarray(to_int(combined[nz//2,:,:], np.uint8)).save(f'{output_dir}/{sample}-gauss+edt-xy.png')
        Image.fromarray(to_int(combined[:,ny//2,:], np.uint8)).save(f'{output_dir}/{sample}-gauss+edt-xz.png')
        Image.fromarray(to_int(combined[:,:,nx//2], np.uint8)).save(f'{output_dir}/{sample}-gauss+edt-yz.png')

    if verbose >= 1: print(f"Converting to uint16")
    combined = to_int(combined, np.uint16)

    if verbose >= 1: print(f"Applying cylinder mask")
    combined *= cylinder_mask

    if verbose >= 1: print(f"Writing combined field to {output_dir}/{sample}.npy")
    np.save(f"{output_dir}/{sample}.npy", combined)