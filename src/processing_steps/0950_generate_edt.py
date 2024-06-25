import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(sys.path[0]+"/../")
from config.paths import binary_root, hdf5_root
import datetime
import edt
import h5py
import matplotlib.pyplot as plt
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

    output_dir = f"{binary_root}/fields/implant-edt/{scale}x"
    os.makedirs(output_dir, exist_ok=True)

    if verbose >= 1: print(f"Loading implant_solid mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
    with h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5","r") as f:
        implant_solid = f['implant_solid/mask']
        nz,ny,nx = implant_solid.shape
        implant_mask = implant_solid[:]

    if verbose >= 1: print(f"Negating mask")
    implant_mask = ~implant_mask

    if verbose >= 1: print(f"Computing EDT")
    cylinder_mask = generate_cylinder_mask(nx)
    hyperthreading = True
    n_cores = mp.cpu_count() // (2 if hyperthreading else 1) # Only count physical cores
    fedt = edt.edt(implant_mask, parallel=n_cores)
    del implant_mask

    if verbose >= 1: print(f"Applying cylinder mask")
    fedt *= cylinder_mask

    if verbose >= 2:
        print(f"Debug: Writing PNGs of result slices to {output_dir}")
        Image.fromarray(to_int(fedt[nz//2,:,:], np.uint8)).save(f'{output_dir}/{sample}-edt-xy.png')
        Image.fromarray(to_int(fedt[:,ny//2,:], np.uint8)).save(f'{output_dir}/{sample}-edt-xz.png')
        Image.fromarray(to_int(fedt[:,:,nx//2], np.uint8)).save(f'{output_dir}/{sample}-edt-yz.png')

    if verbose >= 1: print(f"Converting to uint16")
    start = datetime.datetime.now()
    fedt = to_int(fedt, np.uint16)
    end = datetime.datetime.now()
    print (f"Conversion took {end-start}")

    if verbose >= 1: print(f"Applying cylinder mask")
    fedt *= cylinder_mask

    if verbose >= 1: print(f"Writing EDT-field to {output_dir}/{sample}.npy")
    np.save(f'{output_dir}/{sample}.npy', fedt)