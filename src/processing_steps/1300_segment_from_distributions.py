import os, sys, pathlib, h5py, numpy as np, scipy.ndimage as ndi
sys.path.append(sys.path[0]+"/../")
#import pybind_kernels.histograms as histograms
#import pybind_kernels.label as label
from lib.cpp.cpu.label import material_prob_justonefieldthx
from lib.cpp.cpu.io import write_slice
from config.paths import binary_root, hdf5_root as hdf5_root
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from lib.py.helpers import block_info, commandline_args, load_block
na = np.newaxis
verbose = 1

def load_probabilities(path, group, axes_names, field_names, m):
    try:
        prob_file = h5py.File(path, 'r')
        P_axes   = [prob_file[f'{group}/{name}/P{m}'][:,:] for name in axes_names]
        P_fields = [prob_file[f'{group}/{name}/P{m}'][:,:] for name in field_names]
        prob_file.close()
        return P_axes, P_fields
    except Exception as e:
        print(f"Couldn't load {group}/{axes_names}|{field_names}/P{m} from {path}: {e}")
        sys.exit(-1)

def load_value_ranges(path, group):
    if verbose >= 1: print(f"Reading value_ranges from {group} in {path}\n")
    try:
        f = h5py.File(path, 'r')
        return f[group]['value_ranges'][:].astype(int)
    except Exception as e:
        print(f"Couldn't load {group}/value_ranges from {path}: {e}")
        sys.exit(-1)


def nblocks(size, block_size):
    return (size // block_size) + (1 if size % block_size > 0 else 0)

if __name__ == '__main__':
    sample, block_start, n_blocks, region_mask, group, mask_scale, scheme, verbose = commandline_args({'sample' : '<required>',
                                                                                                       "block_start" : 0,
                                                                                                       "n_blocks" : 0,
                                                                                                       'region_mask' :  'bone_region',
                                                                                                       'group' :  'otsu_separation',
                                                                                                       'mask_scale' :  8,
                                                                                                       'scheme' : "edt", #MIDLERTIDIG
                                                                                                       'verbose' : 1})

    # Iterate over all subvolumes
    bi = block_info(f'{hdf5_root}/hdf5-byte/msb/{sample}.h5', block_size=0, n_blocks=n_blocks, z_offset=block_start)
    Nz, Ny, Nx = bi['dimensions'][:3]
    fz, fy, fx = np.array((Nz, Ny, Nx)) // 2
    axes_names =  []     # ["x", "y", "z", "r"] # For later
    field_names = ["edt"] #,"gauss"] # TODO: Vi bruger kun eet field p.t.

    probs_file = f'{hdf5_root}/processed/probabilities/{sample}.h5'
    for b in tqdm(range(block_start,block_start+bi['n_blocks']), desc='segmenting subvolumes'):
#        if str(region_mask) == "None":
        group_name = f"{group}/bone_region{b}/"            #TODO: two different masks from command line
#        else:
#            group_name = f"{group}/{region_mask}{b}/"

        block_size = bi['subvolume_nzs'][b]
        zstart = bi['subvolume_starts'][b]

        # zstart     = 400        # DEBUGGING
        # block_size = 200        # DEBBUGGING

        zend = zstart + block_size
        fzstart, fzend = zstart // 2, zend // 2

        voxels, fields = load_block(sample, zstart, block_size, region_mask, mask_scale, field_names)
        (vmin, vmax), (fmin, fmax) = load_value_ranges(probs_file, group_name)

        # TODO: Flyt til generering af figurer - hører ikke til her
        try:
            f = h5py.File(f"{hdf5_root}/masks/2x/{sample}.h5","r")
            solid_implant   = f["implant_solid/mask"][zstart//2:zstart//2+block_size//2]
            (nz,ny,nx) = solid_implant.shape
            f.close()

            # solid_implant1x = np.broadcast_to(solid_implant[:,na,:,na,:,na],(nz,2,ny,2,nx,2)).reshape(2*nz,2*ny,2*nx)
            # solid_implant1x = ndi.grey_dilation(solid_implant1x,2)
            # voxels[:2*nz,:2*ny,:2*nx] *= ~solid_implant1x

            # del solid_implant, solid_implant1x
        except Exception as e:
            print(f"Couldn't remove dilated solid implant: {e}")

        for m in [0,1]:
            output_dir  = f'{binary_root}/segmented/{scheme}/P{m}/1x/'
            output_file = f"{output_dir}/{sample}.uint16";
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            P_axes, P_fields = load_probabilities(probs_file, group_name, axes_names, field_names, m)
            n_probs = len(P_axes) + len(P_fields)
            result = np.zeros((zend-zstart,Ny,Nx), dtype=np.uint16)


            material_prob_justonefieldthx(voxels,fields[0],P_fields[0],result,
                                                (vmin,vmax),(fmin,fmax),
                                                (zstart,0,0), (zend,Ny,Nx));

            # label.material_prob(
            #     voxels, fields,
            #     P_axes, 0,#0b1111,
            #     P_fields, 0b01,#0b111,
            #     np.array([1. / n_probs] * n_probs), # Average of all of the probabilities
            #     result,
            #     (vmin, vmax), (fmin, fmax),
            #     (zstart, 0, 0), (zend, sy, sx)
            # )

            if verbose >= 2: print (f'Segmentation has min {result.min()} and max {result.max()}')

            if verbose >= 1: print(f"Writing results from block {b}")
            write_slice(result, zstart*Ny*Nx, output_file)

