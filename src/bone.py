
#%%

import h5py
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

############################
##### Define functions #####
############################


def combine_bits(hi_bits,lo_bits):

    """ Combine the two 8-bit representations into one 16-bit """

    # x << y : shift left, corresponds to x = x * 2^y
    # x >> y : shift right, corresponds to x = x / 2^y (floats are always rounded down to nearest int)

    combined_bits = np.bitwise_or(np.left_shift(lo_bits, 8), hi_bits) # (low_bits << 8) | high_bits
    
    return combined_bits


def get_slice_from_subvolume(fname):

    """ Get individual slices from full 3D images - with combined bit-planes """

    t_init = time.time()

    with h5py.File(fname, 'r') as f:

        # mask
        # ...

        # voxels_hi
        dset_voxels_hi = f['subvolume']['voxels_hi']
        #test3d = dset_voxels_hi[:10,:,:]

        # voxels_lo
        dset_voxels_lo = f['subvolume']['voxels_lo']

        # get max and min values from single subvolume
        valmin = float(f['metadata'].attrs['valmin'])
        valmax = float(f['metadata'].attrs['valmax'])

        ### Get cross-sections

        # A -> from left to right (screw from "front")
        a = dset_voxels_hi[:,1250,:3473]
        a_low = dset_voxels_lo[:,1250,:3473]
        
        # B -> from top to bottom (screw from "side")
        b = dset_voxels_hi[:,:3473,2000]
        b_low = dset_voxels_lo[:,:3473,2000]

        # C -> from above (screw from (above))
        c = dset_voxels_hi[500,:3473,:3473]
        c_low = dset_voxels_lo[500,:3473,:3473]

        ### Combine the two 8-bit representations into one 16-bit
        slice_A = combine_bits(hi_bits=a, lo_bits=a_low) * (valmin - valmax) + valmin
        slice_B = combine_bits(hi_bits=b, lo_bits=b_low) * (valmin - valmax) + valmin
        slice_C = combine_bits(hi_bits=c, lo_bits=c_low) * (valmin - valmax) + valmin
    
    print('Slice {:s} finished in {:.0f} seconds.'.format(fname[-8], time.time()-t_init))

    return slice_A, slice_B, slice_C


def get_3d_volume_from_subvol(fname, dim):

    """ extract smaller 3d-cube from full subvolume, used for traversing images """

    with h5py.File(fname, 'r') as f:

        vol_hi_bits = f['subvolume']['voxels_hi'][:dim,:dim,:dim]
        vol_lo_bits = f['subvolume']['voxels_lo'][:dim,:dim,:dim]

        volume = combine_bits(hi_bits=vol_hi_bits, lo_bits=vol_lo_bits)

    return volume


def plot_3d_volume(fname, i, j, k, span):

    with h5py.File(fname, 'r') as f:

        vol_hi_bits = f['subvolume']['voxels_hi'][i:i+span, j:j+span, k:k+span]
        vol_lo_bits = f['subvolume']['voxels_lo'][i:i+span, j:j+span, k:k+span]

        volume = combine_bits(hi_bits=vol_hi_bits, lo_bits=vol_lo_bits)

    X, Y, Z = np.mgrid[i:i+span, j:j+span, k:k+span]
    values = volume

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=np.min(volume),
        isomax=np.max(volume),
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
        ))
    fig.show()

    return


#########################
##### Run functions #####
#########################

# ---------------------------------------------------------------------------------------------------------- #

# Get all individual slices from two views
slice1_A, slice1_B, slice1_C = get_slice_from_subvolume('HA_xc520_50kev_1_88mu_implant_769c_001_pag.h5')
slice2_A, slice2_B, slice2_C = get_slice_from_subvolume('HA_xc520_50kev_1_88mu_implant_769c_002_pag.h5')
slice3_A, slice3_B, slice3_C = get_slice_from_subvolume('HA_xc520_50kev_1_88mu_implant_769c_003_pag.h5')
slice4_A, slice4_B, slice4_C = get_slice_from_subvolume('HA_xc520_50kev_1_88mu_implant_769c_004_pag.h5')

# Combine individual slices into full images
full_A = np.concatenate((slice1_A,slice2_A,slice3_A,slice4_A),axis=0)
full_B = np.concatenate((slice1_B,slice2_B,slice3_B,slice4_B),axis=0)
# a stacking of the C view makes no sense

# Plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,10))
ax[0].imshow(full_A)
ax[0].set_title('A ("front view") - slice 1250/3473')
ax[1].imshow(full_B)
ax[1].set_title('B ("side view") - slice 2000/3473')
ax[2].imshow(slice2_C)
ax[2].set_title('C ("above view") - slice 1311/3244')
plt.show()

# ---------------------------------------------------------------------------------------------------------- #

# Plot of 3D-segment from subvolume

#plot_3d_volume(fname='HA_xc520_50kev_1_88mu_implant_769c_003_pag.h5', i=500, j=2000, k=2500, span=100)

# tried out coordinates:
# i=400, j=800, k=1650, span=100

# ---------------------------------------------------------------------------------------------------------- #




#%%