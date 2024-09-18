'''
Global constants for the Osteomorph project.
'''
import numpy as np

implant_threshold = 3
'''The value threshold for finding the implant in the raw float32 data.'''
implant_threshold_u16 = 32000
'''The value threshold for finding the implant in the normalized uint16 data.'''
implant_threshold_u16_novisim = 40000
'''The value threshold for finding the implant in the normalized uint16 data for the novisim samples.'''
global_vmin, global_vmax = -4,12
'''The value ranges of the raw float32 data.'''
implant_threshold_byte = int(((implant_threshold-global_vmin)/(global_vmax-global_vmin))*256 + 1)
'''The value threshold for finding the implant in the byte data.'''

# From Wikipedia (https://en.wikipedia.org/wiki/Osteocyte):
# Osteocytes have a stellate shape, approximately 7 micrometers deep and wide by 15 micrometers in length.[3]
# The cell body varies in size from 5–20 micrometers in diameter and contain 40–60 cell processes per cell,
# [4] with a cell to cell distance between 20–30 micrometers.[3]
osteocyte_minmax_buffer = 1.25
osteocyte_diam_min = 5/osteocyte_minmax_buffer
'''Minimum diameter (with a slack of `config.constants.osteocyte_minmax_buffer`) of an osteocyte in micrometers. Value taken from from https://en.wikipedia.org/wiki/Osteocyte.'''
osteocyte_diam_max = 20*osteocyte_minmax_buffer
'''Minimum diameter (with a slack of `config.constants.osteocyte_minmax_buffer`) of an osteocyte in micrometers. Value taken from from https://en.wikipedia.org/wiki/Osteocyte.'''
osteocyte_lmin = osteocyte_diam_min/np.sqrt(3)
'''Smallest possible side-length in micrometers: assuming box-shaped osteocyte of minimum diameter.'''
osteocyte_lmax = osteocyte_diam_max/np.sqrt(3)
'''Largest possible side-length in micrometers: assuming box-shaped osteocyte of minimum diameter.'''
osteocyte_Vmin = osteocyte_lmin**3
'''Smallest possible volume in micrometers: assuming box-shaped osteocyte of minimum diameter.'''
osteocyte_Vmax = osteocyte_lmax**3
'''Largest possible volume in micrometers: assuming box-shaped osteocyte of minimum diameter.'''