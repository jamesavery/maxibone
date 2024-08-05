import numpy as np

implant_threshold = 3
global_vmin, global_vmax = -4,12
implant_threshold_byte = int(((implant_threshold-global_vmin)/(global_vmax-global_vmin))*256 + 1)

# From Wikipedia:
# Osteocytes have a stellate shape, approximately 7 micrometers deep and wide by 15 micrometers in length.[3]
# The cell body varies in size from 5–20 micrometers in diameter and contain 40–60 cell processes per cell,
# [4] with a cell to cell distance between 20–30 micrometers.[3]
osteocyte_minmax_buffer = 1.25
osteocyte_diam_min = 5/osteocyte_minmax_buffer
osteocyte_diam_max = 20*osteocyte_minmax_buffer
osteocyte_lmin = osteocyte_diam_min/np.sqrt(3)  # Smallest possible side-length: assuming box-shaped osteocyte of minimum diameter
osteocyte_lmax = osteocyte_diam_max/np.sqrt(3)
osteocyte_Vmin = osteocyte_lmin**3
osteocyte_Vmax = osteocyte_lmax**3