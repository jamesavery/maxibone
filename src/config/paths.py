'''
Global paths for the Osteomorph project.
'''

import pathlib
import sys

data_root = "/data"
'''The root directory for the data on potentially slow, but large, storage.'''
fast_root = "/data_fast"
'''The root directory for the data on fast(er), but smaller, storage.'''

hdf5_root        = f"{data_root}/MAXIBONE/Goats/tomograms"
'''The root directory for the HDF5 data on potentially slow, but large, storage.'''
hdf5_root_fast   = f"{fast_root}/MAXIBONE/Goats/tomograms"
'''The root directory for the HDF5 data on fast(er), but smaller, storage.'''
binary_root      = f"{hdf5_root}/binary"
'''The root directory for the binary data on potentially slow, but large, storage.'''
binary_root_fast = f"{hdf5_root_fast}/binary"
'''The root directory for the binary data on fast(er), but smaller, storage.'''

esrf_data_local     = f"{hdf5_root}/ESRF/"
'''The root directory for the local copy of ESRF data.'''
esrf_data_sftp      =  "/XNS/XrayImaging/MiG/manjula.esci.nbi.dk.2_localhost/"
'''The root directory for the SFTP copy of ESRF data.'''
esrf_implants_root  = f"{esrf_data_local}/esrf_dental_implants_april_2013/"
'''The root ESRF directory of the implant data.'''
esrf_granules_root  = f"{esrf_data_local}/esrf_dental_granules_july_2012/"
'''The root ESRF directory of the granule data.'''

plotting_root = f'{hdf5_root}/reports'
'''The root directory of where the plots are stored.'''

def get_plotting_dir(sample, scale):
    '''
    This function returns the plotting directory for a given sample and scale.

    Parameters
    ----------
    `sample` : str
        The sample name.
    `scale` : int
        The scale of the image.

    Returns
    -------
    `plotting_dir` : str
        The plotting directory.
    '''

    return f'{plotting_root}/{pathlib.Path(sys.argv[0]).stem}/{sample}/{scale}x'