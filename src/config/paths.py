data_root = "/data"
fast_root = "/data_fast"

hdf5_root        = f"{data_root}/MAXIBONE/Goats/tomograms"
hdf5_root_fast   = f"{fast_root}/MAXIBONE/Goats/tomograms"
binary_root      = f"{hdf5_root}/binary"
binary_root_fast = f"{hdf5_root_fast}/binary"

esrf_data_local     = f"{hdf5_root}/ESRF/"
esrf_data_sftp      =  "/XNS/XrayImaging/MiG/manjula.esci.nbi.dk.2_localhost/"
esrf_implants_root  = f"{esrf_data_local}/esrf_dental_implants_april_2013/"
esrf_granules_root  = f"{esrf_data_local}/esrf_dental_granules_july_2012/"