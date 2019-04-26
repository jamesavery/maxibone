
def esrf_2011_vol_to_npy(filename):
    tomo = np.fromfile(filename,dtype=np.float32);
    nz   = len(tomo)/(2048*2048);
    return ma.masked_array(tomo,mask=(tomo==0)).reshape((nz,2048,2048));


def esrf_2011_volslice_to_npy(filename,sl):
    tomo = np.fromfile(filename,dtype=np.float32);
    nz   = len(tomo)/(2048*2048);
    tomo = tomo.reshape((nz,2048,2048));
    tomo = tomo[sl[0,0]:sl[0,1],sl[1,0]:sl[1,1],sl[2,0]:sl[2,1]];
    return ma.masked_array(tomo,mask=(tomo==0));


