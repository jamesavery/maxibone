from read_raw import *;
import pyqtgraph as pg;

base = "/diskstation/data/xns/maxibone/esrf_dental_implants_2013/772/raw_data/HA_xc520_50kev_1_88mu_implant_772_001_{:04d}.edf";
base = "/diskstation/data/xns/maxibone/esrf_dental_implants_2013/770c/HA_xc520_50kev_1_88mu_implant_770c_001_pag_{:04d}.edf";

offset = 200;
n = 60;
data = np.zeros((n,3481,3481),dtype=np.uint16);

for i in range(offset,offset+n):
    print(i);
    (m,d) = raw_to_npy(base.format(i));
    data[i-offset] = d;

pg.image(data);
