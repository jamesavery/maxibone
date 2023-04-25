import find_lines as fl
import numpy as np

bin_idx = 0
cfg_path = 'config.json'
hist_path = '/mnt/data/MAXIBONE/Goats/tomograms/processed/histograms/770c_pag/bins1.npz'
label_path = 'labeled.npy'

f = np.load(hist_path)
config = fl.load_config(cfg_path)
keys = [key for key in f.keys()]
hist = f[keys[bin_idx]]
rng = fl._range(0,hist.shape[1],0,hist.shape[0])
py, px = fl.scatter_peaks(hist, config)
mask = np.zeros(hist.shape, dtype=np.uint8)
mask[px, py] = 255
dilated, eroded = fl.process_closing(mask, config)
labeled, labels = fl.process_contours(eroded, rng, config)
np.save(label_path, labeled)
