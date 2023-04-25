import numpy as np
import vedo

sample = 'partials/c2_0.npy'
partial = np.load(sample)

vol = vedo.Volume(partial)
vedo.show(vol)