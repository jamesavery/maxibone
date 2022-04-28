import numpy as np
import vedo

total = np.zeros((2048,2048,2048), np.uint8)
for z in range(2):
    for y in range(2):
        for x in range(2):
            partial = np.load(f'{z}_{y}_{x}.npy')
            total[z*1024:(z+1)*1024,y*1024:(y+1)*1024,x*1024:(x+1)*1024] = partial

vol = vedo.Volume(total[:1024,:,:])
vedo.show(vol)