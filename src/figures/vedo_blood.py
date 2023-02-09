import numpy as np
import vedo
import h5py as h5


sample = '770c_pag'
with h5.File(f'{sample}.h5', 'r') as f:
    # f["blood"].attrs["voxel_size"]
    blood_mask = f["blood/mask"]
    nz, ny, nx = blood_mask.shape
    sz, ez, sy, ey, sx, ex = 1200, 1400, 600, ny, 0, nx
    voxels = (np.memmap(f"{sample}.uint16",shape=blood_mask.shape,dtype=np.uint16,mode="r") >> 8).astype(np.uint8)
    blood_voxels = blood_mask[sz:ez,sy:ey,sx:ex]*voxels[sz:ez,sy:ey,sx:ex]

print (blood_voxels.max(), blood_voxels.mean())
del voxels
del blood_mask


plt = vedo.Plotter()
bl = vedo.Volume(blood_voxels)
plt.show([bl])
