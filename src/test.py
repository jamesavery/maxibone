import numpy as np, h5py as h5, vedo, geometry, config.paths as paths, scipy.linalg as la, scipy.ndimage as ndi, matplotlib.pyplot as plt

f = h5.File(f"{paths.hdf5_root}/processed/volume_matched/6x/770c_pag.h5","r")
imp = np.array(f["voxels"][:]>140, dtype=np.float32)

cm = geometry.center_of_mass(imp)
M = np.array(geometry.inertia_matrix_serial(imp,cm)).reshape(3,3)

(lam,E) = la.eig(M)


def integrate_axes(img, x0, v_axis, w_axis):
    X0,X1  = np.array([0.,0.,0.])-cm, np.array(imp.shape)-cm    
    vrange = np.dot(X0,v_axis), np.dot(X1,v_axis)
    wrange = np.dot(X0,w_axis), np.dot(X1,w_axis)

    vmin = np.floor(1.2*np.min(vrange))
    wmin = np.floor(1.2*np.min(wrange))

    integral = np.zeros((int(1.2*abs(vrange[1]-vrange[0])),int(1.2*abs(wrange[1]-wrange[0]))),dtype=float)
    geometry.integrate_axes(img,x0,v_axis, w_axis,vmin, wmin,integral);

    return integral;

int_z = integrate_axes(imp, cm, E[:,1], E[:,2])
int_y = integrate_axes(imp, cm, E[:,0], E[:,2])
int_x = integrate_axes(imp, cm, E[:,0], E[:,1])
plt.imshow(int_z)
plt.imshow(int_z)
plt.imshow(int_x)
plt.show()

# vol = vedo.Volume(imp, alpha=[0,0.01])
# au  = vedo.shapes.Arrow(cm,cm+400*E[:,0],c='green')
# av  = vedo.shapes.Arrow(cm,cm+400*E[:,1],c='blue')
# aw  = vedo.shapes.Arrow(cm,cm+400*E[:,2],c='red')
# x0  = vedo.shapes.Sphere(cm,r=15)
# vedo.show([vol,au,av,aw,x0],axes=1)


