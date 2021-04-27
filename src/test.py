import numpy as np, h5py as h5, vedo, geometry, config.paths as paths, scipy.linalg as la, scipy.ndimage as ndi, matplotlib.pyplot as plt

f = h5.File(f"{paths.hdf5_root}/processed/volume_matched/6x/770c_pag.h5","r")
imp = np.array(f["voxels"][:]>140, dtype=np.float32)


def axis_parameter_bounds(shape, center, axis):
    d     = len(axis)
    signs = np.sign(axis)

    # (0,0,..,0) corner and furthest corner of grid, relative to center
    print(center)
    x0 = -center
    x1 = np.array(shape)[::-1]-center # Data has z,y,x-order, but we keep x,y,z in geometry calc

    xmin = (signs==1)*x0 + (signs==-1)*x1 # minimizes dot(x,axis)
    xmax = (signs==1)*x1 + (signs==-1)*x0 # maximizes dot(x,axis)

    return (np.dot(xmin,axis), np.dot(xmax,axis)), (xmin,xmax)

def integrate_axes(img, cm, v_axis, w_axis):
    (vmin,vmax), (vxmin,vxmax) = axis_parameter_bounds(img.shape, cm, v_axis)
    (wmin,wmax), (wxmin,wxmax) = axis_parameter_bounds(img.shape, cm, w_axis)

    print(f"img.shape = {img.shape}")
    print(f"cm = {cm}; x0 = {-cm}; x1 = {np.array(img.shape)-cm}")
    print(f"v = {v_axis};    (vmin,vmax) = {vmin,vmax};         (vxmin,vxmax) = {vxmin,vxmax}")
    print(f"w = {w_axis};    (wmin,wmax) = {wmin,wmax};         (wxmin,wxmax) = {wxmin,wxmax}")

    integral = np.zeros((int(vmax-vmin+2),int(wmax-wmin+2)),dtype=float)
    geometry.integrate_axes(img,cm,v_axis, w_axis,vmin, wmin,integral);

    return integral;

def integrate_axis(img, x0, v_axis):
    (vmin,vmax), (vxmin,vxmax) = axis_parameter_bounds(img.shape, x0, v_axis)

    print(f"img.shape = {img.shape}")
    print(f"cm = {cm}; x0 = {-cm}; x1 = {np.array(img.shape)-cm}")
    print(f"v = {v_axis};    (vmin,vmax) = {vmin,vmax};         (vxmin,vxmax) = {vxmin,vxmax}")

    integral = np.zeros((int(vmax-vmin+2),1),dtype=float)
    geometry.integrate_axes(img,x0,v_axis, [0,0,0],vmin,0,integral);

    return integral.reshape(-1);

def bounding_volume(mask):
    cm = np.array(geometry.center_of_mass(mask))
    M  = np.array(geometry.inertia_matrix_serial(mask,cm)).reshape(3,3)
    
    (lam,E) = la.eigh(M)

    int_vw = integrate_axes(mask, cm, E[:,1], E[:,2])
    int_uw = integrate_axes(mask, cm, E[:,0], E[:,2])
    int_uv = integrate_axes(mask, cm, E[:,0], E[:,1])
    int_u  = np.sum(int_uv,axis=1)
    int_v  = np.sum(int_uv,axis=0)
    int_w  = np.sum(int_uw,axis=0)

    (umin,umax), _ = axis_parameter_bounds(mask.shape, cm, E[:,0])
    (vmin,vmax), _ = axis_parameter_bounds(mask.shape, cm, E[:,1])
    (wmin,wmax), _ = axis_parameter_bounds(mask.shape, cm, E[:,2])

    u_prefix, u_postfix = np.sum(int_u[0:int(np.ceil(abs(umin)))]>0), np.sum(int_u[int(np.floor(abs(umin))):]>0)
    v_prefix, v_postfix = np.sum(int_v[0:int(np.ceil(abs(vmin)))]>0), np.sum(int_v[int(np.floor(abs(vmin))):]>0)
    w_prefix, w_postfix = np.sum(int_w[0:int(np.ceil(abs(wmin)))]>0), np.sum(int_w[int(np.floor(abs(wmin))):]>0)


    return {
        'u_axis':E[:,0],'v_axis':E[:,1],'w_axis':E[:,2],
        'u_range':(-u_prefix,u_postfix),
        'v_range':(-v_prefix,v_postfix),
        'w_range':(-w_prefix,w_postfix),
        'centre_of_mass':cm
    };

def bounding_cylinder(volume_meta):
    u_axis       = volume_meta["u_axis"]
    u_min, u_max = volume_meta["u_range"]
    v_min, v_max = volume_meta["v_range"]
    w_min, w_max = volume_meta["w_range"]
    cn           = volume_meta["centre_of_mass"]
    
    radius         = np.abs([v_min,v_max,w_min,w_max]).max()
    x_start, x_end = cm+u_min*u_axis, cm+u_max*u_axis
    return x_start, x_end, radius


bound = bounding_volume(imp)


cm = bound['centre_of_mass']
umin, umax = bound['u_range']
vmin, vmax = bound['v_range']
wmin, wmax = bound['w_range']
u_axis, v_axis, w_axis = bound['u_axis'], bound['v_axis'], bound['w_axis']

vol = vedo.Volume(imp, alpha=[0,0.01])
au  = vedo.shapes.Arrow(cm,cm+umax*u_axis,c='green')
av  = vedo.shapes.Arrow(cm,cm+vmax*v_axis,c='blue')
aw  = vedo.shapes.Arrow(cm,cm+wmax*w_axis,c='red')

p0,p1,radius = bounding_cylinder(bound)
cyl = vedo.shapes.Cylinder((p0, p1),
                           r=radius,
                           alpha=0.2)
# x0  = vedo.shapes.Sphere(cm,r=15)
vedo.show([vol,au,av,aw,cyl],axes=1)


