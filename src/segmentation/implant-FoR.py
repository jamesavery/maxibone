import h5py, sys, os.path, pathlib, numpy as np, numpy.linalg as la, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root, commandline_args
from pybind_kernels.geometry import center_of_mass, inertia_matrix, integrate_axes, sample_plane
from pybind_kernels.histograms import load_slice
import matplotlib.pyplot as plt
import scipy as sp, scipy.ndimage as ndi, scipy.interpolate as interpolate, scipy.signal as signal
import vedo, vedo.pointcloud as pc
from io_modules.io import update_hdf5

# Hvor skal disse hen?
def sphere(n):
    xs = np.linspace(-1,1,n)
    return (xs[:,NA,NA]**2 + xs[NA,:,NA]**2 + xs[NA,NA,:]**2) <= 1

def coordinate_image(shape):
    Nz,Ny,Nx   = shape
    print(f"Broadcasting coordinates for {shape} image")
    zs, ys, xs = np.broadcast_to(np.arange(Nz)[:,NA,NA],shape),\
                 np.broadcast_to(np.arange(Ny)[NA,:,NA],shape),\
                 np.broadcast_to(np.arange(Nx)[NA,NA,:],shape);
    zyxs = np.stack([zs,ys,xs],axis=-1)
    print(f"Done")
    return zyxs


NA = np.newaxis

sample, scale = commandline_args({"sample":"<required>","scale":8})

print(f"Loading {scale}x implant mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
implant_file = h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5",'r')
implant      = implant_file["implant/mask"][:]
voxel_size   = implant_file["implant"].attrs["voxel_size"]
implant_file.close()

print(f"Loading {scale}x voxels from {binary_root}/voxels/{scale}x/{sample}.uint16")
voxels  = np.fromfile(f"{binary_root}/voxels/{scale}x/{sample}.uint16",dtype=np.uint16).reshape(implant.shape)

sphere_diameter = 2*int(60/(voxel_size)+0.5)+1
if(sphere_diameter>1):
    sph5 = sphere(sphere_diameter)

implant[sphere_diameter//2:-sphere_diameter//2] = ndi.binary_closing(implant,sph5)[sphere_diameter//2:-sphere_diameter//2]


# TODO multiply by scale to get 1x coordinates, or voxel_size to get physical coordinates
Nz,Ny,Nx = implant.shape
cm    = np.array(center_of_mass(implant))                  # in downsampled-voxel index coordinates
IM    = np.array(inertia_matrix(implant,cm)).reshape(3,3)  
ls,E  = la.eigh(IM)

# TODO: Automatic!!!
if sample == "770c_pag":
    E[:,0] *= -1
if sample == "770_pag":
    E[:,2] *= -1
    
ix = np.argsort(np.abs(ls));
ls, E = ls[ix], E[:,ix]
u_vec,v_vec,w_vec = E.T

# Per-nonzero-point stuff
implant_zyxs = np.array(np.nonzero(implant)).T - cm   # Implant points in z,y,x-coordinates (relative to upper-left-left corner, in {scale}x voxel units)
implant_uvws = implant_zyxs @ E                       # Implant points in u,v,w-coordinates (relative to origin cm, in {scale}x voxel units)

w0  = implant_uvws[:,2].min();  # In {scale}x voxel units
w0v = np.array([0,0,w0])        # w-shift to get to center of implant back-plane

implant_UVWs = (implant_uvws - w0v)*voxel_size   # Physical U,V,W-coordinates, relative to actual implant center(*TODO, now center of implant back-plane), in micrometers
implant_Us,implant_Vs,implant_Ws = implant_UVWs.T     # Implant point coordinates
implant_thetas = np.arctan2(implant_Vs,implant_Ws)
implant_rs     = np.sqrt(implant_Vs**2 + implant_Ws**2)
implant_radius = implant_rs.max()                      # Full radius of the implant

theta_integrals, theta_bins = np.histogram(implant_thetas[implant_rs>0.55*implant_radius],200)

theta_from, theta_to = -np.pi/2,np.pi/2 # TODO: Tag fra implant_thetas
theta_center   = theta_from + (theta_to-theta_from)/2  # Angle between cutoffs points forward, + pi points backwards

front_facing_mask = (implant_thetas>theta_from)&(implant_thetas<theta_center)&(implant_rs>0.55*implant_rs.max())
U_integrals, U_bins = np.histogram(implant_Us[front_facing_mask],200)
V_integrals, V_bins = np.histogram(implant_Vs[front_facing_mask],200)
W_integrals, W_bins = np.histogram(implant_Ws[front_facing_mask],200)
Z_integrals, Z_bins = np.histogram(implant_zyxs[:,0],implant.shape[0])

implant_shell = implant_UVWs[implant_rs>0.55*implant_radius]

# Voxel-image-shaped stuff
zyxs = coordinate_image(implant.shape)
uvws = (zyxs - cm) @ E                  # raw voxel-scale relative to center of mass
UVWs = (uvws - w0v) * scale * voxel_size # Micrometer scale relative to almost-center-of-implant-before-sawing-in-half
Us,Vs,Ws = UVWs[...,0], UVWs[...,1], UVWs[...,2]        # Physical image coordinates
thetas, rs = np.arctan2(Vs,Ws), np.sqrt(Vs**2+Ws**2)    

rmaxs = (rs*(implant==True)).reshape(Nz,-1).max(axis=1)[:,NA,NA]

implant_shell_image = implant*(rs >= 0.55*rmaxs)

back_part  = voxels*(Ws<0)
front_part = voxels*(Ws>100)*(rs>=0.7*rmaxs)*(~implant)*(thetas>=theta_from)*(thetas<=theta_to)

output_dir = f"{hdf5_root}/masks/{scale}x/"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
print(f"Saving cut_cylinder_air mask to {output_dir}/{sample}.h5")
update_hdf5(f"{output_dir}/{sample}.h5",
            group_name="cut_cylinder_air",
            datasets={"mask":back_part},
            attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})

print(f"Saving cut_cylinder_bone mask to {output_dir}/{sample}.h5")
update_hdf5(f"{output_dir}/{sample}.h5",
            group_name="cut_cylinder_bone",
            datasets={"mask":front_part},
            attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})

output_dir = f"{hdf5_root}/hdf5-byte/msb/"
print(f"Writing frame-of-reference metadata to {output_dir}/{sample}.h5")
update_hdf5(f"{output_dir}/{sample}.h5",
            group_name="implant-FoR",
            datasets={"UVW":E.T, "center_of_mass":cm*voxel_size},
            attributes={"backplane_W_shift":w0*voxel_size},
            dimensions={"center_of_mass":"xyz in micrometers"},
            chunk_shape=None
)




    
    














def show_FoR():
    vol = vedo.Volume(implant,alpha=[0,0,0.05,0.1])
    u_arrow = vedo.Arrow(cm[::-1],cm[::-1]+1/np.sqrt(ls[0]/ls[2])*100*u_vec[::-1],c='r',s=1)
    v_arrow = vedo.Arrow(cm[::-1],cm[::-1]+1/np.sqrt(ls[1]/ls[2])*100*v_vec[::-1],c='g',s=1)
    w_arrow = vedo.Arrow(cm[::-1],cm[::-1]+100*w_vec[::-1],c='b',s=1)    

    vedo.show([vol,u_arrow,v_arrow,w_arrow])

    
def slope(dx,dy):
    if abs(dx)>abs(dy):
        return dy/dx, 0
    else:
        return dx/dy, 1

def circle_center(p1,p2,p3):
    m12, m13 = (p1+p2)/2, (p1+p3)/2

    s12, ax12 = slope(p2[0]-p1[0], p2[1]-p1[1])
    s13, ax13 = slope(p3[0]-p1[0], p3[1]-p1[1])
