import h5py, sys, os.path, pathlib, numpy as np, numpy.linalg as la, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root, commandline_args
from pybind_kernels.geometry import center_of_mass, inertia_matrix, integrate_axes, sample_plane
from pybind_kernels.histograms import load_slice, erode_3d_sphere_gpu as erode_3d, dilate_3d_sphere_gpu as dilate_3d
import matplotlib.pyplot as plt
import scipy as sp, scipy.ndimage as ndi, scipy.interpolate as interpolate, scipy.signal as signal
import vedo, vedo.pointcloud as pc
from io_modules.io import update_hdf5
from numpy import array, newaxis as NA

# Hvor skal disse hen?
def circle_center(p0,p1,p2):
    m1, m2               = (p0+p1)/2, (p0+p2)/2   # Midpoints 
    (dx1,dy1), (dx2,dy2) = (p1-p0), (p2-p0)       # Slopes of connecting lines
    n1, n2               = np.array([dy1,-dx1]).T, np.array([dy2,-dx2]).T # Normals perpendicular to connecting lines

    
    A       = np.array([n1,-n2]).T   # Solve m1 + t1*n1 == m2 + t2*n2   <=> t1*n1 - t2*n2 = m2-m1

    # print(f"p0,p1,p2 = {p0,p1,p2}")
    # print(f"m1,m2    = {m1,m2}")
    # print(f"n1,n2    = {n1,n2}")
    # print(f"A        =\n {A}")                
    
    (t1,t2) = la.solve(A, m2-m1)

    c1, c2 = m1+t1*n1, m2+t2*n2  # Center of circle!
    
    assert(np.allclose(c1,c2))

    return c1


def sphere(n):
    xs = np.linspace(-1,1,n)
    return (xs[:,NA,NA]**2 + xs[NA,:,NA]**2 + xs[NA,NA,:]**2) <= 1

def close_3d(image, r):
    (Nz,Ny,Nx) = image.shape
    I1 = np.zeros((Nz+2*r,Ny+2*r,Nx+2*r), dtype=np.uint8)
    I2 = np.zeros((Nz+2*r,Ny+2*r,Nx+2*r), dtype=np.uint8)
    I1[r:-r,r:-r,r:-r] = image;

    dilate_3d(I1,r,I2);
    erode_3d (I2,r,I1)

    return I1[r:-r,r:-r,r:-r].astype(image.dtype)

def open_3d(image, r):
    (Nz,Ny,Nx) = image.shape
    I1 = np.zeros((Nz+2*r,Ny+2*r,Nx+2*r), dtype=np.uint8)
    I2 = np.zeros((Nz+2*r,Ny+2*r,Nx+2*r), dtype=np.uint8)
    I1[r:-r,r:-r,r:-r] = image;

    erode_3d (I1,r,I2);
    dilate_3d(I2,r,I1)

    return I1[r:-r,r:-r,r:-r].astype(image.dtype)


def coordinate_image(shape):
    Nz,Ny,Nx   = shape
    print(f"Broadcasting coordinates for {shape} image")
    zs, ys, xs = np.broadcast_to(np.arange(Nz)[:,NA,NA],shape),\
                 np.broadcast_to(np.arange(Ny)[NA,:,NA],shape),\
                 np.broadcast_to(np.arange(Nx)[NA,NA,:],shape);
    zyxs = np.stack([zs,ys,xs],axis=-1)
    print(f"Done")
    return zyxs




sample, scale = commandline_args({"sample":"<required>","scale":8})

print(f"Loading {scale}x implant mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
implant_file = h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5",'r')
implant      = implant_file["implant/mask"][:]
voxel_size   = implant_file["implant"].attrs["voxel_size"]
implant_file.close()

print(f"Loading {scale}x voxels from {binary_root}/voxels/{scale}x/{sample}.uint16")
voxels  = np.fromfile(f"{binary_root}/voxels/{scale}x/{sample}.uint16",dtype=np.uint16).reshape(implant.shape)


Nz,Ny,Nx = implant.shape
cm    = np.array(center_of_mass(implant))                  # in downsampled-voxel index coordinates
IM    = np.array(inertia_matrix(implant,cm)).reshape(3,3)  
ls,E  = la.eigh(IM)

# TODO: Make automatic! Requires additional analysis:
#  - u-direction: find implant head, that's up
#  - w-direction: find round direction (largest midpoint between theta_integrals peaks).
#    w-direction with positive dot-product is the right one
#  - v-direction: Either one works, but choose according to right-hand rule
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

implant_UVWs = (implant_uvws - w0v)*voxel_size   # Physical U,V,W-coordinates, relative to implant back-plane center, in micrometers
implant_Us,implant_Vs,implant_Ws = implant_UVWs.T     # Implant point coordinates

#TODO: This will be 500 times faster with C segmenting as planned
U_bins = np.linspace(implant_Us.min(), implant_Us.max(), 101)
U_values = (U_bins[1:]+U_bins[:-1])/2
Cs = np.zeros((len(U_values),3),dtype=float)
Rs = np.zeros((len(U_values),),dtype=float)
for i in tqdm.tqdm(range(len(U_bins)-1),"Cylinder centres as fn of U"):
    # Everything is in micrometers
    U0,U1 = U_bins[i], U_bins[i+1]

    slab = implant_UVWs[(implant_Us>=U0) & (implant_Us<=U1)]
    slab_Us, slab_Vs, slab_Ws = slab.T

    W0, W1 = slab_Ws.min(), slab_Ws.max()
    V0, V1 = slab_Vs.min(), slab_Vs.max()

    p0 = np.array([0,W1])
    p1 = np.array([V0,0])
    p2 = np.array([V1,0])

    # Will be way faster to 
    c = circle_center(p0,p1,p2)     # circle center in VW-coordinates
    Cs[i] = np.array([(U0+U1)/2, c[0], c[1]])
    Rs[i] = la.norm(p0-c)

def proj(u,v):                  # Project u onto v
    return (np.dot(u,v)/np.dot(v,v))*v

def gramschmidt(u,v,w):
    vp = v  - proj(v,u)
    wp = w  - proj(w,u) - proj(w,vp)

    return np.array([u/la.norm(u), vp/la.norm(v), wp/la.norm(w)])


def highest_peaks(data,n,height=0.7):
    peaks, info = signal.find_peaks(data,height=height*data.max())
    return peaks[np.argsort(info['peak_heights'])][:n]

def largest_cc_of(mask):
    label, n_features = ndi.label(mask)
    bincnts           = np.bincount(label[label>0],minlength=n_features+1)
    
    largest_cc_ix   = np.argmax(bincnts)
    return (label==largest_cc_ix)    

# Find least squares direction vector u_prime so C(U) = C0 + U*u_prime + e(U) with minimal residual error e(U)
#
# 
#
# U*u_prime = C(U) - C0  
#
# Cs: (N,3)
# U: N -> (N,3)
Ub       = U_values.reshape(-1,1) #np.broadcast_to(U_values[:,NA], (len(U_values),3))
C0 = np.mean(Cs,axis=0)
u_prime, _,_,_ = la.lstsq(Ub, Cs-C0)
u_prime = u_prime[0]

Ep = gramschmidt(u_prime,np.array([0,1,0]),np.array([0,0,1]))
u_prime, v_prime, w_prime = Ep # U',V',W' in U,V,W coordinates

c1 = C0 + implant_Us.min()*u_prime
c2 = C0 + implant_Us.max()*u_prime
cp = (c1+c2)/2

#pts = pc.Points(implant_UVWs)
## Back to xyz FoR
def UVW2xyz(p):
    return ((np.array(p)/voxel_size + w0v) @ E.T + cm)[::-1]

C1, C2, Cp = UVW2xyz(c1), UVW2xyz(c2), UVW2xyz(cp)

implant_length = (implant_Us.max()-implant_Us.min())
implant_radius = Rs.max()

implant_length_voxels = implant_length/voxel_size
implant_radius_voxels = implant_radius/voxel_size

# SHOW CYLINDER FoR
# center_line = vedo.Arrow(C1,C2)
# cylinder = vedo.Cylinder((C1+C2)/2,r=implant_radius_voxels,height=implant_length_voxels, axis=(C2-C1),alpha=0.3)

# Up_arrow = vedo.Arrow(Cp, UVW2xyz(cp+implant_length*u_prime), c='r')
# Vp_arrow = vedo.Arrow(Cp, UVW2xyz(cp+implant_radius*2*v_prime), c='g')
# Wp_arrow = vedo.Arrow(Cp, UVW2xyz(cp+implant_radius*2*w_prime), c='b')

# vedo.show([vol,cylinder,Up_arrow,Vp_arrow,Wp_arrow])

implant_UVWps = (implant_UVWs - cp) @ Ep # We now transform to fully screw aligned coordinates with screw center origin

implant_Ups, implant_Vps, implant_Wps = implant_UVWps.T

implant_thetas = np.arctan2(implant_Vps,implant_Wps)
implant_rs     = np.sqrt(implant_Vps**2 + implant_Wps**2)
implant_radius = implant_rs.max()                      # Full radius of the implant

theta_integrals, theta_bins = np.histogram(implant_thetas[implant_rs>0.55*implant_radius],200)
theta_values = (theta_bins[1:] + theta_bins[:-1])/2

itheta_from, itheta_to = np.argmax(np.gradient(theta_integrals)), np.argmin(np.gradient(theta_integrals))
theta_from,   theta_to = theta_values[itheta_from], theta_values[itheta_to]
theta_center           = theta_from + (theta_to-theta_from)/2  # Angle between cutoffs points forward, + pi points backwards

front_facing_mask = (implant_thetas>theta_from)&(implant_thetas<theta_center)
U_integrals, U_bins = np.histogram(implant_Us[front_facing_mask & (implant_rs>0.6*implant_radius)],200)
V_integrals, V_bins = np.histogram(implant_Vs[front_facing_mask & (implant_rs>0.6*implant_radius)],200)
W_integrals, W_bins = np.histogram(implant_Ws[front_facing_mask & (implant_rs>0.6*implant_radius)],200)

implant_shell = implant_UVWs[implant_rs>0.7*implant_radius]


# Voxel-image-shaped stuff
zyxs = coordinate_image(implant.shape)
uvws = (zyxs - cm) @ E                  # raw voxel-scale relative to center of mass
UVWs = (uvws - w0v) * voxel_size        # Micrometer scale relative to backplane-center
Us,Vs,Ws = UVWs[...,0], UVWs[...,1], UVWs[...,2]        # UVW physical image coordinates 

UVWps = (UVWs - cp) @ Ep                # relative to center-of-implant-before-sawing-in-half
Ups,Vps,Wps = UVWps[...,0], UVWps[...,1], UVWps[...,2]      # U',V',W' physical image coordinates
thetas, rs = np.arctan2(Vps,Wps), np.sqrt(Vps**2+Wps**2)    # This is the good reference frame for cylindrical coords

#TODO: rmaxs som funktion af Up
rmaxs = (rs*(implant==True)).reshape(Nz,-1).max(axis=1)[:,NA,NA]

implant_shell_mask = implant&(rs >= 0.7*rmaxs)
solid_implant = (implant | (rs < 0.7*rmaxs) & (Ws >= 0))

solid_quarter = solid_implant & (thetas>=theta_from) & (thetas<=theta_center)
solid_implant_UVWps   = ((((np.array(np.nonzero(solid_quarter)).T - cm) @ E) - w0v)*voxel_size - cp) @ Ep
Up_integrals, Up_bins = np.histogram(solid_implant_UVWps[:,0],200)

# Show solid implant
# vol = vedo.Volume(voxels*solid_implant)
# vedo.show([vol])

back_mask  = (Ws<0)
front_mask = largest_cc_of((Ws>50)*(~solid_implant))#*(thetas>=theta_from)*(thetas<=theta_to)

# back_part = voxels*back_mask
front_part = voxels*front_mask

output_dir = f"{hdf5_root}/hdf5-byte/msb/"
print(f"Writing frame-of-reference metadata to {output_dir}/{sample}.h5")
update_hdf5(f"{output_dir}/{sample}.h5",
            group_name="implant-FoR",
            datasets={"UVW":E.T,
                      "UVWp": Ep,                      
                      "center_of_mass_xyz":cm,
                      "center_of_cylinder_UVW": cp,
                      "center_of_cylinder_xyz": Cp,
                      "bounding_box_UVWp": np.array([[implant_Ups.min(),implant_Ups.max()],
                                                     [implant_Vps.min(),implant_Vps.max()],
                                                     [implant_Wps.min(),implant_Wps.max()]]),
                      "U_values": U_values,
                      "Up_integrals": Up_integrals,
                      "theta_range": np.array([theta_from, theta_to])
            },
            attributes={"backplane_W_shift":w0*voxel_size,
                        "implant_radius": implant_radius                        
            },
            dimensions={"center_of_mass":"xyz in micrometers"},
            chunk_shape=None
)


output_dir = f"{hdf5_root}/masks/{scale}x/"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
print(f"Saving implant_solid mask to {output_dir}/{sample}.h5")
update_hdf5(f"{output_dir}/{sample}.h5",
            group_name="implant_solid",
            datasets={"mask":solid_implant},
            attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})

print(f"Saving implant_shell mask to {output_dir}/{sample}.h5")
update_hdf5(f"{output_dir}/{sample}.h5",
            group_name="implant_shell",
            datasets={"mask":implant_shell_mask},
            attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})

print(f"Saving cut_cylinder_air mask to {output_dir}/{sample}.h5")
update_hdf5(f"{output_dir}/{sample}.h5",
            group_name="cut_cylinder_air",
            datasets={"mask":back_mask},
            attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})

print(f"Saving cut_cylinder_bone mask to {output_dir}/{sample}.h5")
update_hdf5(f"{output_dir}/{sample}.h5",
            group_name="cut_cylinder_bone",
            datasets={"mask":front_mask},
            attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})


print(f"Computing bone region")
hist, bins = np.histogram(front_part, 256)
hist[0] = 0
peaks, info = signal.find_peaks(hist,height=0.5*hist.max())

p1, p2 = peaks[np.argsort(info['peak_heights'])[:2]]

midpoint = int(round((bins[p1]+bins[p2+1])/2)) # p1 is left-edge of p1-bin, p2+1 is right edge of p2-bin
print(f"p1, p2 = ({p1,bins[p1]}), ({p2,bins[p2]}); midpoint = {midpoint}")

bone_mask1 = front_part > midpoint
                                                                                                                                                                                                                                       
closing_diameter, opening_diameter = 500, 300           # micrometers                                                                                                                                                                   
closing_voxels = 2*int(round(closing_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length
opening_voxels = 2*int(round(opening_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length

for i in tqdm.tqdm(range(1),f"Closing with sphere of diameter {closing_diameter} micrometers, {closing_voxels} voxels.\n"):
    bone_region_mask = close_3d(bone_mask1, closing_voxels//2)

for i in tqdm.tqdm(range(1),f"Opening with sphere of diameter {opening_diameter} micrometers, {opening_voxels} voxels.\n"):
    bone_region_mask &= ~solid_implant #~open_3d(implant_shell_mask, opening_voxels)
    bone_region_mask = open_3d(bone_region_mask,opening_voxels//2)

    
bone_region_mask = largest_cc_of(bone_region_mask)
    
print(f"Saving bone_region mask to {output_dir}/{sample}.h5")
update_hdf5(f"{output_dir}/{sample}.h5",
            group_name="bone_region",
            datasets={"mask":bone_region_mask},
            attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})





    
    














def show_FoR():
    vol = vedo.Volume(implant,alpha=[0,0,0.05,0.1])
    u_arrow = vedo.Arrow(cm[::-1],cm[::-1]+1/np.sqrt(ls[0]/ls[2])*100*u_vec[::-1],c='r',s=1)
    v_arrow = vedo.Arrow(cm[::-1],cm[::-1]+1/np.sqrt(ls[1]/ls[2])*100*v_vec[::-1],c='g',s=1)
    w_arrow = vedo.Arrow(cm[::-1],cm[::-1]+100*w_vec[::-1],c='b',s=1)    

    vedo.show([vol,u_arrow,v_arrow,w_arrow])
