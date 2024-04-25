import matplotlib
matplotlib.use('Agg')
import h5py, sys, os.path, pathlib, numpy as np, numpy.linalg as la, tqdm
sys.path.append(sys.path[0]+"/../")
from config.constants import *
from config.paths import hdf5_root, binary_root
from lib.cpp.gpu.geometry import center_of_mass, inertia_matrix, sample_plane
from lib.cpp.gpu.morphology import erode_3d_sphere as erode_3d, dilate_3d_sphere as dilate_3d
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import scipy as sp, scipy.ndimage as ndi, scipy.interpolate as interpolate, scipy.signal as signal
import vedo, vedo.pointcloud as pc
from lib.py.helpers import update_hdf5, update_hdf5_mask, commandline_args
from numpy import array, newaxis as NA

verbose = 1

# Hvor skal disse hen?
def circle_center(p0,p1,p2):
    m1, m2               = (p0+p1)/2, (p0+p2)/2   # Midpoints
    (dx1,dy1), (dx2,dy2) = (p1-p0), (p2-p0)       # Slopes of connecting lines
    n1, n2               = np.array([dy1,-dx1]).T, np.array([dy2,-dx2]).T # Normals perpendicular to connecting lines

    A       = np.array([n1,-n2]).T   # Solve m1 + t1*n1 == m2 + t2*n2   <=> t1*n1 - t2*n2 = m2-m1

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
    I1[r:-r,r:-r,r:-r] = image

    dilate_3d(I1,r,I2)
    erode_3d (I2,r,I1)

    return I1[r:-r,r:-r,r:-r].astype(image.dtype)

def open_3d(image, r):
    (Nz,Ny,Nx) = image.shape
    I1 = np.zeros((Nz+2*r,Ny+2*r,Nx+2*r), dtype=np.uint8)
    I2 = np.zeros((Nz+2*r,Ny+2*r,Nx+2*r), dtype=np.uint8)
    I1[r:-r,r:-r,r:-r] = image

    erode_3d (I1,r,I2)
    dilate_3d(I2,r,I1)

    return I1[r:-r,r:-r,r:-r].astype(image.dtype)

def coordinate_image(shape):
    Nz,Ny,Nx   = shape
    if verbose >= 1: print(f"Broadcasting coordinates for {shape} image")
    zs, ys, xs = np.broadcast_to(np.arange(Nz)[:,NA,NA],shape),\
                 np.broadcast_to(np.arange(Ny)[NA,:,NA],shape),\
                 np.broadcast_to(np.arange(Nx)[NA,NA,:],shape);
    zyxs = np.stack([zs,ys,xs],axis=-1)
    if verbose >= 1: print(f"Done")
    return zyxs

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

def hom_translate(x):
    T = np.eye(4,dtype=float)
    T[0:3,3] = x
    return T

def hom_linear(A):
    M = np.eye(4,dtype=float)
    M[:3,:3] = A
    return M

def homogeneous_transform(xs, M):
    shape = np.array(xs.shape)
    assert(shape[-1] == 3)
    shape[-1] = 4
    hxs = np.empty(shape,dtype=xs.dtype)
    hxs[...,:3] = xs;
    hxs[..., 3]  = 1

    if verbose >= 1: print(hxs.shape, M.shape)
    return hxs @ M.T

def zyx_to_UVWp_transform():
    Tcm   = hom_translate(-cm*voxel_size)
    Muvw  = hom_linear(UVW)
    TW0   = hom_translate((0,0,-w0*voxel_size))
    Tcp   = hom_translate(-cp)
    Muvwp = hom_linear(UVWp)

    return Muvwp @ Tcp @ TW0 @ Muvw @ Tcm

vaxis = {'z':np.array((0,0,1.)), 'y':np.array((0,-1.,0)), 'z2':np.array((0,0,1.))}
daxis = {'z':np.array([-1,1,0]), 'y':np.array([0,0,1]), 'z2':np.array([-1.5,0,0])}

def figure_FoR_UVW(debug=2):
    if debug >= 2:
        vol = vedo.Volume(implant)
        vol.alpha([0,0,0.05,0.2])
        u_arrow = vedo.Arrow(cm[::-1],cm[::-1]+1/np.sqrt(ls[0]/ls[2])*100*u_vec[::-1],c='r',s=0.7)
        v_arrow = vedo.Arrow(cm[::-1],cm[::-1]+1/np.sqrt(ls[1]/ls[2])*100*v_vec[::-1],c='g',s=0.7)
        w_arrow = vedo.Arrow(cm[::-1],cm[::-1]+100*w_vec[::-1],c='b',s=0.7)

        for axis in vaxis.keys():
            pl = vedo.Plotter(offscreen=True, interactive=False,sharecam=False)
            pl.show([vol,u_arrow,v_arrow,w_arrow],camera={
                'pos': np.array((nz/2,ny/2,nx/2)) + 2.5*ny*daxis[axis],
                'focalPoint': (nz/2,ny/2,nx/2),
                'viewup':-vaxis[axis]
            })

            pl.screenshot(f"{image_output_dir}/implant-FoR_UVW-{axis}.png")

        if debug >= 3:
            pl = vedo.Plotter(offscreen=False, interactive=True)
            pl.show([vol,u_arrow,v_arrow,w_arrow],camera={
                'pos': np.array((nz/2,ny/2,nx/2)) + 2.5*ny*daxis[axis],
                'focalPoint': (nz/2,ny/2,nx/2),
                'viewup':-vaxis[axis]
            })

# TODO: Fix lengths (voxel_size times...)
def figure_FoR_UVWp(debug=2):
    if debug >= 2:
        implant_uvwps = homogeneous_transform(implant_zyxs * voxel_size, Muvwp)
        pts = pc.Points(implant_uvwps[:,:3])

        u_arrow = vedo.Arrow([0,0,0],1/np.sqrt(ls[0]/ls[2])*100*np.array([0,0,1]),c='r',s=0.7)
        v_arrow = vedo.Arrow([0,0,0],1/np.sqrt(ls[1]/ls[2])*100*v_vec[::-1],c='g',s=0.7)
        w_arrow = vedo.Arrow([0,0,0],100*w_vec[::-1],c='b',s=0.7)

        pl = vedo.Plotter(offscreen=True, interactive=False, sharecam=False)
        for axis in vaxis.keys():
            pl.show([pts,u_arrow,v_arrow,w_arrow],camera={
                'pos': np.array((nz/2,ny/2,nx/2)) + 2.5*ny*daxis[axis],
                'focalPoint': (nz/2,ny/2,nx/2),
                'viewup':-vaxis[axis]
            })

            pl.screenshot(f"{image_output_dir}/implant-FoR_UVWp-{axis}.png")

        if debug >= 3:
            vedo.show([pts,u_arrow,v_arrow,w_arrow],interactive=True)

def figure_FoR_circle(name,center,v_vec,w_vec,radius,implant_bbox,debug=True):
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    [U_min,U_max,V_min,V_max,W_min,W_max] = implant_bbox

    sample = np.zeros((800,800),dtype=np.float32)
    sample_bbox = (-2905.,2905,-1000,4810.)
    sample_plane(voxels,voxel_size,
                 tuple(center), tuple(v_vec), tuple(w_vec),
                 sample_bbox,sample)

    print (voxel_size, cm, v_vec, w_vec, sample_bbox)
    plt.imshow(sample)
    plt.savefig(f'{image_output_dir}/sample_plane_check.png')
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(sample.T[::-1], extent=sample_bbox,cmap='RdYlBu')

    circle_color = colorConverter.to_rgba('green', alpha=0.2)

    p0 = np.array((0,W_max))
    pb = np.array((0,W_min))
    p1 = np.array((V_min,W_min))
    p2 = np.array((V_max,W_min))

    m1, m2 = (p0+p1)/2, (p0+p2)/2

    ax.add_patch(Circle((0,0), radius*1.01, ec='black',fc=circle_color))
    ax.add_patch(Circle(p1, radius/40, fc='purple'))
    ax.add_patch(Circle(p2, radius/40, fc='purple'))
    ax.add_patch(Circle(p0, radius/40, fc='blue'))
    ax.add_patch(Circle(pb, radius/40, fc=(0,0,1,0.2)))
    ax.add_patch(Circle((0,0),     radius/20, fc='black'))

    ax.add_line(Line2D([p0[0],p1[0]],[p0[1],p1[1]],c='red'))
    ax.add_line(Line2D([p0[0],p2[0]],[p0[1],p2[1]],c='red'))

    ax.add_line(Line2D([m1[0]*1.05,0],[m1[1]*1.05,0],c='green'))
    ax.add_line(Line2D([m2[0]*1.05,0],[m2[1]*1.05,0],c='green'))

    fig.savefig(f"{image_output_dir}/implant-FoR_{name}.png",dpi=300)

    if debug:
        plt.show()

def figure_FoR_profiles(debug):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot((Up_bins[1:]+Up_bins[:-1])/2, Up_integrals);
    fig1.savefig(f"{image_output_dir}/implant-FoR_Up-profile.png")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot((theta_bins[1:]+theta_bins[:-1])/2, theta_integrals)
    fig2.savefig(f"{image_output_dir}/implant-FoR_theta-profile.png")

    if debug:
        plt.show()

def figure_FoR_cylinder(debug=2):
    if debug >= 2:
    #    center_line = vedo.Arrow(C1,C2)
        center_line = vedo.Cylinder((C1+C2)/2,r=implant_radius_voxels/20,height=implant_length_voxels, axis=(C2-C1),alpha=1,c='r')
        cylinder = vedo.Cylinder((C1+C2)/2,r=implant_radius_voxels,height=implant_length_voxels, axis=(C2-C1),alpha=0.3)

        Up_arrow = vedo.Arrow(Cp, UVW2xyz(cp+implant_length*u_prime), c='r')
        Vp_arrow = vedo.Arrow(Cp, UVW2xyz(cp+implant_radius*2*v_prime), c='g')
        Wp_arrow = vedo.Arrow(Cp, UVW2xyz(cp+implant_radius*2*w_prime), c='b')

        vol = vedo.Volume(implant)
        vol.alpha([0,0,0.05,0.1])

        pl = vedo.Plotter(offscreen=True, interactive=False,sharecam=False)
        for axis in vaxis.keys():
            pl.show([vol,center_line,Vp_arrow,Wp_arrow,cylinder],camera={
                'pos': np.array((nz/2,ny/2,nx/2)) + 2.5*ny*daxis[axis],
                'focalPoint': (nz/2,ny/2,nx/2),
                'viewup':-vaxis[axis]
            })

            pl.screenshot(f"{image_output_dir}/implant-FoR_cylinder-{axis}.png")

        if debug >= 3:
            vedo.show([vol,cylinder,Up_arrow,Vp_arrow,Wp_arrow],interactive=True)

def figure_FoR_voxels(name, voxels, debug=2):
    if debug >= 2:
        vol = vedo.Volume(voxels)
        vol.alpha([0,0,0.05,0.1])

        pl  = vedo.Plotter(offscreen=True, interactive=False,sharecam=False)
        for axis in vaxis.keys():
            pl.show([vol],camera={
                'pos': np.array((nz/2,ny/2,nx/2)) + 2.5*ny*daxis[axis],
                'focalPoint': (nz/2,ny/2,nx/2),
                'viewup':-vaxis[axis]
            })
            pl.screenshot(f"{image_output_dir}/implant-FoR_voxels_{name}-{axis}.png")

        if debug >= 3:
            vedo.show([vol],interactive=True)

if __name__ == "__main__":
    sample, scale, verbose = commandline_args({"sample" : "<required>",
                                               "scale" : 8,
                                               "verbose" : 1})

    if(scale<8):
        if verbose >= 1: print(f"Selected scale is {scale}x: This should not be run at high resolution, use scale>=8.")
        #sys.exit(-1)

    ## STEP 0: LOAD MASKS, VOXELS, AND METADATA
    image_output_dir = f"{hdf5_root}/processed/implant-FoR/{sample}/"
    if verbose >= 1: print(f"Storing all debug-images to {image_output_dir}")
    pathlib.Path(image_output_dir).mkdir(parents=True, exist_ok=True)

    if verbose >= 1: print(f"Loading {scale}x implant mask from {hdf5_root}/masks/{scale}x/{sample}.h5")
    implant_file = h5py.File(f"{hdf5_root}/masks/{scale}x/{sample}.h5",'r')
    implant      = implant_file["implant/mask"][:].astype(np.uint8)
    voxel_size   = implant_file["implant"].attrs["voxel_size"]
    implant_file.close()

    if verbose >= 1: print(f"Loading {scale}x voxels from {binary_root}/voxels/{scale}x/{sample}.uint16")
    voxels  = np.fromfile(f"{binary_root}/voxels/{scale}x/{sample}.uint16",dtype=np.uint16).reshape(implant.shape)

    if verbose >= 1: print(f'Plotting sanity images')
    plt.imshow(implant[implant.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy.png'); plt.clf()
    plt.imshow(implant[:,implant.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz.png'); plt.clf()
    plt.imshow(implant[:,:,implant.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz.png'); plt.clf()
    plt.imshow(voxels[voxels.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/voxels-sanity-xy.png'); plt.clf()
    plt.imshow(voxels[:,voxels.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/voxels-sanity-xz.png'); plt.clf()
    plt.imshow(voxels[:,:,voxels.shape[2]//2]); plt.savefig(f'{image_output_dir}/voxels-sanity-yz.png'); plt.clf()
    voxels_without_implant = voxels.copy()
    voxels_without_implant[implant.astype(bool)] = 0
    plt.imshow(voxels_without_implant[voxels.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/voxels-without-implant-xy.png'); plt.clf()
    plt.imshow(voxels_without_implant[:,voxels.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/voxels-without-implant-xz.png'); plt.clf()
    plt.imshow(voxels_without_implant[:,:,voxels.shape[2]//2]); plt.savefig(f'{image_output_dir}/voxels-without-implant-yz.png'); plt.clf()
    del voxels_without_implant

    nz,ny,nx = implant.shape
    if verbose >= 1: print (f'Implant shape is {implant.shape}')

    ### STEP 1: COMPUTE IMPLANT PRINCIPAL AXES FRAME OF REFERENCE
    ## STEP1A: DIAGONALIZE MOMENT OF INTERTIA MATRIX TO GET PRINCIPAL AXES
    cm    = np.array(center_of_mass(implant))                  # in downsampled-voxel index coordinates
    if verbose >= 1: print(f"Center of mass is: {cm}")
    IM    = np.array(inertia_matrix(implant,cm)).reshape(3,3)
    if verbose >= 1: print (f'IM: {IM}')
    ls,E  = la.eigh(IM)

    ## STEP 1B: PRINCIPAL AXES ARE ONLY DEFINED UP TO A SIGN.
    ##          We want U to point in the direction of the screw head, W towards front, V we don't care.
    ##
    ## TODO: Make automatic! Requires additional analysis:
    ##  - u-direction: find implant head, that's up
    ##  - w-direction: find round direction (largest midpoint between theta_integrals peaks).
    ##    w-direction with positive dot-product is the right one
    ##  - v-direction: Either one works, but choose according to right-hand rule
    # TODO old hand held values:
    #if sample == "770c_pag":
    #    E[:,0] *= -1
    #    E[:,1] *= -1
    #if sample == "770c_pag_sub2":
    #    E[:,:] *= -1
    #    Ec = E.copy()
    #    lsc = ls.copy()
    #    E[:,0] = Ec[:,1]
    #    ls[0] = lsc[1]
    #    E[:,1] = Ec[:,0]
    #    ls[1] = ls[0]
    #if sample == "770_pag":
    #    E[:,2] *= -1

    assert(len(ls) == 3) # Automatic only works for 3 eigenvectors. TODO the 3 biggest should be the ones we need?

    # Shuffle
    maxs = np.argmax(np.abs(E), axis=0)
    assert (len(set(maxs)) == 3) # All axes should be different
    ui = np.argwhere(maxs == 0)[0][0]
    vi = np.argwhere(maxs == 2)[0][0]
    wi = np.argwhere(maxs == 1)[0][0]
    shift_is = [ui,vi,wi]
    Ec = E.copy()
    lsc = ls.copy()
    E[:,:] = Ec[:,shift_is]
    ls = lsc[shift_is]

    # Sign flipping
    # Assumes that head is pointing "up" - i.e. negative z, and that x and y should be positive
    E[:,0] *= -np.sign(E[0,0])
    E[:,1] *= np.sign(E[2,1])
    E[:,2] *= np.sign(E[1,2])

    #ix = np.argsort(np.abs(ls));
    #ls, E = ls[ix], E[:,ix]
    UVW = E.T
    u_vec,v_vec,w_vec = UVW

    if verbose >= 1:
        print (f'u_vec {u_vec}')
        print (f'v_vec {v_vec}')
        print (f'w_vec {w_vec}')
    #quit()

    figure_FoR_UVW(verbose)

    ### STEP 2: COMPUTE PHANTOM SCREW GEOMETRY
    #
    implant_zyxs = np.array(np.nonzero(implant)).T - cm   # Implant points in z,y,x-coordinates (relative to upper-left-left corner, in {scale}x voxel units)
    implant_uvws = implant_zyxs @ E                       # Implant points in u,v,w-coordinates (relative to origin cm, in {scale}x voxel units)

    ## 2A: Compute distance along W to back-plane
    w0  = implant_uvws[:,2].min();  # In {scale}x voxel units
    w0v = np.array([0,0,w0])        # w-shift to get to center of implant back-plane


    ## 2B: Transform to backplane-centered coordinates in physical units
    implant_UVWs = (implant_uvws - w0v)*voxel_size     # Physical U,V,W-coordinates, relative to implant back-plane center, in micrometers
    implant_Us,implant_Vs,implant_Ws = implant_UVWs.T  # Implant point coordinates

    ## 2C Segment nonzero points along U-axis injto 100 U-bins and get best circle for each segment
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

    ## 2D: Best circle centers along U forms a helix, due to the winding screw threads. To get the best cylinder,
    ##     we solve for direction vector u_prime so C(U) = C0 + U*u_prime + e(U) with minimal least square residual error e(U)
    ##     where C0 is the mean of the segment circle centers.
    #
    # U*u_prime = C(U) - C0
    #
    # Cs: (N,3)
    # U: N -> (N,3)
    Ub = U_values.reshape(-1,1) #np.broadcast_to(U_values[:,NA], (len(U_values),3))
    C0 = np.mean(Cs,axis=0)
    u_prime, _,_,_ = la.lstsq(Ub, Cs-C0)
    u_prime = u_prime[0]

    UVWp = gramschmidt(u_prime,np.array([0,1,0]),np.array([0,0,1]))
    u_prime, v_prime, w_prime = UVWp # U',V',W' in U,V,W coordinates

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

    figure_FoR_cylinder(verbose)

    ### 3: In the cylinder coordinates, find radii and angle ranges to fill in the "holes" in the implant and make it solid
    ###    (More robust than closing operations, as we don't want to effect the screw threads).

    ## 3A: Transform to implant cylinder coordinates
    implant_UVWps = (implant_UVWs - cp) @ UVWp # We now transform to fully screw aligned coordinates with screw center origin
    implant_Ups, implant_Vps, implant_Wps = implant_UVWps.T

    Up_min, Up_max = implant_Ups.min(), implant_Ups.max()
    Vp_min, Vp_max = implant_Vps.min(), implant_Vps.max()
    Wp_min, Wp_max = implant_Wps.min(), implant_Wps.max()

    #TODO: Local circle figure (instead of showing global fit on local slice, which isn't snug)
    bbox_uvwp = [Up_min,Up_max,Vp_min,Vp_max,Wp_min,Wp_max]
    figure_FoR_circle("prime-circle",Cp*voxel_size,v_vec,w_vec,implant_radius,bbox_uvwp,verbose)

    ## 3B: Profile of radii and angles
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

    # Voxel-image-shaped stuff: This is the part sthat should only be done for coarse resolution (>= 8x)
    zyxs = coordinate_image(implant.shape)
    uvws = (zyxs - cm) @ E                  # raw voxel-scale relative to center of mass
    UVWs = (uvws - w0v) * voxel_size        # Micrometer scale relative to backplane-center
    Us,Vs,Ws = UVWs[...,0], UVWs[...,1], UVWs[...,2]        # UVW physical image coordinates

    UVWps = (UVWs - cp) @ UVWp                # relative to center-of-implant-before-sawing-in-half
    Ups,Vps,Wps = UVWps[...,0], UVWps[...,1], UVWps[...,2]      # U',V',W' physical image coordinates
    thetas, rs = np.arctan2(Vps,Wps), np.sqrt(Vps**2+Wps**2)    # This is the good reference frame for cylindrical coords

    #TODO: rmaxs som funktion af Up
    rmaxs = (rs*(implant==True)).reshape(nz,-1).max(axis=1)[:,NA,NA]

    implant_shell_mask = implant&(rs >= 0.7*rmaxs)
    solid_implant = (implant | (rs < 0.7*rmaxs) & (Ws >= 0))

    solid_quarter = solid_implant & (thetas>=theta_from) & (thetas<=theta_center)
    solid_implant_UVWps   = ((((np.array(np.nonzero(solid_quarter)).T - cm) @ E) - w0v)*voxel_size - cp) @ UVWp
    Up_integrals, Up_bins = np.histogram(solid_implant_UVWps[:,0],200)

    figure_FoR_profiles(verbose)
    figure_FoR_voxels("solid_implant",solid_implant,verbose)

    back_mask  = (Ws<0)
    front_mask = largest_cc_of((Ws>50)&(~solid_implant))#*(thetas>=theta_from)*(thetas<=theta_to)
    plt.imshow(front_mask[front_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-front_mask.png')
    plt.imshow(front_mask[:,front_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-front_mask.png')
    plt.imshow(front_mask[:,:,front_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-front_mask.png')

    # back_part = voxels*back_mask

    front_part = voxels*front_mask
    figure_FoR_voxels("back_part", voxels*back_mask, verbose)
    figure_FoR_voxels("front_part",voxels*front_mask, verbose)

    Cp_zyx = Cp[::-1]*voxel_size

    Muvwp = zyx_to_UVWp_transform()
    if verbose >= 1: print(f"MUvpw = {np.round(Muvwp,2)}")
    if verbose >= 1: print(f"UVW  = {np.round(UVW,2)}")
    if verbose >= 1: print(f"UVWp = {np.round(UVWp,2)}")
    if verbose >= 1: print(f"Cp = {np.round(Cp_zyx,2)}")
    if verbose >= 1: print(f"cp = {np.round(cp,2)}")
    if verbose >= 1: print(f"cm = {np.round(cm,2)}")

    figure_FoR_UVWp(verbose)

    if verbose >= 1: print(f"Physical Cp = {Cp[::-1]*voxel_size}")

    output_dir = f"{hdf5_root}/hdf5-byte/msb/"
    if verbose >= 1: print(f"Writing frame-of-reference metadata to {output_dir}/{sample}.h5")
    update_hdf5(f"{output_dir}/{sample}.h5",
                group_name="implant-FoR",
                datasets={"UVW":UVW,
                          "UVWp": UVWp,
                          "center_of_mass":cm*voxel_size,
                          "center_of_cylinder_UVW": cp,
                          "UVWp_transform": Muvwp,
                          "center_of_cylinder_zyx": Cp_zyx, # Cp is in scaled voxel xyz
                          "bounding_box_UVWp": np.array([[implant_Ups.min(),implant_Ups.max()],
                                                         [implant_Vps.min(),implant_Vps.max()],
                                                         [implant_Wps.min(),implant_Wps.max()]]),
                          "Up_values": (Up_bins[1:]+Up_bins[:-1])/2,
                          "Up_integrals": Up_integrals,
                          "theta_range": np.array([theta_from, theta_to])
                },
                attributes={"backplane_W_shift":w0*voxel_size,
                            "implant_radius": implant_radius
                },
                dimensions={
                    "center_of_mass":"zyx micrometers",
                    "center_of_cylinder_UVW":"UVW in micrometers",
                    "center_of_cylinder_zyx":"zyx in micrometers",
                    "bounding_box_UVWp":"U'V'W' in micrometers",
                    "U_values":"micrometers",
                    "theta_range":"angle around phantom implant center"
                },
                chunk_shape=None
        )

    output_dir = f"{hdf5_root}/masks/{scale}x/"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if verbose >= 1: print(f"Saving implant_solid mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="implant_solid",
                     datasets={"mask":solid_implant},
                     attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plt.imshow(solid_implant[solid_implant.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-solid.png')
    plt.imshow(solid_implant[:,solid_implant.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-solid.png')
    plt.imshow(solid_implant[:,:,solid_implant.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-solid.png')

    if verbose >= 1: print(f"Saving implant_shell mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="implant_shell",
                     datasets={"mask":implant_shell_mask},
                     attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plt.imshow(implant_shell_mask[implant_shell_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-shell.png')
    plt.imshow(implant_shell_mask[:,implant_shell_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-shell.png')
    plt.imshow(implant_shell_mask[:,:,implant_shell_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-shell.png')

    if verbose >= 1: print(f"Saving cut_cylinder_air mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="cut_cylinder_air",
                     datasets={"mask":back_mask},
                     attributes={"sample":sample,"scale":scale,"voxel_size":voxel_size})
    plt.imshow(back_mask[back_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-back.png')
    plt.imshow(back_mask[:,back_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-back.png')
    plt.imshow(back_mask[:,:,back_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-back.png')

    if verbose >= 1: print(f"Saving cut_cylinder_bone mask to {output_dir}/{sample}.h5")
    update_hdf5_mask(f"{output_dir}/{sample}.h5",
                     group_name="cut_cylinder_bone",
                     datasets={"mask":front_mask},
                     attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})
    plt.imshow(front_mask[front_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-front.png')
    plt.imshow(front_mask[:,front_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-front.png')
    plt.imshow(front_mask[:,:,front_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-front.png')

    if verbose >= 1: print(f"Computing bone region")
    hist, bins = np.histogram(front_part, 256)
    hist[0] = 0
    peaks, info = signal.find_peaks(hist,height=0.1*hist.max())


    if verbose >= 1: plt.clf(); plt.plot(bins[1:],hist); plt.savefig(f'{image_output_dir}/bone_histogram.png')
    print (f'peaks: {peaks}')

    try:
        p1, p2 = peaks[np.argsort(info['peak_heights'])[:2]]
        midpoint = int(round((bins[p1]+bins[p2+1])/2)) # p1 is left-edge of p1-bin, p2+1 is right edge of p2-bin
        if verbose >= 1: print(f"p1, p2 = ({p1,bins[p1]}), ({p2,bins[p2]}); midpoint = {midpoint}")

        bone_mask1 = front_part > midpoint
        plt.imshow(bone_mask1[bone_mask1.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-bone1.png')
        plt.imshow(bone_mask1[:,bone_mask1.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-bone1.png')
        plt.imshow(bone_mask1[:,:,bone_mask1.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-bone1.png')

        closing_diameter, opening_diameter = 400, 300           # micrometers
        closing_voxels = 2*int(round(closing_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length
        opening_voxels = 2*int(round(opening_diameter/(2*voxel_size))) + 1 # Scale & ensure odd length

        for i in tqdm.tqdm(range(1),f"Closing with sphere of diameter {closing_diameter} micrometers, {closing_voxels} voxels.\n"):
            bone_region_mask = close_3d(bone_mask1, closing_voxels//2)

        for i in tqdm.tqdm(range(1),f"Opening with sphere of diameter {opening_diameter} micrometers, {opening_voxels} voxels.\n"):
            bone_region_mask &= (~solid_implant).astype(bool) #~open_3d(implant_shell_mask, opening_voxels)
            bone_region_mask = open_3d(bone_region_mask,opening_voxels//2)

        bone_region_mask = largest_cc_of(bone_region_mask)
        plt.imshow(bone_region_mask[bone_region_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-bone.png')
        plt.imshow(bone_region_mask[:,bone_region_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-bone.png')
        plt.imshow(bone_region_mask[:,:,bone_region_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-bone.png')
        voxels_boned = voxels.copy()
        voxels_boned[~bone_region_mask] = 0
        plt.imshow(voxels_boned[voxels_boned.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/voxels-boned-xy.png')
        plt.imshow(voxels_boned[:,voxels_boned.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/voxels-boned-xz.png')
        plt.imshow(voxels_boned[:,:,voxels_boned.shape[2]//2]); plt.savefig(f'{image_output_dir}/voxels-boned-yz.png')

        if verbose >= 1: print(f"Saving bone_region mask to {output_dir}/{sample}.h5")
        update_hdf5_mask(f"{output_dir}/{sample}.h5",
                         group_name="bone_region",
                         datasets={"mask":bone_region_mask},
                         attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})
    except:
        if verbose >= 1: print(f"Wasnt able to separate into resin and bone region. Assuming all is bone region.")
        bone_region_mask = front_mask

        if verbose >= 1: print(f"Saving bone_region mask to {output_dir}/{sample}.h5")
        update_hdf5_mask(f"{output_dir}/{sample}.h5",
                         group_name="bone_region",
                         datasets={"mask":bone_region_mask},
                         attributes={"sample":sample, "scale":scale, "voxel_size":voxel_size})
        plt.imshow(bone_region_mask[bone_region_mask.shape[0]//2,:,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xy-bone.png')
        plt.imshow(bone_region_mask[:,bone_region_mask.shape[1]//2,:]); plt.savefig(f'{image_output_dir}/implant-sanity-xz-bone.png')
        plt.imshow(bone_region_mask[:,:,bone_region_mask.shape[2]//2]); plt.savefig(f'{image_output_dir}/implant-sanity-yz-bone.png')
