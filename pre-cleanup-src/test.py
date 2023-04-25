import numpy as np, scipy.linalg as la, scipy.ndimage as ndi, matplotlib.pyplot as plt, pathlib, time # Standard modules
import cv2, h5py as h5, vedo, edt           # pip installed modules
import geometry, histograms, config.paths as paths # MAXIBONE Modules 


NA = np.newaxis
N_CPU_THREADS = 16

def axis_parameter_bounds(shape, center, axis):
    d     = len(axis)
    signs = np.sign(axis)

    # (0,0,..,0) corner and furthest corner of grid, relative to center
#    print(center)
    x0 = -np.array(center)
    x1 = np.array(shape)[::-1]-center # Data has z,y,x-order, but we keep x,y,z in geometry calc

    xmin = (signs==1)*x0 + (signs==-1)*x1 # minimizes dot(x,axis)
    xmax = (signs==1)*x1 + (signs==-1)*x0 # maximizes dot(x,axis)

    return (np.dot(xmin,axis), np.dot(xmax,axis)), (xmin,xmax)

def integrate_axes(img, cm, v_axis, w_axis):
    (vmin,vmax), (vxmin,vxmax) = axis_parameter_bounds(img.shape, cm, v_axis)
    (wmin,wmax), (wxmin,wxmax) = axis_parameter_bounds(img.shape, cm, w_axis)

    # print(f"img.shape = {img.shape}")
    # print(f"cm = {cm}; x0 = {-cm}; x1 = {np.array(img.shape)-cm}")
    # print(f"v = {v_axis};    (vmin,vmax) = {vmin,vmax};         (vxmin,vxmax) = {vxmin,vxmax}")
    # print(f"w = {w_axis};    (wmin,wmax) = {wmin,wmax};         (wxmin,wxmax) = {wxmin,wxmax}")

    integral = np.zeros((int(vmax-vmin+2),int(wmax-wmin+2)),dtype=float)
    geometry.integrate_axes(img,cm,v_axis, w_axis,vmin, wmin,integral);

    return integral;

def integrate_axis(img, x0, v_axis):
    (vmin,vmax), (vxmin,vxmax) = axis_parameter_bounds(img.shape, x0, v_axis)

    # print(f"img.shape = {img.shape}")
    # print(f"cm = {cm}; x0 = {-cm}; x1 = {np.array(img.shape)-cm}")
    # print(f"v = {v_axis};    (vmin,vmax) = {vmin,vmax};         (vxmin,vxmax) = {vxmin,vxmax}")

    integral = np.zeros((int(vmax-vmin+2),1),dtype=float)
    geometry.integrate_axes(img,x0,v_axis, [0,0,0],vmin,0,integral);

    return integral.reshape(-1);

def bounding_volume(mask,voxelsize=1.85):
    cm = np.array(geometry.center_of_mass(mask))
    M  = np.array(geometry.inertia_matrix_serial(mask,cm)).reshape(3,3)
    
    lam,E = la.eigh(M)    
    ix = np.argsort(np.abs(lam))
    lam,E = np.array(lam)[ix], np.array(E)[:,ix]

    u_axis, v_axis, w_axis = E[:,0], E[:,1], E[:,2]
    print(lam)
    print("u:",u_axis)
    print("v:",v_axis)
    print("w:",w_axis)
    
    int_vw = integrate_axes(mask, cm, v_axis, w_axis)
    int_uw = integrate_axes(mask, cm, u_axis, w_axis)
    int_uv = integrate_axes(mask, cm, u_axis, v_axis)
    int_u  = np.sum(int_uv,axis=1)
    int_v  = np.sum(int_uv,axis=0)
    int_w  = np.sum(int_uw,axis=0)

    lengths = np.array([np.sum(int_u>0), np.sum(int_v>0), np.sum(int_w>0)])
    ix = np.argsort(lengths)[::-1]
    print("lengths: ",lengths, ", ix: ",ix)
    
    (umin,umax), _ = axis_parameter_bounds(mask.shape, cm, u_axis)
    (vmin,vmax), _ = axis_parameter_bounds(mask.shape, cm, v_axis)
    (wmin,wmax), _ = axis_parameter_bounds(mask.shape, cm, w_axis)

    u_prefix, u_postfix = np.sum(int_u[0:int(np.ceil(abs(umin)))]>0), np.sum(int_u[int(np.floor(abs(umin))):]>0)
    v_prefix, v_postfix = np.sum(int_v[0:int(np.ceil(abs(vmin)))]>0), np.sum(int_v[int(np.floor(abs(vmin))):]>0)
    w_prefix, w_postfix = np.sum(int_w[0:int(np.ceil(abs(wmin)))]>0), np.sum(int_w[int(np.floor(abs(wmin))):]>0)


    return {
        'principal_axes':np.array([u_axis,v_axis,w_axis]),
        'principal_axes_ranges':np.array([[-u_prefix*voxelsize,u_postfix*voxelsize],
                                          [-v_prefix*voxelsize,v_postfix*voxelsize],
                                          [-w_prefix*voxelsize,w_postfix*voxelsize]]),
        'centre_of_mass':cm*voxelsize
    };

def bounding_cylinder(volume_meta):
    u_axis       = volume_meta["principal_axes"][0]
    [[u_min, u_max], [v_min, v_max], [w_min,w_max]] = volume_meta["principal_axes_ranges"]
    cm           = volume_meta["centre_of_mass"]
    
    radius         = np.abs(np.array([v_min,v_max,w_min,w_max])).max()
    x_start, x_end = cm+u_min*u_axis, cm+u_max*u_axis
    return x_start, x_end, radius


def bounding_box(volume_meta):
    u_axis, v_axis, w_axis  = volume_meta["principal_axes"]
    [u_range, v_range, w_range] = volume_meta["principal_axes_ranges"]
    cm      = volume_meta["centre_of_mass"]

    corners = np.zeros((2,2,2,3),dtype=float)
    for i in range(8):
        u, v, w = i&1, (i>>1)&1, (i>>2)&1
        corners[u,v,w] = cm + u_range[u]*u_axis + v_range[v]*v_axis + w_range[w]*w_axis;

    return corners

def slice_images(img,sample,dir,rescale=False):
    (Nz,Ny,Nx) = img.shape
    xy = img[Nz//2,:,:]
    xz = img[:,Ny//2,:]
    yz = img[:,:,Nx//2]
    if(rescale):
        xy = (255*xy/xy.max()).astype(np.uint8)
        xz = (255*xz/xz.max()).astype(np.uint8)
        yz = (255*yz/yz.max()).astype(np.uint8)
        
    cv2.imwrite(f"{dir}/{sample}-xy.png",xy)
    cv2.imwrite(f"{dir}/{sample}-xz.png",xz)
    cv2.imwrite(f"{dir}/{sample}-yz.png",yz)        

# Preconditions:
#  - volume matching of sample is already calculated
def subvolume_coordinates(sample):
    # Get scan metadata
    f    = h5.File(f"{paths.hdf5_root}/hdf5-byte/msb/{sample}.h5","r")
    (Nz,Ny,Nx) = f["voxels"].shape
    
    shifts   = f["volume_matching_shifts"][:]
    subdims  = f["subvolume_dimensions"][:]
    Nsubvolumes = len(subdims)

    # Extract voxel size in micrometer from experiment data
    meta0 = f["metadata"]["subvolume0"].attrs
    voxelsize = float(meta0["voxelsize"])
    f.close()
    
    offset = [0]+list(np.cumsum(subdims[:,0]))
    lag = [0]+list(np.cumsum(shifts))

    world_Z0 = np.zeros(Nsubvolumes,dtype=int)
    world_Z1   = np.zeros(Nsubvolumes,dtype=int)
    for subvolume in range(Nsubvolumes):
        world_Z0[subvolume] = offset[subvolume]-lag[subvolume]
        world_Z1[subvolume]   = world_Z0[subvolume]+subdims[subvolume,0]

    # xs = np.linspace(-Nx/2,Nx/2)*voxelsize
    # ys = np.linspace(-Ny/2,Ny/2)*voxelsize
    # zss = np.array([np.linspace(world_Z0[i],world_Z1[i],subdims[i,0])*voxelsize for i in range(Nsubvolumes)])
        
    return offset, world_Z0, world_Z1 #, (xs,ys,zss)
    

implant_bound = {}

def process_implant(sample, voxelsize=1.85, coarse_scale=6, fine_scale=2):
    global implant_bound, D, cm, uvw_axes, uvw_ranges, fine_implant, mmtofi
    from config.constants import implant_threshold_byte
    
    # 1. Load and threshold data (coarse-scaled, so just load all)    
    with h5.File(f"{paths.hdf5_root}/processed/volume_matched/{coarse_scale}x/{sample}.h5","r") as f:
        coarse_implant = np.array(f["voxels"][:]>implant_threshold_byte)
        
        # 2. Calculate principal axis + bounding volume
        implant_bound = bounding_volume(coarse_implant, voxelsize*coarse_scale)

        f.close()        
        del coarse_implant

    # 3. Now calculate EDT for high-resolution data
    with h5.File(f"{paths.hdf5_root}/processed/volume_matched/{fine_scale}x/{sample}.h5","r") as f:
        (Nz,Ny,Nx) = f["voxels"].shape

        # 4. Load and process fine-scaled imlant. Assumes it fits in RAM, TODO: blockwise
        print(f"Loading and thresholding implant for full volume_matched tomogram",flush=True)        
        fine_implant = np.array(f["voxels"][:]>implant_threshold_byte,dtype=np.uint8) # np.bool messes up pybind

        # 5. Set mask to zero outsize principal axis bounding box.
        mmtofi = 1/(voxelsize*fine_scale) # Conversion factor from micrometers to index
        uvw_axes   = implant_bound["principal_axes"]
        uvw_ranges = implant_bound["principal_axes_ranges"]*mmtofi
        cm         = implant_bound["centre_of_mass"]*mmtofi
        
        print(f"cm = {cm} voxels")
        print(f"urange = {uvw_ranges[0]} voxels, vrange = {uvw_ranges[1]} voxels, wrange = {uvw_ranges[2]} voxels")
        print(f"fine_implant.shape = {fine_implant.shape}")
        print(f"Zeroing voxels outside of bounding box")        
        geometry.zero_outside_bbox(uvw_axes.flatten(),uvw_ranges.flatten(),cm,
                                   fine_implant);       

        implant_dir = f"{paths.hdf5_root}/processed/implant/{fine_scale}x/"
        print(f"Storing cleaned-up implant mask in {implant_dir}/{sample}.h5")        
        pathlib.Path(implant_dir).mkdir(parents=True, exist_ok=True)                
        with h5.File(f"{implant_dir}/{sample}.h5","w") as implant_h5:
            implant_h5.create_dataset("voxels",data=fine_implant,compression="gzip")
            slice_images(fine_implant,sample,implant_dir,rescale=True);
            # TODO: Also store bounding volume
            
        # 6. Calculating distance transform
        edt_dir = f"{paths.hdf5_root}/processed/implant-edt/{fine_scale}x/"
        pathlib.Path(edt_dir).mkdir(parents=True, exist_ok=True)        
        with h5.File(f"{edt_dir}/{sample}.h5","w") as edt_h5:
            print("Calculating 1/(d+1) transform",flush=True)
            EDT = edt.edt3d(np.logical_not(fine_implant),parallel=N_CPU_THREADS)
            print(f"EDT range = {EDT.min(), EDT.max()}")            
            print("Zeroing implant mask")
            EDT *= np.logical_not(fine_implant)
            print(f"EDT range = {EDT.min(), EDT.max()})")
            slice_images(EDT,f"{sample}-edt",edt_dir);            
            print(f"Generating 1/(100/scale+EDT) - EDT/(100*Nx)")
            D = (1.0/((100.0/fine_scale)+EDT)) - EDT/(100.0*Nx)
            print(f"Maxing")            
            dmin, dmax = histograms.float_minmax(D)

            print(f"Renormalizing distance transform from [{dmin};{dmax}] to 16 bit",flush=True)
            D = (2**16-1)*(D-dmin)/(dmax-dmin)
            D = D.astype(np.uint16)

            print(D.shape)
            print(D.dtype)
            print(f"Storing EDT in {edt_dir}/{sample}.h5")
            edt_voxels = edt_h5.create_dataset("voxels",data=D);
            edt_voxels.attrs.create("range", [dmin,dmax], dtype=float)
            edt_voxels.attrs.create("voxelsize",fine_scale*voxelsize)
            edt_h5.close()

            slice_images(D,sample,edt_dir);
        
sample="770c_pag"
process_implant(sample,voxelsize=1.85,coarse_scale=9,fine_scale=2)

# if __name__ == "__main__":
#     input_h5, dataset_name, output_rel_path, chunk_size, compression = commandline_args({"input_h5":"<required>", "dataset": "voxels", "output_relative_to_input":"..",
#                                                                                          "chunk_size":6*20, "compression":"lzf"})
