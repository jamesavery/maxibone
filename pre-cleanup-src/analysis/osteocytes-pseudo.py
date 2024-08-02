import numpy as np;
from scipy import ndimage as ndi;

#0: Haps metadata for tomogrammet
h5meta       = h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5","r")  # Eller hvor metainformationen nu er nu
voxel_length = h5meta["voxels"].attrs["voxelsize"]*segment_scale
voxel_volume = voxel_length**3

# Og biologiske konstanter - boer flyttes til constants.py
# Fra Wikipedia: 
# Osteocytes have a stellate shape, approximately 7 micrometers deep and wide by 15 micrometers in length.[3]
# The cell body varies in size from 5–20 micrometers in diameter and contain 40–60 cell processes per cell,
# [4] with a cell to cell distance between 20–30 micrometers.[3] 
minmax_buffer = 1.25
osteocyte_diam_min = 5/minmax_buffer
osteocyte_diam_max = 20*minmax_buffer
osteocyte_lmin = osteocyte_diam_min/np.sqrt(3)  # Smallest possible side-length: assuming box-shaped osteocyte of minimum diameter
osteocyte_lmax = osteocyte_diam_max/np.sqrt(3)
osteocyte_Vmin = osteocyte_lmin**3
osteocyte_Vmax = osteocyte_lmax**3

#1: ...do your thing for at soft-tissue minus blodnetvaerk nu ligger i "holes"

#2: Label potentielle osteocytter
hole_id, num_holes = ndi.label(holes) # But use fast parallel out-of-core labeling function

#3: Udregn volumener og sorter urealistisk store eller smaa osteocytter fra
volumes = np.bincount(hole_id)*voxel_volume # But use fast parallel bincount
small_unknown   = volumes < osteocyte_min     # mask on range(0,max(hole_id)+1)
large_unknown   = volumes > osteocyte_max     # -||-
osteocyte_sized = (volumes >= osteocyte_Vmin) & (volumes <= osteocyte_Vmax)

#4: Udregn ellipsoide-fits
pot_osteocyte_segments = mb.select_segments(mb.collect_nonzeros(holes), osteocyte_sized) # sparse representation, index lists per id (in contiguous mem).
inertia_matrices   = mb.inertia_matrices(pot_osteocyte_segments)   # n_pot_osteocytes x 3 x 3
principal_lambdas  = mb.eigenvalues3x3(intertia_matrices)      # n_pot_osteocytes x 3, eigenvalues of moment of inertia matrices: smallest, middle, largest

# Osteocytter er i gennemsnit dobbelt saa lange som brede, men kan variere lidt. Vi godtager ned til sfaerisk (1:1:1) og op til 3x saa lang.
abc      = 1/np.sqrt(principal_lambdas)              # n_pot_osteocytes x 3, ellipsoid dimensions a,b,c: longest, middle, shortest
a, b, c  = abc.T
weirdly_long       = (a/c)>3

# Osteocytterne er tilnaermelsesvist ellipsoideformede. Vi tester at de ikke er for forskellige fra bedste ellipsoide.
osteocyte_cms     = mb.centres_of_mass(pot_osteocyte_segments)
ellipsoid_errors  = mb.substract_ellipsoids(pot_osteocyte_segments, osteocyte_cms, abc)
ellipsoid_volumes = 4*np.pi/3 * a*b*c

# TODO: Kig paa histogram af ellipsoid_errors/ellipsoid_volumes og se om der er en paen hoved-normalfordeling (osteocytterne) + snask (snask).
#       Det giver nok fornuftig cutoff. 
weirdly_shaped = (ellipsoid_errors/ellipsoid_volumes) > ellipsoid_error_threshold

# Final osteocyte segmentation!
osteocyte_segments = mb.select_segments(pot_osteocyte_segments, (~weirdly_long) & (~weirdly_shaped))



#...gammel kode, som vi maaske skal snuppe fra
def moment_of_inertia(rho,x=None):
    (nx,ny,nz) = rho.shape;
    (ix,iy,iz) = (np.arange(nx),np.arange(ny),np.arange(nz));

    if(x==None):
        x = center_of_mass(rho);
    
    # Change to x-coordinate system
    (rx,ry,rz) = (ix-x[0],iy-x[1],iz-x[2]);

    I = np.zeros((3,3),dtype=np.float);
    I[0,0] = np.sum( rho*(ry[None,:,None]**2 + rz[None,None,:]**2) );
    I[1,1] = np.sum( rho*(rx[:,None,None]**2 + rz[None,None,:]**2) );
    I[2,2] = np.sum( rho*(rx[:,None,None]**2 + ry[None,:,None]**2) );
    I[0,1] = I[1,0] = -np.sum( rho*rx[:,None,None]*ry[None,:,None]);
    I[0,2] = I[2,0] = -np.sum( rho*rx[:,None,None]*rz[None,None,:]);
    I[1,2] = I[2,1] = -np.sum( rho*ry[None,:,None]*rz[None,None,:]);

    return I;

#TODO: Closed-form solutions to eigensystem instead of NumPy
def principal_axes(I):
    lam,Ix = np.linalg.eigh(I);
#TODO: Sanity checks    
    return Ix;

def blob_principal_axes(osteocytes,cm,diameter):
    (start,end) = (int(np.round(x-diameter)), int(np.round(x+diameter+1)));

    rho = osteocytes[start[0]:end[0],start[1]:end[1],start[2]:end[2]];
    I   = moment_of_inertia(rho);
    return principal_axes(I);

def osteocytes_principal_axes(osteocyte_image,osteocyte_centres):
    osteocyte_length = 7;         # TODO: Calculate from voxelsize. TODO2: .py-file with biological sizes
    fn  = np.vectorize(lambda cm: blob_principal_axes(osteocyte_image,cm,1.5*osteocyte_length));

    return fn(osteocyte_centres);

def osteocyte_centres(block,osteocyte_nz, osteocyte_ids, num_osteocytes):
    vols = np.zeros((num_osteocytes+1));
    cms  = np.zeros((num_osteocytes+1),3);

    for (x,y,z) in osteocyte_nz:
        oid = osteocyte_ids[x,y,z];
        vols[oid] += block[x,y,z];
        cms[oid]  += (x,y,z);

    return cms;
