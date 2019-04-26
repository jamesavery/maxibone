def center_of_mass(b):
    (nx,ny,nz) = b.shape;
    (ix,iy,iz) = (np.arange(nx),np.arange(ny),np.arange(nz));
    vol = np.sum(b);
    cx  = np.sum(b*ix[:,None,None])/vol;
    cy  = np.sum(b*iy[None,:,None])/vol;
    cz  = np.sum(b*iz[None,None,:])/vol;

    return np.array([cx,cy,cz]);
    
def moment_of_inertia(rho,x):
    (nx,ny,nz) = rho.shape;
    (ix,iy,iz) = (np.arange(nx),np.arange(ny),np.arange(nz));

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

