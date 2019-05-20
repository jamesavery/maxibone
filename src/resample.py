import numpy as np

def sample(image,xs,ys):
    yminus,iminus= np.modf(ys-0.5);
    yplus, iplus = np.modf(ys+0.5);
    xminus,jminus= np.modf(xs-0.5);
    xplus, jplus = np.modf(xs+0.5); 
    
    ny,nx = image.shape[-2:];
    
    LD = (iminus*nx+jminus).astype(np.uint64); # x-,y-
    LU = (iplus*nx +jminus).astype(np.uint64); # x-,y+
    RD = (iminus*nx+jplus).astype(np.uint64);  # x+,y-
    RU = (iplus*nx +jplus).astype(np.uint64);  # x+,y+
    
    I = image.reshape((-1,nx*ny));
    
    I_polar = (1-xminus)*(1-yminus)*I[:,LD] \
             +(1-xminus)*yplus     *I[:,LU] \
             +xplus     *yplus     *I[:,RD] \
             +xplus*(1-yminus)     *I[:,RU];
    
    return I_polar.reshape((-1,xs.shape[0],xs.shape[1]))

def cart_to_polar(image,nr,ntheta, r=0, R=None):
    midy, midx = np.array(image.shape[-2:])/2;
    assert(midx==midy);
    if(R==None):
        R = midx;
    
    rs = np.linspace(r,R,nr,endpoint=False);    
    thetas = np.linspace(0,2*np.pi,ntheta,endpoint=False);
    
    xs = rs[:,None]*np.cos(thetas)[None,:]+midx;
    ys = rs[:,None]*np.sin(thetas)[None,:]+midy;
    return sample(image,xs,ys);



def polar_to_cart(polar_image,nx,ny):
    R = polar_image.shape[1];    
    xs = np.arange(nx)+0.5 - R; 
    ys = np.arange(ny)+0.5 - R;
    print(nx,ny,R)
    print(xs.min(),xs.max())
    print(ys.min(),ys.max())
        
    rs     = np.sqrt(xs[None,:]**2 + ys[:,None]**2);
    invalid = rs>=R;
    thetas = np.arctan2(xs[None,:],ys[:,None])
    
    rs     = ma.masked_array(rs,    mask=invalid)
    thetas = ma.masked_array(thetas,mask=invalid)
    
    print(rs.min(),rs.max())
    print(thetas.min(),thetas.max())
    return sample(polar_image,rs,thetas)


