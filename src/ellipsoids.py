def draw_ellipsis(a,b,va,vb,xrange):
    X = xrange[:,None]*va[0] + xrange[None,:]*va[1];
    Y = xrange[:,None]*vb[0] + xrange[None,:]*vb[1];

    vals = (X/a)**2 + (Y/b)**2;    
    
    return vals<=1;

def draw_ellipsoid(abc,V,xrange):
    (a,b,c) = abc;
    X = xrange[:,None,None]*V[0,0] + xrange[None,:,None]*V[0,1] + xrange[None,None,:]*V[0,2];
    Y = xrange[:,None,None]*V[1,0] + xrange[None,:,None]*V[1,1] + xrange[None,None,:]*V[1,2];
    Z = xrange[:,None,None]*V[2,0] + xrange[None,:,None]*V[2,1] + xrange[None,None,:]*V[2,2];
    
    vals = (X/a)**2 + (Y/b)**2 + (Z/c)**2;    
    
    return vals<=1;

def ellipsoid_dimensions(cm,Ix,masksels):
    (dx,dy,dz)=(masksels.shape-cm);
    d     = int(np.ceil(np.sqrt(dx*dx + dy*dy + dz*dz)))
    lines =cm+np.linspace(-d,d,2*d)[:,None,None]*Ix;
    ilines=np.round(lines).astype(np.int16);
    abc=np.sum(masksels[ilines[:,:,0],ilines[:,:,1],ilines[:,:,2]],axis=0);
    
    return abc;


