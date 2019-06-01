import bohrium as bh

def prod(a):
    return bh.product(a,dtype=bh.int64)

def axis_split(A,axis=0):
    (Am,Ai,An) = (prod(A.shape[:axis]),  A.shape[axis], prod(A.shape[axis+1:]));
    return (Am,Ai,An)


# Erosion, dilation, opening, and closing with square structuring elements (the super-easy case)

def erosion_1d(A_,n=1, axis=0):
    (Am,Ai,Ar) = axis_split(A_,axis)
    A = A_.astype(bool).reshape(Am,Ai,Ar)
    B = A.copy();
    
    for i in range(n):
        B[:,1:,:]  &= A[:,:-1,:]
        B[:,:-1,:] &= A[:,1:,:]
        A, B = B, A
    return B.reshape(A_.shape)


def dilation_1d(A_,n=1, axis=0):
    (Am,Ai,Ar) = axis_split(A_,axis)
    A = A_.astype(bool).reshape(Am,Ai,Ar)
    B = A.copy();
    
    for i in range(n):
        B[:,1:,:]  |= A[:,:-1,:]
        B[:,:-1,:] |= A[:,1:,:]
        A, B = B, A
    return B.reshape(A_.shape)


def erosion(A,n=1, axes=None):
    if axes is None:
        axes = range(len(A.shape))
        
    B = erosion_1d(A,n,axes[0])
    for axis in axes[1:]:
        B = erosion_1d(B,n,axis)
    return B

def dilation(A,n=1, axes=None):
    if axes is None:
        axes = range(len(A.shape))
        
    B = dilation_1d(A,n,axes[0])
    for axis in axes[1:]:
        B = dilation_1d(B,n,axis)
    return B


def opening(A,n=1,axes=None,iterations=1):
    B = erosion(A,n,axes)
    B = dilation(B,n,axes)    
    for i in range(iterations-1):
        B = erosion(B,n,axes)
        B = dilation(B,n,axes)    
    return B

def closing(A,n=1,axes=None,iterations=1):
    B = dilation(A,n,axes)
    B = erosion(B,n,axes)    
    for i in range(iterations-1):
        B = dilation(B,n,axes)            
        B = erosion (B,n,axes)
    return B


# Erosion, dilation, opening, and closing with cross-shaped element (the slightly less easy case)
def axis2_split(A,axes=[0,1]):
    (a0,a1) = axes;
    (Am,Ai,Ap,Aj,An) = (prod(A.shape[:a0]),  A.shape[a0], prod(A.shape[a0+1:a1]), A.shape[a1], prod(A.shape[a1+1:]));
    return (Am,Ai,Ap,Aj,An)

def erosion_cross_2d(A_,n=1, axes=[0,1]):
    (Am,Ai,Ap,Aj,Ar) = axis2_split(A_,axes)
    A = A_.astype(bool).reshape(Am,Ai,Ap,Aj,Ar)
    B = A.copy();
    
    for i in range(n):
        # nearest neighbours along axes[0]
        B[:, 1:] &= A[:,:-1]
        B[:,:-1] &= A[:, 1:]
        # nearest neighbours along axes[1]        
        B[:,:,:,1:]  &= A[:,:,:,:-1]
        B[:,:,:,:-1] &= A[:,:,:,1:]
        A, B = B, A
    return B.reshape(A_.shape)

def dilation_cross_2d(A_,n=1, axes=[0,1]):
    (Am,Ai,Ap,Aj,Ar) = axis2_split(A_,axes)
    A = A_.astype(bool).reshape(Am,Ai,Ap,Aj,Ar)
    B = A.copy();
    
    for i in range(n):
        # nearest neighbours along axes[0]
        B[:, 1:] |= A[:,:-1]
        B[:,:-1] |= A[:, 1:]
        # nearest neighbours along axes[1]        
        B[:,:,:,1:]  |= A[:,:,:,:-1]
        B[:,:,:,:-1] |= A[:,:,:,1:]
        A, B = B, A
    return B.reshape(A_.shape)


def erosion_cross(A_,n=1, axes=None):
    if axes is None:
        axes = range(len(A.shape))
        
    n_ax   = len(axes)
    shapes = bh.array([axis_split(A_,axis) for axis in axes],dtype=int)

    A = A_.astype(bool)
    B = A.copy();
    
    for i in range(n):
        for axis in axes:
            # nearest neighbours along axis
            Bax = B.reshape(shapes[axis])
            Aax = A.reshape(shapes[axis])
            Bax[:,:-1] &= Aax[:, 1:]
            Bax[:,1:]  &= Aax[:,:-1]            
        
        A, B = B, A
    return B.reshape(A.shape)


def dilation_cross(A_,n=1, axes=None):
    if axes is None:
        axes = range(len(A.shape))
        
    n_ax   = len(axes)
    shapes = bh.array([axis_split(A_,axis) for axis in axes],dtype=int)

    A = A_.astype(bool)
    B = A.copy();
    
    for i in range(n):
        for axis in axes:
            # nearest neighbours along axis
            Bax = B.reshape(shapes[axis])
            Aax = A.reshape(shapes[axis])
            Bax[:,:-1] |= Aax[:, 1:]
            Bax[:,1:]  |= Aax[:,:-1]            

        A, B = B, A
    return B.reshape(A.shape)

def closing_cross(A,n=1,axes=None,iterations=1):
    B = dilation_cross(A,n,axes)
    B = erosion_cross(B,n,axes)    
    for i in range(iterations-1):
        B = dilation_cross(B,n,axes)            
        B = erosion_cross (B,n,axes)
    return B


def opening_cross(A,n=1,axes=None,iterations=1):
    B = erosion_cross(A,n,axes)
    B = dilation_cross(B,n,axes)    
    for i in range(iterations-1):
        B = erosion_cross(B,n,axes)
        B = dilation_cross(B,n,axes)    
    return B
