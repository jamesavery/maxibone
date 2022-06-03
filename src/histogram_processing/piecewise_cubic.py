from numpy import array, linspace, sin, empty, zeros, linalg, random, pad, concatenate
import numpy.linalg as la
# Scheme:
#  Fit an N-segment piecewise cubic polynomial to a set of points with linear least squares with
#  two exact conditions:
#   (1) Continuity at the borders:        f_n (X_n) = f_{n-1} (X_n)
#   (2) Differentiability at the borders: f_n'(X_n) = f'_{n-1}(X_n)
#  
#  Input: A list of M data points [(x1,y1),....(x_M,y_M)] and 
#         A list of N+1 borders   [X0,X1,....,X_N]
# 
#  This file generalizes the 2-segment method given in cubic2.py to an arbitrary
#  number of segments.
#           / f1(x) = A1 + B1*(x-X0) + C1*(x-X0)**2 + D1*(x-X0)**3          if x>=X0 & x<=X1
#    f(x) = |
#           | f2(x) = f1(X1) + f1'(X1)*(x-X2) + C2*(x-X2)**2 + D2*(x-X2)**3 if x>=X1 & x<=X2
#           |   
#           | f3(x) = f2(X2) + f2'(X2)*(x-X3) + C2*(x-X3)**2 + D2*(x-X3)**3 if x>=X2 & x<=X3
#           |   ...
#           \ f_N(x) = f_{N-1}(X_{N-1}) + f'_{N-1}(X_{N-1})*(x-X_N) + C_N*(x-X_N)**2 + D_N*(x-X_N)**3
#                                                                           if x>=X_{N-1} & x<= X_N
#
# We still want to set up a linear least squares system of equations that implicitly
# obeys the two exact conditions (so they are not weakened by the least squares
# approximation to the over-determined system).

# Eq. (1) and (2) determine A_n (the function value at X_n) and B_n (the derivative at X_n).
# So we have 4+2*(N-1) = 2*(N+1) variables to determine: A1, B1, C1, D1, and C2, D2, C3, D3, ..., C_N, D_N

# Each data point (x_i,y_i)  defines a row with the coefficients for A1,B1,C1,D1,C2,D2,...,C_N,D_N in the matrix, and y_i on the RHS.
def mxrow_f (x, borders):
    if(len(borders) < 2):
        printf(f"No segments: borders = {borders}")
        return
    
    Xleft, Xright = borders[-2:] # x >= Xleft & x <= Xright
    n  = len(borders)-2         # n is the current segment number     
#    print(f"f_{n}({x}): Xleft,Xright = {np.round([Xleft,Xright],2)}; borders = {borders}, {len(borders)-1} segments")
    
    if n==0:
        X = x-Xleft
        return array([ 1, X, X**2,  X**3])
    else:
        # We recursively define the matrix row        
        # f_n(x) = f_{n-1}(Xn) + f'_{n-1}(Xn)*(x-Xn) + Cn*(x-Xn)**2 + Dn*(x-Xn)**3
#        print(f"Recursing down from level {n} to level {n-1}, evaluated at {Xleft}")
        row               = zeros((4+2*n,), dtype=float)
        row[:(4+2*(n-1))] = mxrow_f(Xleft,borders[:-1]) + mxrow_df(Xleft,borders[:-1])*(x-Xleft);
        row[-2:]          = [(x-Xleft)**2, (x-Xleft)**3]
#        print(f"returning length = {len(row)} row at level {n}")
        return row
        
def mxrow_df(x,borders):
    if len(borders) < 2:
        print(f"No segments: borders={borders}")
        return

    Xleft, Xright = borders[-2:] # x >= Xleft & x <= Xright
    n  = len(borders)-2         # n is the current segment number     
#    print(f"f'_{n}({x}): Xleft,Xright = {np.round([Xleft,Xright],2)}; borders = {borders}, {len(borders)-1} segments")

    if n==0:
        X0 = borders[0]
        return array([ 0,    1, 2*(x-X0) ,3*(x-X0)**2]) 
    else:
        # We recursively define the matrix row: f'_n(x) = f'_{n-1}(Xn) + Cn*2*(x-Xn) + Dn*3*(x-Xn)
        row               = zeros((4+2*n,), dtype=float)
        row[:(4+2*(n-1))] = mxrow_df(Xleft,borders[:-1])
        row[-2:]          = [2*(x-Xleft), 3*((x-Xleft)**2)]
        return row 
    

# We can now put these together to construct the matrix and RHS row by row:
def piecewisecubic_matrix(xs,ys, Xs): 
    M = len(xs)                 # M data points 
    N = len(Xs)-1               # N segments, i.e. N+1 borders 
    
    A = empty((M,2*(N+1)),dtype=float)
    b = empty((M,1),dtype=float)

    n = 0                        # Start in first region
    Xleft, Xright = Xs[0], Xs[1] 

    for i in range(len(xs)):
#        print(f"Xleft, Xright = {Xleft, Xright}, Xs = {Xs}, n = {n}")
        if(xs[i] > Xright):
            n += 1
            Xleft, Xright = Xs[n], Xs[n+1]

#        print(f"A[{i}]: n = {n}, 4+2n = {4+2*n}, n+2 = {n+2}")
        A[i,:(4+2*n)] = mxrow_f(xs[i],Xs[:(n+2)]) 
        b[i] = ys[i]

    return A,b

# A function that takes an N-segment piecewise cubic polynomial
# produced by fit_piecewisecubic() and evaluates it on an arbitrary
# set of coordinates xs (not necessarily the same as the data points
# it was fitted to):
def piecewisecubic(pc,all_xs):
    coefs, Xs = pc          # Polynomial coefficients A1,B1,C1,D1,C2,D2,C3,D3,... and borders
    N = len(Xs)-1           # N segments have N+1 borders: |seg1|seg2|...|segN|

    ys = []                 # List of function values for segments
    A,B = coefs[:2]         # A and B coefficients are only defined for 0-segment

    for n in range(N):
        C, D          = coefs[(2+2*n):(2+2*n+2)] # 
        
        Xleft, Xright = Xs[n], Xs[n+1]
        xs_segment    = all_xs[(all_xs>=Xleft) & (all_xs< Xright)]

        
        # f_n(x) = A + B*(x-Xn) + C*(x-Xn)**2 + D*(x-Xn)**3        
        xs   = xs_segment - Xleft                 # Segment coordinate x-Xn
        ys += [A + B*xs + C*(xs**2) + D*(xs**3)]  # f_n(xs)

        # Calculate next A and B coefficients:
        X = Xright-Xleft
        A = A + B*X + C*(X**2) + D*(X**3)   # A_{n+1} = f_n (X_{n+1})
        B =     B   + C*2*X    + D*3*(X**2) # B_{n+1} = f'_n(X_{n+1})
        
    return concatenate(ys)



# ...and a function to construct the overdetermined linear system of 
# equations and find the least squares optimal approximate solution:
def fit_piecewisecubic(xs,ys, Xs):
    A, b = piecewisecubic_matrix(xs,ys,Xs)

    coefs, residuals, rank, sing = linalg.lstsq(A,b,rcond=None)

    return (coefs,Xs)


def smooth_fun(xs,ys,n_segments):
    borders = linspace(xs.min(), xs.max()+1,n_segments)    

    A, b = piecewisecubic_matrix(xs,ys,borders)
    coefs, residuals, rank, sing = la.lstsq(A,b,rcond=None)    
    pc = coefs, borders

    return pc

if __name__ == "__main__":
    # A test:
    import numpy as np

    # N, m = 100, 50
    # xs = linspace(2,15,N)
    # ys = sin(xs) + sin(xs/3)# + random.rand(N) 
    # borders = [xs[0],xs[N//5],xs[2*N//5],xs[3*N//5],xs[4*N//5],xs[-1]+1]
    
    # pc = fit_piecewisecubic(xs,ys,borders)
    
    # Ys = piecewisecubic(pc, xs)
    
    # print(f"coefs = {pc[0]})")

    import matplotlib.pyplot as plt
    # #plt.plot(xs,ys-Ys)
    # plt.plot(xs,ys)
    # plt.plot(xs,Ys)
    # #plt.axvline(x=xs[m])
    # plt.show()


    f = np.load("output_curve_values.npy")

    vals = f[7,:,1].T
    mask = vals>0

    xs      = np.argwhere(mask).astype(float).flatten()
    ys      = vals[mask]
    borders = np.linspace(xs.min(), xs.max()+1,5)
    A, b = piecewisecubic_matrix(xs,ys,borders)

    coefs, residuals, rank, sing = linalg.lstsq(A,b,rcond=None)
    
    pc = coefs, borders
    
    Ys = piecewisecubic(pc,xs)

    plt.plot(xs,ys)
    plt.plot(xs,Ys)
    plt.show()
