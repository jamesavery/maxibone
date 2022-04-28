from numpy import array, linspace, sin, empty, linalg, random
# Scheme:
#  Fit a piecewise cubic polynomial to a set of points with linear least squares with
#  two exact conditions:
#   (1) Continuity at the joins:        f1(x_m)  = f2(x_m)
#   (2) Differentiability at the joins: f1'(x_m) = f2'(x_m)
#
#  This file illustrates how to do it for two segments:
#           / f1(x) = A1 + B1*(x-x1) + C1*(x-x1)**2 + D1*(x-x1)**3 if x<=xm
#    f(x) = |
#           \ f2(x) = A2 + B2*(x-xm) + C2*(x-xm)**2 + D2*(x-xm)**3 if x>=xm
#
# We want to set up a linear least squares system of equations that implicitly
# obeys the two exact conditions (so they are not weakened by the least squares
# approximation to the over-determined system).

# Eq. (1) and (2) determine A2 (the function value at xm) and B2 (the derivative at xm).
# So we have 6 variables to determine: A1, B1, C1, D1, and C2, D2

# Each data point (x_i,y_i)  defines a row with the coefficients for A1,B1,C1,D1,C2,D2 in the matrix, and y_i on the RHS.
#
# For the left region, it is constructed like this:
#  f1(x)  = A1 + B1*(x-x1) + C1*(x-x1)**2 + D1*(x-x1)**3
#  f1'(x) =      B1        + C1*2*(x-x1)  + D1*3*(x-x1)**2
#                                     A1    B1         C1          D1   C2   D2
def mxrow_f1 (x1,x):    return array([ 1, x-x1, (x-x1)**2,  (x-x1)**3,   0,   0])
def mxrow_df1(x1,x):    return array([ 0,    1, 2*(x-x1) ,3*(x-x1)**2,   0,   0]) 

# The two exact conditions give:
# (1): A2 =  f2(xm) = f1(xm)
# (2): B2 = f2'(xm) = f2'(xm)
# Thus: 
# f2(x)  = f1(xm) + f1'(xm)*(x-xm) + C2*(x-xm)**2 + D2*(x-xm)**3
# f2'(x) =          f1'(xm)        + C2*2*(x-xm)  + D2*3*(x-xm)**2
def mxrow_f2 (x1,xm,x): return mxrow_f1(x1,xm)  + mxrow_df1(x1,xm)*(x-xm) + array([0,    0,         0,          0, (x-xm)**2, (x-xm)**3])
def mxrow_df2(x1,xm,x): return mxrow_df1(x1,xm) +                         + array([0,    0,         0,          0,  2*(x-xm), 3*(x-xm)**2])


# We can now put these together to construct the matrix and RHS row by row:
def piecewisecubic2_matrix(xs,ys, m): 
    N = len(xs)

    A = empty((N,6),dtype=float)
    b = empty((N,1),dtype=float)
    
    for i in range(m):
        A[i] = mxrow_f1(xs[0],       xs[i])
        b[i] = ys[i]

    for i in range(m,N):
        A[i] = mxrow_f2(xs[0],xs[m], xs[i])
        b[i] = ys[i]

    return A,b

# ...and a function to construct the overdetermined linear system of 
# equations and find the least squares optimal approximate solution:
def fit_piecewisecubic2(xs,ys, m):
    A, b = piecewisecubic2_matrix(xs,ys,m)

    coefs, residuals, rank, sing = linalg.lstsq(A,b,rcond=None)

    return (coefs,xs[0],xs[m])

# An auxiliary function to evaluate cubic polynomial segments
def eval_cubic(coefs,xs):
    A,B,C,D = coefs

    return A + B*xs + C*(xs**2) + D*(xs**3)


# A function that takes a 2-segment piecewise cubic polynomial
# produced by fit_piecewisecubic2() and evaluates it on an arbitrary
# set of coordinates xs (not necessarily the same as the data points
# it was fitted to):
def piecewisecubic2(pc,xs):
    (A1,B1,C1,D1, C2,D2), x1, xm = pc
    L, R = xs<xm, xs>=xm
    
    ys = empty((len(xs),), dtype=float)

    f1  = A1 + B1*(xm-x1) + C1*((xm-x1)**2) + D1*((xm-x1)**3)
    df1 =      B1         + C1*(2*(xm-x1))  + D1*3*((xm-x1)**2)
    
    ys[L] = eval_cubic( [A1,B1,C1,D1], xs[L]-x1)
    ys[R] = eval_cubic([f1,df1,C2,D2], xs[R]-xm)
    
    return ys



if ___name___ == "___main___":
    # A test:
    N, m = 100, 50
    xs = linspace(2,5,N)
    ys = sin(xs) + random.rand(N)
    
    pc = fit_piecewisecubic2(xs,ys,m)

    Ys = piecewisecubic2(pc, xs)

    print(f"xs:{xs.shape}")
    print(f"ys:{ys.shape}")
    print(f"coefs = {pc[0]}")

    import matplotlib.pyplot as plt
    #plt.plot(xs,ys-Ys)
    plt.plot(xs,ys)
    plt.plot(xs,Ys)
    plt.axvline(x=xs[m])
    plt.show()
