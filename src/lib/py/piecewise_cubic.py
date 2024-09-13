'''
This module contains functions for generating Gaussian, exponential, power, and Lorentzian distributions.

Scheme:
    Fit an N-segment piecewise cubic polynomial to a set of points with linear least squares with two exact conditions:
        (1) Continuity at the borders:        f_n (X_n) = f_{n-1} (X_n)
        (2) Differentiability at the borders: f_n'(X_n) = f'_{n-1}(X_n)

    Input: A list of M data points [(x1,y1),....(x_M,y_M)] and
           A list of N+1 borders   [X0,X1,....,X_N]

This file generalizes the 2-segment method given in cubic2.py to an arbitrary number of segments.
           / f1(x) = A1 + B1*(x-X0) + C1*(x-X0)**2 + D1*(x-X0)**3          if x>=X0 & x<=X1
    f(x) = |
           | f2(x) = f1(X1) + f1'(X1)*(x-X2) + C2*(x-X2)**2 + D2*(x-X2)**3 if x>=X1 & x<=X2
           |
           | f3(x) = f2(X2) + f2'(X2)*(x-X3) + C2*(x-X3)**2 + D2*(x-X3)**3 if x>=X2 & x<=X3
           |   ...
           \ f_N(x) = f_{N-1}(X_{N-1}) + f'_{N-1}(X_{N-1})*(x-X_N) + C_N*(x-X_N)**2 + D_N*(x-X_N)**3
                                                                           if x>=X_{N-1} & x<= X_N

We still want to set up a linear least squares system of equations that implicitly obeys the two exact conditions (so they are not weakened by the least squares approximation to the over-determined system).

Eq. (1) and (2) determine A_n (the function value at X_n) and B_n (the derivative at X_n).
So we have 4+2*(N-1) = 2*(N+1) variables to determine: A1, B1, C1, D1, and C2, D2, C3, D3, ..., C_N, D_N
'''

from numpy import array, linspace, sin, empty, zeros, linalg, random, pad, concatenate
import numpy as np
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
    '''
    Recursive function to construct the matrix row for the function value `f_n(x) = A + B*(x-Xn) + C*(x-Xn)**2 + D*(x-Xn)**3`.
    The function is defined by the borders of the segments, and the current segment number `n`.

    Parameters
    ----------
    `x` : float
        The x-coordinate at which to evaluate the function.
    `borders` : list[float]
        A list of the segment borders, [X0,X1,...,Xn].
    `verbose` : bool
        Print debug information.

    Returns
    -------
    `row` : array
        The matrix row for the function value `f_n(x) = A + B*(x-Xn) + C*(x-Xn)**2 + D*(x-Xn)**3`.
    '''

        print(f"No segments: borders = {borders}")
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

    '''
    Recursive function to construct the matrix row for the derivative of the function value `f'_n(x) = f'_{n-1}(Xn) + Cn*2*(x-Xn) + Dn*3*(x-Xn)`.
    The function is defined by the borders of the segments, and the current segment number `n`.

    Parameters
    ----------
    `x` : float
        The x-coordinate at which to evaluate the function.
    `borders` : list[float]
        A list of the segment borders, [X0,X1,...,Xn].
    `verbose` : bool
        Print debug information.

    Returns
    -------
    `row` : numpy.array[float]
        The matrix row for the derivative of the function value `f'_n(x) = f'_{n-1}(Xn) + Cn*2*(x-Xn) + Dn*3*(x-Xn)`.
    '''

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

# TODO:
#  - THE COEFFICIENT DERIVATIVE SHOULD BE CONSTRUCTED FROM THE MATRIX?
#

# We can now put these together to construct the matrix and RHS row by row:
    '''
    Construct the matrix `A` and the RHS vector `b` for the linear least squares problem to fit a piecewise cubic polynomial to the data points.

    Parameters
    ----------
    `xs` : numpy.array[float]
        The x-coordinates of the data points.
    `ys` : numpy.array[float]
        The y-coordinates of the data points.
    `Xs` : numpy.array[float]
        The borders of the segments.
    `verbose` : bool
        Print debug information.

    Returns
    -------
    `A,b` : numpy.array[float], numpy.array[float]
        The matrix `A` and the RHS vector `b` for the linear least squares problem.
    '''


    A = zeros((M,2*(N+1)),dtype=float)
    b = zeros((M,1),dtype=float)

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

    '''
    A function that takes an N-segment piecewise cubic polynomial produced by `fit_piecewisecubic()` and evaluates it on an arbitrary set of coordinates `all_xs` (not necessarily the same as the data points it was fitted to).

    Parameters
    ----------
    `pc` : tuple(numpy.array[float], numpy.array[float])
        The polynomial coefficients and the borders of the segments.
    `all_xs` : numpy.array[float]
        The x-coordinates at which to evaluate the function.
    `extrapolation` : str
        The type of extrapolation to use for points outside the domain. Options are "cubic", "linear", and "constant".

    Returns
    -------
    `ys` : numpy.array[float]
        The y-coordinates of the function evaluated at the x-coordinates `all_xs`.
    '''

    coefs, Xs = pc          # Polynomial coefficients A1,B1,C1,D1,C2,D2,C3,D3,... and borders
    all_xs = all_xs.astype(float)
    N = len(Xs)-1           # N segments have N+1 borders: |seg1|seg2|...|segN|

    ys = []                 # List of function values for segments
    A,B = coefs[:2]         # A and B coefficients are only stored for 0-segment

    # Process points outside the domain, left side
    Xmin = Xs[0]
    C,D = coefs[2:4]
    xs_left_of_domain = all_xs[all_xs < Xmin]
    xs = xs_left_of_domain - Xmin

    if extrapolation=="cubic":
        ys += [A + B*xs + C*(xs**2) + D*(xs**3)]
    if extrapolation=="linear":
        # evaluate dy/dx at domain edge Xs.min()
        ys   += [A + B*xs]      # A is value at Xmin, B is slope at Xmin
    if extrapolation=="constant":
        ys += [np.ones(xs_left_of_domain.shape)*A] # A is value at Xmin

    # Process points inside domain
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


    # Process points outside the domain, right side
    Xmax = Xs[-1]
    xs_right_of_domain = all_xs[all_xs >= Xmax]
    xs = xs_right_of_domain - Xmax

    if extrapolation=="cubic":
        ys += [A + B*xs + C*(xs**2) + D*(xs**3)]
    if extrapolation=="linear":
        ys   += [A + B*xs]
    if extrapolation=="constant":
        ys += [np.ones(xs_right_of_domain.shape)*A]

    return concatenate(ys)

    '''
    A function to construct the overdetermined linear system of equations and find the least squares optimal approximate solution for a piecewise cubic polynomial.

    Parameters
    ----------
    `xs` : numpy.array[float]
        The x-coordinates of the data points.
    `ys` : numpy.array[float]
        The y-coordinates of the data points.
    `Xs` : numpy.array[float]
        The borders of the segments.
    `regularization_beta` : float
        The regularization parameter for the least squares problem.
    `verbose` : bool
        Print debug information.

    Returns
    -------
    `coefs, Xs` : tuple(numpy.array[float], numpy.array[float])
        The polynomial coefficients and the borders of the segments.
    '''


    if regularization_beta>0:
        m,n = A.shape
        I   = np.eye(n,n)[3:]
        I[:,5:]   -= np.eye(n-3,n-3)[:,:-2]
        I[:-2,7:] -= np.eye(n-5,n-5)[:,:-2]
        print(I)
        Ap  = np.vstack([A,regularization_beta*I])
        bp  = np.concatenate([b,np.zeros((n-3,1))])
        print(f"{regularization_beta} {m,n} {A.shape}, {I.shape}, {Ap.shape}, b:{b.shape}, bp:{bp.shape}")
        coefs, residuals, rank, sing = linalg.lstsq(Ap,bp,rcond=None)
    else:
        coefs, residuals, rank, sing = linalg.lstsq(A,b,rcond=None)

    return (coefs.reshape(-1),Xs)

    '''
    A function to fit a smooth piecewise cubic polynomial to a set of data points.
    The number of segments is given by `n_segments`.

    Parameters
    ----------
    `xs` : numpy.array[float]
        The x-coordinates of the data points.
    `ys` : numpy.array[float]
        The y-coordinates of the data points.
    `n_segments` : int
        The number of segments in the piecewise cubic polynomial.
    `regularization_beta` : float
        The regularization parameter for the least squares problem.
    `verbose` : bool
        Print debug information.

    Returns
    -------
    `pc` : tuple(numpy.array[float], numpy.array[float])
        The polynomial coefficients and the borders of the segments.
    '''


    return fit_piecewisecubic(xs,ys,borders,regularization_beta)

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


    f = np.load("test_data/output_curve_values.npy")

    vals = f[7,:,1].T
    mask = vals>0

    xs      = np.argwhere(mask).astype(float).flatten()
    ys      = vals[mask]
    borders = np.linspace(xs.min(), xs.max()+1,5)
    A, b = piecewisecubic_matrix(xs,ys,borders)

    coefs, residuals, rank, sing = linalg.lstsq(A,b,rcond=None)

    pc = coefs, borders

    xs_len = xs.max()-xs.min()
    new_xs = np.linspace(xs.min()-xs_len/2,xs.max()+xs_len/2,100)

    Ys1 = piecewisecubic(pc,new_xs) # Cubic extrapolation is default
    Ys2 = piecewisecubic(pc,new_xs,extrapolation="linear")
    Ys3 = piecewisecubic(pc,new_xs,extrapolation="constant")

    plt.plot(xs,ys,c='black',linewidth=2.5)
    plt.plot(new_xs,Ys1,c='r')
    plt.plot(new_xs,Ys2,c='g')
    plt.plot(new_xs,Ys3,c='b')
    plt.show()
