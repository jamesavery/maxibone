#! /usr/bin/python3
'''
Optimize the distributions of the materials in the 2D histogram.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')
from config.paths import hdf5_root
from lib.py.distributions import powers
from lib.py.helpers import commandline_args, row_normalize, update_hdf5
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
na = np.newaxis

hist_path = f"{hdf5_root}/processed/histograms/"
sample, region_mask, field, stride, verbose = commandline_args({
    "sample":"<required>",
    "region_mask":"<required>",
    "field":"edt",
    "stride": 4,
    "verbose":8
})

f_hist   = np.load(f"{hist_path}/{sample}/bins-{region_mask}.npz")
f_labels = np.load(f"{hist_path}/{sample}/bins-{region_mask}_labeled.npz")

def material_points(labs, material_id):
    '''
    Returns the x and y coordinates of the points in the 2D histogram that are
    labeled with the given material_id.

    Parameters
    ----------
    `labs` : numpy.array[uint64]
        The 2D histogram with the labels of the materials.
    `material_id` : int
        The material id to look for.

    Returns
    -------
    `xs` : numpy.array[float]
        The x coordinates of the points.
    `ys` : numpy.array[float]
        The y coordinates of the points.
    '''

    mask = labs == material_id
    xs, ys = np.argwhere(mask).astype(float).T

    return xs, ys

#TODO: Weight importance in piecewise cubic fitting

field_id = { 'edt': 0, 'gauss': 1, 'gauss+edt': 2}; # TODO: Get from data

hist = f_hist["field_bins"][field_id[field]][::stride,::stride]
lab  = f_labels[field][::stride,::stride]

if verbose >= 2:
    plt.imshow(lab)
    plt.show()

nmat    = lab.max()
(nx,nv) = hist.shape
xs = np.arange(0,nx*stride,stride)
vs = np.arange(0,nv*stride,stride)

# Compute start parameter approximation samples
labm = np.array([lab == m+1 for m in range(nmat)])
amx = np.sqrt(np.array([np.max(labm[m] * hist, axis=1) for m in range(nmat)]))
cmx = np.array([np.argmax(labm[m] * hist, axis=1) for m in range(nmat)]) * stride

starts = np.argmax(labm, axis=-1) * stride
ends   = (nv - np.argmax(labm[...,::-1], axis=-1)) * stride
ends[ends == nv] = 0

widths = np.abs(np.concatenate([ [(cmx[1]-cmx[0]) / 2], (cmx[1:]-cmx[:-1]) / 2 ]).astype(float))
bmx    = np.sqrt(3/(widths**2+(widths==0)))
dmx    = np.sqrt(np.ones_like(bmx)*2)
goodix  = amx>0

print(f"Optimizing distributions for {field} with {lab.max()} materials")

if (verbose >= 3):
    plt.ion()
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    line1, = ax.plot(vs, np.zeros_like(hist[0]), 'g-')
    line3, = ax.plot(vs, np.zeros_like(hist[0]), 'b:')
    line4, = ax.plot(vs, np.zeros_like(hist[0]), 'r:')
    line2, = ax.plot(vs, hist[0], 'black', linewidth=4)
    plt.show()

def opt_bs(bs, *args):
    '''
    Optimize the b values for a given set of a, c, and d values.

    Parameters
    ----------
    `bs` : numpy.array[float]
        The b values to optimize.
    `args` : tuple(numpy.array[float], numpy.array[float])
        The a, c, and d values and the histogram.

    Returns
    -------
    `energy` : float
        The energy of the distribution.
    '''

    abcd0, vs = args
    n = len(abcd0) // 4
    abcd = abcd0.copy()
    abcd[n:2*n] = bs
    ax.set_title(f"bs = {bs}")

    return energy1d(abcd, args)

def opt_bds(bds, *args):
    '''
    Optimize the b and d values for a given set of a and c values.

    Parameters
    ----------
    `bds` : numpy.array[float]
        The b and d values to optimize.
    `args` : tuple(numpy.array[float], numpy.array[float])
        The a and c values and the histogram.

    Returns
    -------
    `energy` : float
        The energy of the distribution.
    '''

    abcd0, vs = args
    n = len(abcd0) // 4
    abcd = abcd0.copy()
    abcd[n:2*n] = bds[:n]
    abcd[3*n:4*n] = bds[n:]
    ax.set_title(f"bs = {bds[:n]}, ds = {bds[n:]}")
    return energy1d(abcd, args)

def opt_all(abcd, *args):
    '''
    Optimize the a, b, c, and d values for a given histogram.

    Parameters
    ----------
    `abcd` : numpy.array[float]
        The a, b, c, and d values to optimize.
    `args` : tuple[int, float, numpy.array[float], numpy.array[float], numpy.array[float]]
        The index, x value, the a, b, c, and d values, and the histogram.

    Returns
    -------
    `energy` : float
        The energy of the distribution
    '''

    i, x, abcd0, vs, hist_x = args
    model    = np.sum(powers(vs, abcd), axis=0)
    residual = hist_x - model

    dx = 1 / len(vs)
    E1 = dx * np.sum(residual * residual)
    E2 = dx * np.sum((residual < 0) * residual * residual)

    n = len(abcd) // 4
    A, B, C, D = abcd[:n], abcd[n:2*n], abcd[2*n:3*n], abcd[3*n:4*n]
    Ecloseness = np.sum(1 / (np.abs(C[1:] - C[:-1]) + 0.001))

    if(verbose >= 3):
        line1.set_ydata(model)
        ax.set_title(f"{x}: a = {np.round(A*A,1)}, b = {np.round(B*B,1)}, c = {np.round(C,1)}, d = {np.round(D*D,1)}")
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    return E1 + 1e2*Ecloseness + E2

good_xs = [[] for m in range(nmat)] # BE CAREFUL: [[]] * nmat makes MULTIPLE REFERENCES TO THE SAME LIST MEMORY. Two hours bughunting for that one.
good_is = [[] for m in range(nmat)]
ABCDm   = [[] for m in range(nmat)]
ABCD    = []
ABCD_ms = []
ABCD_xs = []
ABCD_is = []
m_max   = 0

# FLAT OPTIMIZATION
for i, x in enumerate(xs):
    ms = np.array([m for m in range(nmat) if labm[m][i].max() > 0])
    n  = len(ms)
    if (n > 0):
        abcd0 = np.array([amx[ms,i], bmx[ms,i], cmx[ms,i], dmx[ms,i]]).flatten()

        if (verbose >= 2):
            model = powers(vs, abcd0)
            line1.set_ydata(np.sum(model, axis=0))
            line2.set_ydata(hist[i])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

        if(verbose >= 3):
            ax.set_title(f"x = {x}")
            line2.set_ydata(hist[i])

        constants  = i, x, abcd0, vs, hist[i]
        midpoints = np.maximum(
            np.concatenate([(cmx[ms,i][1:] + cmx[ms,i][:-1]) / 2, [vs.max()]]),
            0.9*cmx[ms,i])

        bounds = opt.Bounds(np.concatenate([0.3*amx[ms,i],
                                            np.minimum(0.5, np.maximum(0.1*bmx[ms,i], 1e-3)),
                                            starts[ms,i], #0.9*cmx[ms,i],
                                            np.sqrt(1.5) * np.ones(ms.shape)]),
                            np.concatenate([1.1*amx[ms,i],
                                            np.minimum(2.0*bmx[ms,i], 0.7),
                                            ends[ms,i], #midpoints,
                                            np.sqrt(2.5) * np.ones(ms.shape)]),
                            True)
        opt_result = opt.minimize(opt_all, abcd0, constants, bounds=bounds)

        abcd, success = opt_result['x'], opt_result['success']

        if success:
            print(f"Histogram at x={x} converged successfully for materials {ms}.")
            ABCD    += [abcd]
            ABCD_ms += [ms]
            ABCD_xs += [x]
            ABCD_is += [i]
            m_max    = max(m_max, np.max(ms))

            for im, m in enumerate(ms):
                good_xs[m] += [x]
                good_is[m] += [i]
                ABCDm[m]   += [abcd.reshape(4,n)[:,im]]
                print(f"ABCDm[{m}]    += {[abcd.reshape(4,n)[:,im]].copy()}")

        if(verbose >= 5):
            colors = ['r', 'orange']
            lines  = [line3, line4]
            model = powers(vs, abcd)
            n = len(abcd) // 4
            A, B, C, D = abcd[:n], abcd[n:2*n], abcd[2*n:3*n], abcd[3*n:4*n]

            fig.suptitle(f"x = {x}, ms = {ms}")
            line1.set_ydata(np.sum(model, axis=0))
            line2.set_ydata(hist[i])
            ax.collections.clear()
            ax.fill_between(vs, np.sum(model, axis=0), color='grey', alpha=0.5)
            for i, m in enumerate(ms):
                lines[m].set_ydata(model[i])
                ax.fill_between(vs, model[i], color=colors[m], alpha=0.7)

            ax.set_title(f"a = {np.round(A*A, 1)}, 1/b = {np.round(1 / (B*B), 3)}, c = {np.round(C, 1)}, d = {np.round(D*D, 1)}")
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.savefig(f"opt_debug-{i:03}.png", bbox_inches='tight')

hist_modeled = np.zeros_like(hist)
hist_m = np.zeros((m_max+1,) + hist.shape, dtype=float)

for i, gi in enumerate(ABCD_is):
    abcd = ABCD[i]
    ms   = ABCD_ms[i]
    model = powers(vs, abcd)
    hist_modeled[gi] = np.sum(model, axis=0)
    hist_m[ms, gi] = model

if (verbose >= 6):
    fig = plt.figure(figsize=(10,10))
    axarr = fig.subplots(2,2)
    fig.suptitle(f'{sample} {region_mask}') # or plt.suptitle('Main title')
    axarr[0,0].imshow(row_normalize(hist, hist.max(axis=1)))
    axarr[0,0].set_title(f"{field}-field 2D Histogram")
    axarr[0,1].imshow(row_normalize(hist_modeled, hist.max(axis=1)))
    axarr[0,1].set_title("Remodeled 2D Histogram")
    axarr[1,0].imshow(row_normalize(hist_m[0], hist.max(axis=1)))
    axarr[1,0].set_title("Material 1")
    axarr[1,1].imshow(row_normalize(hist_m[1], hist.max(axis=1)))
    axarr[1,1].set_title("Material 2")
    fig.tight_layout()
    fig.savefig(f"{hdf5_root}/processed/histograms/{sample}/hist_vs_modeled_{field}_{region_mask}.png", bbox_inches='tight')
    plt.show()

# TODO: How to generalize to arbitrarily many materials?
update_hdf5(f"{hdf5_root}/processed/histograms/{sample}.h5",
            group_name=f"{region_mask}/{field}",
            datasets={
                "value_ranges": f_hist["value_ranges"][:],
                "histogram": f_hist["field_bins"][field_id[field]][:],
                "labels": f_labels[field][:],
                "good_xs0": np.array(good_xs[0]),
                "good_xs1": np.array(good_xs[1]),
                "ABCD0":    np.array(ABCDm[0]),
                "ABCD1":    np.array(ABCDm[1])
            })
