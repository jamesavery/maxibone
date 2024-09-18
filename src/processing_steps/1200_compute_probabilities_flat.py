#! /usr/bin/python3
'''
This script computes the probabilities of the materials in the implant mask.
'''
import sys
sys.path.append(sys.path[0]+"/../")
import matplotlib
matplotlib.use('Agg')
from config.paths import hdf5_root
import h5py
from lib.py.helpers import commandline_args, row_normalize, update_hdf5
from lib.py.piecewise_cubic import piecewisecubic, smooth_fun as smooth_fun_c
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import scipy.ndimage as ndi
import tqdm

NA = np.newaxis

# TODO: Til fælles fil.
def save_probabilities(Ps, sample, region_mask, field_name, value_ranges, prob_method):
    '''
    Save the probabilities `Ps` to an HDF5 file.

    Parameters
    ----------
    `Ps` : list[numpy.array[float32]]
        List of probabilities for each material.
    `sample` : str
        Sample name.
    `region_mask` : str
        Region mask name.
    `field_name` : str
        Field name.
    `value_ranges` : numpy.array[float32]
        Value ranges for the field.
    `prob_method` : str
        Method used to compute the probabilities.

    Returns
    -------
    `None`
    '''

    output_path = f'{hdf5_root}/processed/probabilities/{sample}.h5'
    if verbose >= 1:
        print(f"output_path = {output_path}")
        print(f"group_name1 = {prob_method}/{region_mask}")
        print(f"group_name2 = {prob_method}/{region_mask}/{field_name}")

    update_hdf5(
        output_path,
        group_name = f'{prob_method}/{region_mask}',
        datasets = { 'value_ranges' : value_ranges },
        attributes = {}
    )

    for m, P in enumerate(Ps):
        if verbose >= 1: print(f"Storing {P.shape} probabilities P{m}")
        update_hdf5(
            output_path,
            group_name = f'{prob_method}/{region_mask}/{field_name}',
            datasets = {
                f'P{m}': P
            }
        )

def evaluate_2d(G, xs, vs):
    '''
    Evaluate the 2D distribution given by the piecewise cubic functions in `G`.

    Parameters
    ----------
    `G` : tuple[int, numpy.array[float32], tuple[numpy.array[float32], numpy.array[float32], numpy.array[float32], numpy.array[float32]]]
        Tuple containing the piecewise cubic functions.
    `xs` : numpy.array[int]
        X-values.
    `vs` : numpy.array[int]
        V-values.

    Returns
    -------
    `image` : numpy.array[float32]
        The 2D distribution.
    '''

    m, bins_c, (pca, pcb, pcc, pcd) = G

    A, B, C, D = ABCD[m].T

    ax = piecewisecubic((pca, bins_c), xs, extrapolation='constant')[:,NA]
    bx = piecewisecubic((pcb, bins_c), xs, extrapolation='constant')[:,NA]
    cx = piecewisecubic((pcc, bins_c), xs, extrapolation='cubic')[:,NA]
    dx = piecewisecubic((pcd, bins_c), xs, extrapolation='constant')[:,NA]

    image = ((ax*ax) * np.exp(-(bx*bx) * np.abs(vs[NA,:] - cx)**(dx*dx)))

    return image

sample, region_mask, field_name, n_segments_c, verbose = commandline_args({
    "sample" : "<required>",
    "region_mask" : "<required>",
    "field_name" : "edt",
    "n_segments" : 4,
    "verbose" : 8
})
hist_path = f"{hdf5_root}/processed/histograms/"
f_labels = np.load(f"{hist_path}/{sample}/bins-{region_mask}_labeled.npz")
input_filename  = f"{hdf5_root}/processed/histograms/{sample}.h5"
output_dir = f"{hdf5_root}/processed/probabilities/{sample}/"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

lab = f_labels[field_name]

try:
    model_file = h5py.File(input_filename, "r")
    g = model_file[f"{region_mask}/{field_name}"]
    hist, labels = g["histogram"][:], g["labels"][:]
    nmat = labels.max()
    if nmat > 2:
        # TODO: ARH FOR HELVEDE. GOER DET RIGTIGT, NAAR DER ER MERE TID.
        # TODO: Generalize to nmat materials, also in optimize_distributions_flat.py
        print(f"Found {nmat} materials - killing off everything > 2. You gotta write this more general, man!")
        labels[labels > 2] = 0
        nmat = min(nmat, 2)

    good_xs      = [g["good_xs0"][:], g["good_xs1"][:]]
    ABCD         = [g["ABCD0"][:], g["ABCD1"][:]]
    value_ranges = g['value_ranges'][:]
    model_file.close()
except Exception as e:
    print(f"Error in loading {region_mask}/{field_name} from {input_filename}: {e}")
    sys.exit(-1)

if verbose >= 1:
    plt.plot(good_xs[0])
    plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_good_xs0.png", bbox_inches='tight')
    plt.clf()
    plt.plot(good_xs[1])
    plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_good_xs1.png", bbox_inches='tight')
    plt.clf()

if verbose >= 1:
    plt.imshow(hist)
    plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_hist.png", bbox_inches='tight')
    plt.clf()
    plt.imshow(labels)
    plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_labels.png", bbox_inches='tight')
    plt.clf()

nx, nv = hist.shape
xs, vs = np.arange(nx), np.arange(nv)
ms = range(nmat)

#---- COMPUTE PIECEWISE CUBIC REPRESENTATIONS OF DISTRIBUTIONS ---
# Smooth piecewise polynomial representations
n_segments_l = 10 * n_segments_c
n_coefs_c = 2 * n_segments_c     # cubic
n_coefs_l = n_segments_l         # linear

pca  = np.zeros((nmat, n_coefs_c),    dtype=float)
pcb  = np.zeros((nmat, n_coefs_c),    dtype=float)
pcc  = np.zeros((nmat, n_coefs_c),    dtype=float)
pcd  = np.zeros((nmat, n_coefs_c),    dtype=float)
bins = np.zeros((nmat, n_segments_c), dtype=float)

for m in tqdm.tqdm(ms,"Fit PWC parameters"):
    A, B, C, D = ABCD[m].T

    pca[m,:], bins[m] = smooth_fun_c(good_xs[m], A, n_segments_c)
    pcb[m,:], _       = smooth_fun_c(good_xs[m], B, n_segments_c)
    pcc[m,:], _       = smooth_fun_c(good_xs[m], C, n_segments_c)
    pcd[m,:], _       = smooth_fun_c(good_xs[m], D, n_segments_c)

    # gax = piecewisecubic((pca[m],bins[0]), good_xs[m])
    # gbx = piecewisecubic((pcb[m],bins[0]), good_xs[m])
    # gcx = piecewisecubic((pcc[m],bins[0]), good_xs[m])
    # gdx = piecewisecubic((pcd[m],bins[0]), good_xs[m])
    # fig.suptitle("At construction")
    # plt.plot(good_xs[0],A,c='r'); plt.plot(good_xs[0],gax,c='black',linewidth=3);  plt.show();
    # plt.plot(good_xs[0],B,c='g'); plt.plot(good_xs[0],gbx,c='black',linewidth=3);  plt.show();
    # plt.plot(good_xs[0],C,c='b'); plt.plot(good_xs[0],gcx,c='black',linewidth=3);  plt.show();
    # plt.plot(good_xs[0],D,c='black'); plt.plot(good_xs[0],gdx,c='black',linewidth=3);  plt.show();

Gs = [(m, bins[m], np.array([pca[m], pcb[m], pcc[m], pcd[m]])) for m in range(nmat)]

hist_m = np.zeros((2,) + hist.shape, dtype=float)
P_m = np.zeros((2,) + hist.shape, dtype=float)

all_vs = np.arange(4096) # TODO: AUTOMATISK
all_xs = np.arange(2048) # TODO: AUTOMATISK

for m in ms:
    hist_m[m] = evaluate_2d(Gs[m], xs, vs)

    if verbose >= 1:
        plt.imshow(hist_m[m])
        plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_hist_m{m}.png", bbox_inches='tight')
        plt.clf()
        slice = 450
        plt.plot(hist_m[m][slice], label=f'm{m}')
        plt.plot(hist[slice], label='hist')
        #plt.plot(labels[slice], label='labels')
        plt.vlines(np.argwhere(labels[slice]!=0),0,hist.max(), color='black', alpha=0.5, label='labels')
        plt.legend()
        #plt.yscale('log')
        plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_hist_m{m}_slice.png", bbox_inches='tight')
        plt.clf()

    long_tail = (hist_m[m] < 0.001*hist_m[m].max(axis=1)[:,NA])
    long_tail |= all_xs[:,NA] > (1.00 * good_xs[m].max())
    long_tail |= all_xs[:,NA] < (good_xs[m].min() / 1.00)

    if verbose >= 1:
        plt.imshow(long_tail)
        plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_hist_m{m}_long_tail.png", bbox_inches='tight')
        plt.clf()

    if m==1:
        Cm = piecewisecubic((pcc[0], bins[0]), all_xs)
        long_tail |= all_vs[NA,:] <= Cm[:,NA]

    if m==0:
        Cp = piecewisecubic((pcc[1], bins[1]), all_xs)
        print (all_vs[NA,:].shape, Cp[:,NA].shape)
        long_tail |= all_vs[NA,:] >= Cp[:,NA]

    hist_m[m][long_tail] = 0

hist_modeled = np.sum(hist_m, axis=0)

if verbose >= 1:
    plt.imshow(hist_modeled)
    plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_hist_modeled.png", bbox_inches='tight')
    plt.clf()

for m in ms:
    P_m[m]    = hist_m[m] / np.maximum(hist + (hist==0), hist_modeled)
    P_m[m]    = ndi.gaussian_filter(P_m[m], 10, mode='constant', cval=0)

P_modeled = np.minimum(np.sum(P_m, axis=0), 1)

save_probabilities(P_m,sample, region_mask,field_name, value_ranges, "optimized_distributions")

##---- TODO: STICK THE DEBUG-PLOTTING FUNCTIONS SOMEWHERE CENTRAL
if (verbose & 7):
    plt.ion()
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    line1, = ax.plot(vs, np.zeros_like(hist[0]), 'g-')
    line3, = ax.plot(vs, np.zeros_like(hist[0]), 'b:')
    line4, = ax.plot(vs, np.zeros_like(hist[0]), 'r:')
    line2, = ax.plot(vs, hist[0], 'black',linewidth=4)

    plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_seven.png", bbox_inches='tight')

if (verbose == 4):
    colors = ['b','r']
    lines  = [line3,line4]

    for i, x in enumerate(xs):
        line1.set_ydata(hist_modeled[i])
        line2.set_ydata(hist[i])
        for collection in ax.collections:
            collection.remove()
        #ax.collections.clear()
        ax.fill_between(vs,hist_modeled[i],color='green',alpha=0.5)
        for m in ms:
            lines[m].set_ydata(hist_m[m,i])
            ax.fill_between(vs,hist_m[m][i],color=colors[m],alpha=0.7)

        ax.set_title(f"x={x}")
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_fourth.png", bbox_inches='tight')
    plt.clf()

if (verbose >= 8):
    fig = plt.figure(figsize=(10,10))
    axarr = fig.subplots(3,2)
    fig.suptitle(f'{sample} {region_mask}')
    axarr[0,0].imshow(row_normalize(hist,hist.max(axis=1)), cmap='bone')
    axarr[0,0].set_title(f"{field_name}-field 2D Histogram")
    axarr[0,1].imshow(row_normalize(hist_modeled,hist_modeled.max(axis=1)), cmap='bone')
    axarr[0,1].set_title("Reconstructed 2D Histogram from model")
    axarr[1,0].imshow(row_normalize(hist_m[0], hist_modeled.max(axis=1)))
    axarr[1,0].set_title("Material 1")
    axarr[1,1].imshow(row_normalize(hist_m[1], hist_modeled.max(axis=1)))
    axarr[1,1].set_title("Material 2")

    axarr[2,0].imshow(P_m[0])
    axarr[2,0].set_title("P(m0|x,v)")
    axarr[2,1].imshow(P_m[1])
    axarr[2,1].set_title("P(m1|x,v)")
    fig.tight_layout()

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}.png", bbox_inches='tight')
    plt.clf()

if (verbose >= 10):
    fig = plt.figure(figsize=(15,15))
    axarr = fig.subplots(2,2)
    fig.suptitle(f'{sample} {region_mask}')
    axarr[0,0].imshow(row_normalize(hist,hist.max(axis=1)), cmap='bone')
    axarr[0,0].set_title(f"(a) {field_name}-field 2D Histogram")

    labim = np.zeros(lab.shape+(3,), dtype=np.uint8)
    labim[lab==1] = (255,0,0)
    labim[lab==2] = (255,240,0)
    axarr[0,1].set_title(f"(b) Computed ridges")
    axarr[0,1].imshow(labim)

    axarr[1,0].imshow(row_normalize(hist_modeled, hist_modeled.max(axis=1)), cmap='bone')
    axarr[1,0].set_title("(c) Reconstructed 2D Histogram from model")

    axarr[1,1].set_title("(d) m0 and m1 model distributions")
    mx    = hist_m.max()
    m0    = row_normalize(hist_m[0], hist_modeled.max(axis=1))
    m1    = row_normalize(hist_m[1], hist_modeled.max(axis=1))
    modim = np.zeros(m0.shape + (3,), dtype=np.float32)

    modim[...,0] = np.minimum(1, .8*m0 +   m1)
    modim[...,1] = np.minimum(1,        .9*m1)
    axarr[1,1].imshow(modim)
    fig.tight_layout()

    fig.savefig(f"{output_dir}/compute_probabilities_{field_name}_{region_mask}_10.png", bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(15,5))
    colors = ['r', 'orange']
    lines  = [line3, line4]

    i = 1000
    i0 = list(good_xs[0]).index(i)
    i1 = list(good_xs[1]).index(i)
    a0,b0,c0,d0 = ABCD[0][i0]
    a1,b1,c1,d1 = ABCD[1][i1]

    A,B,C,D = [[a0*a0, a1*a1], [b0*b0, b1*b1], [c0, c1], [d0*d0, d1*d1]]

    fig.suptitle(f"")
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    ax.plot(vs, hist_modeled[i], 'g-')
    ax.plot(vs, hist[i], 'black', linewidth=4)
    ax.plot(vs, hist_m[0][i], 'r:')
    ax.plot(vs, hist_m[1][i], 'y:')
    ax.fill_between(vs, hist_modeled[i], color='grey', alpha=0.5)
    ax.fill_between(vs, hist_m[0][i], color='r', alpha=0.7)
    ax.fill_between(vs, hist_m[1][i], color='orange', alpha=0.7)

    ax.set_title(f"(e) 1D histogram slice at field index {i}: a = {np.round(A,1)}, b = {np.round(B,3)}, c = {np.round(C,1)}, d = {np.round(D,1)}")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/hist_slice_{field_name}_{region_mask}.png", bbox_inches='tight')
    plt.show()
