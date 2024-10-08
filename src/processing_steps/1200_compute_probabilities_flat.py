#! /usr/bin/python3
'''
This script computes the probabilities of the materials in the implant mask.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.paths import hdf5_root, get_plotting_dir
import h5py
from lib.py.commandline_args import add_volume, default_parser
from lib.py.helpers import row_normalize, update_hdf5
from lib.py.piecewise_cubic import piecewisecubic, smooth_fun as smooth_fun_c
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import scipy.ndimage as ndi
import tqdm

NA = np.newaxis

def save_probabilities(output_dir, Ps, sample, region_mask, field_name, value_ranges, prob_method, verbose):
    '''
    Save the probabilities `Ps` to an HDF5 file.

    Parameters
    ----------
    `output_dir` : str
        The output directory.
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
    `verbose` : int
        The verbosity level.

    Returns
    -------
    `None`
    '''

    output_path = f'{output_dir}/{sample}.h5'
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

if __name__ == '__main__':
    argparser = default_parser(__doc__)
    argparser = add_volume(argparser, 'field')
    argparser = add_volume(argparser, 'region_mask')
    argparser.add_argument('-n', '--n-segments', action='store', type=int, default=4,
        help='The number of segments to use in the piecewise cubic functions. Default is 4.')
    args = argparser.parse_args()

    hist_path = f"{hdf5_root}/processed/histograms/"
    f_labels = np.load(f"{hist_path}/{args.sample}/bins-{args.region_mask}_labeled.npz")
    input_filename  = f"{hdf5_root}/processed/histograms/{args.sample}.h5"
    output_dir = f"{hdf5_root}/processed/probabilities/"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    plotting_dir = get_plotting_dir(args.sample, args.sample_scale)
    if args.plotting:
        pathlib.Path(plotting_dir).mkdir(parents=True, exist_ok=True)

    lab = f_labels[args.field]

    try:
        model_file = h5py.File(input_filename, "r")
        g = model_file[f"{args.region_mask}/{args.field}"]
        hist, labels = g["histogram"][:], g["labels"][:]
        nmat = labels.max()
        if nmat > 2:
            # TODO: Generalize to nmat materials, also in optimize_distributions_flat.py
            print(f"Found {nmat} materials - killing off everything > 2. You gotta write this more general, man!")
            labels[labels > 2] = 0
            nmat = min(nmat, 2)

        good_xs      = [g["good_xs0"][:], g["good_xs1"][:]]
        ABCD         = [g["ABCD0"][:], g["ABCD1"][:]]
        value_ranges = g['value_ranges'][:]
        model_file.close()
    except Exception as e:
        print(f"Error in loading {args.region_mask}/{args.field} from {input_filename}: {e}")
        sys.exit(-1)

    if args.plotting:
        plt.plot(good_xs[0])
        plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_good_xs0.pdf", bbox_inches='tight')
        plt.close()

        plt.plot(good_xs[1])
        plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_good_xs1.pdf", bbox_inches='tight')
        plt.close()

        plt.imshow(hist)
        plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_hist.pdf", bbox_inches='tight')
        plt.close()

        plt.imshow(labels)
        plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_labels.pdf", bbox_inches='tight')
        plt.close()

    nx, nv = hist.shape
    xs, vs = np.arange(nx), np.arange(nv)
    ms = range(nmat)

    #---- COMPUTE PIECEWISE CUBIC REPRESENTATIONS OF DISTRIBUTIONS ---
    # Smooth piecewise polynomial representations
    n_segments_l = 10 * args.n_segments
    n_coefs_c = 2 * args.n_segments     # cubic
    n_coefs_l = n_segments_l         # linear

    pca  = np.zeros((nmat, n_coefs_c),    dtype=float)
    pcb  = np.zeros((nmat, n_coefs_c),    dtype=float)
    pcc  = np.zeros((nmat, n_coefs_c),    dtype=float)
    pcd  = np.zeros((nmat, n_coefs_c),    dtype=float)
    bins = np.zeros((nmat, args.n_segments), dtype=float)

    for m in tqdm.tqdm(ms,"Fit PWC parameters"):
        A, B, C, D = ABCD[m].T

        pca[m,:], bins[m] = smooth_fun_c(good_xs[m], A, args.n_segments)
        pcb[m,:], _       = smooth_fun_c(good_xs[m], B, args.n_segments)
        pcc[m,:], _       = smooth_fun_c(good_xs[m], C, args.n_segments)
        pcd[m,:], _       = smooth_fun_c(good_xs[m], D, args.n_segments)

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

        if args.plotting:
            plt.imshow(hist_m[m])
            plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_hist_m{m}.pdf", bbox_inches='tight')
            plt.close()

            slice = 450
            plt.plot(hist_m[m][slice], label=f'm{m}')
            plt.plot(hist[slice], label='hist')
            #plt.plot(labels[slice], label='labels')
            plt.vlines(np.argwhere(labels[slice]!=0),0,hist.max(), color='black', alpha=0.5, label='labels')
            plt.legend()
            #plt.yscale('log')
            plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_hist_m{m}_slice.pdf", bbox_inches='tight')
            plt.close()

        long_tail = (hist_m[m] < 0.001*hist_m[m].max(axis=1)[:,NA])
        long_tail |= all_xs[:,NA] > (1.00 * good_xs[m].max())
        long_tail |= all_xs[:,NA] < (good_xs[m].min() / 1.00)

        if args.plotting:
            plt.imshow(long_tail)
            plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_hist_m{m}_long_tail.pdf", bbox_inches='tight')
            plt.close()

        if m == 1:
            Cm = piecewisecubic((pcc[0], bins[0]), all_xs)
            long_tail |= all_vs[NA,:] <= Cm[:,NA]

        if m == 0:
            Cp = piecewisecubic((pcc[1], bins[1]), all_xs)
            print (all_vs[NA,:].shape, Cp[:,NA].shape)
            long_tail |= all_vs[NA,:] >= Cp[:,NA]

        hist_m[m][long_tail] = 0

    hist_modeled = np.sum(hist_m, axis=0)

    if args.plotting:
        plt.imshow(hist_modeled)
        plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_hist_modeled.pdf", bbox_inches='tight')
        plt.close()

    for m in ms:
        P_m[m] = hist_m[m] / np.maximum(hist + (hist==0), hist_modeled)
        P_m[m] = ndi.gaussian_filter(P_m[m], 10, mode='constant', cval=0)

    P_modeled = np.minimum(np.sum(P_m, axis=0), 1)

    save_probabilities(output_dir, P_m, args.sample, args.region_mask, args.field, value_ranges, "optimized_distributions")

    if args.plotting:
        plt.ion()
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        line1, = ax.plot(vs, np.zeros_like(hist[0]), 'g-')
        line3, = ax.plot(vs, np.zeros_like(hist[0]), 'b:')
        line4, = ax.plot(vs, np.zeros_like(hist[0]), 'r:')
        line2, = ax.plot(vs, hist[0], 'black',linewidth=4)

        plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_seven.pdf", bbox_inches='tight')

    colors = ['b','r']
    lines  = [line3,line4]

    if args.plotting:
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
            plt.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_fourth.pdf", bbox_inches='tight')
            plt.close()

        fig = plt.figure(figsize=(10,10))
        axarr = fig.subplots(3,2)
        fig.suptitle(f'{args.sample} {args.region_mask}')
        axarr[0,0].imshow(row_normalize(hist,hist.max(axis=1)), cmap='bone')
        axarr[0,0].set_title(f"{args.field}-field 2D Histogram")
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

        fig.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}.pdf", bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(15,15))
        axarr = fig.subplots(2,2)
        fig.suptitle(f'{args.sample} {args.region_mask}')
        axarr[0,0].imshow(row_normalize(hist,hist.max(axis=1)), cmap='bone')
        axarr[0,0].set_title(f"(a) {args.field}-field 2D Histogram")

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

        fig.savefig(f"{plotting_dir}/compute_probabilities_{args.field}_{args.region_mask}_10.pdf", bbox_inches='tight')
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
        fig.savefig(f"{plotting_dir}/hist_slice_{args.field}_{args.region_mask}.pdf", bbox_inches='tight')
        plt.show()