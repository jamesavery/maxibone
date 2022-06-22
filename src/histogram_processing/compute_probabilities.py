import os, sys, tqdm, numpy as np, matplotlib.pyplot as plt, numpy.linalg as la, scipy.ndimage as ndi, scipy.optimize as opt, time
sys.path.append(sys.path[0]+"/../")
#from piecewise_linear import piecewiselinear_matrix, piecewiselinear, smooth_fun as smooth_fun_l
from piecewise_cubic import piecewisecubic_matrix, piecewisecubic, smooth_fun as smooth_fun_c
from config.paths import commandline_args, hdf5_root as hdf5_root
from distributions import *
from helper_functions import *
na = np.newaxis


# TODO: Til fælles fil.
def save_probabilities(Ps,sample, region_mask,field_name, value_ranges, prob_method):
    output_path = f'{hdf5_root}/processed/probabilities/{sample}.h5'
    update_hdf5(
        output_path,
        group_name = f'{prob_method}/{region_mask}',
        datasets = { 'value_ranges' : value_ranges },
        attributes = {}
    )
    for m,P in enumerate(Ps):
        update_hdf5(
            output_path,
            group_name = f'{prob_method}/{region_mask}/{field_name}',
            datasets = {
                f'P{m}': P
            }
        )


def evaluate_2d(G, xs, vs):
    m, bins_c, (pca,pcb,pcc,pcd) = G

    A, B, C, D = ABCD[m].T    

    gax = piecewisecubic((pca,bins_c), good_xs[m])
    gbx = piecewisecubic((pcb,bins_c), good_xs[m])
    gcx = piecewisecubic((pcc,bins_c), good_xs[m])
    gdx = piecewisecubic((pcd,bins_c), good_xs[m])
    
    
    ax = piecewisecubic((pca,bins_c), xs,extrapolation='constant')[:,na]
    bx = piecewisecubic((pcb,bins_c), xs,extrapolation='constant')[:,na]
    cx = piecewisecubic((pcc,bins_c), xs)[:,na]
    dx = piecewisecubic((pcd,bins_c), xs,extrapolation='constant')[:,na]

    # ax[(xs[:,na]<bins[0])|(xs[:,na]>bins[-1])] = 0
    # bx[(xs[:,na]<bins[0])|(xs[:,na]>bins[-1])] = 0
    # cx[(xs[:,na]<bins[0])|(xs[:,na]>bins[-1])] = 0
    # dx[(xs[:,na]<bins[0])|(xs[:,na]>bins[-1])] = 0
    
    # plt.plot(good_xs[m],A,c='r'); plt.plot(good_xs[m],gax,c='black',linewidth=3); plt.plot(xs, ax); plt.show();
    # plt.plot(good_xs[m],B,c='g'); plt.plot(good_xs[m],gbx,c='black',linewidth=3); plt.plot(xs, bx); plt.show();
    # plt.plot(good_xs[m],C,c='b'); plt.plot(good_xs[m],gcx,c='black',linewidth=3); plt.plot(xs, cx); plt.show();
    # plt.plot(good_xs[m],D,c='black'); plt.plot(good_xs[m],gdx,c='black',linewidth=3); plt.plot(xs, dx); plt.show();        

    image = ((ax*ax)*np.exp(-(bx*bx)*np.abs(vs[na,:]-cx)**(dx*dx)))
    plt.imshow(image); plt.show()
    return image


hist_path = f"{hdf5_root}/processed/histograms/"
sample, region_mask, field, n_segments_c, debug = commandline_args({"sample":"<required>",
                                                                    "region_mask":"<required>",
                                                                    "field":"edt",
                                                                    "n_segments": 4,
                                                                    "debug":8
})


try:
    model_filename = f"{hdf5_root}/processed/histograms/{sample}.h5"
    model_file     = h5py.File(model_filename,"r") 
    g = model_file[f"{region_mask}/{field}"]
    hist, labels = g["histogram"][:], g["labels"][:]
    nmat = labels.max()

    # TODO: Generalize to nmat materials, also in optimize_distributions_flat.py
    assert(nmat==2)    
    good_xs      = [g["good_xs0"][:], g["good_xs1"][:]]
    ABCD         = [g["ABCD0"][:], g["ABCD1"][:]]

    model_file.close()
except Exception as e:
    print(f"Error in loading {region_mask}/{field} from {model_filename}: {e}")
    sys.exit(-1)

nx, nv = hist.shape
xs, vs = np.arange(nx), np.arange(nv)
ms = range(nmat)

#---- COMPUTE PIECEWISE CUBIC REPRESENTATIONS OF DISTRIBUTIONS ---
# Smooth piecewise polynomial representations
n_segments_l = 10*n_segments_c
n_coefs_c = 2*n_segments_c       # cubic
n_coefs_l = n_segments_l         # linear

pca  = np.zeros((nmat,n_coefs_c),dtype=float)
pcb  = np.zeros((nmat,n_coefs_c),dtype=float)
pcc  = np.zeros((nmat,n_coefs_c),dtype=float)
pcd  = np.zeros((nmat,n_coefs_c),dtype=float)
bins = np.zeros((nmat,n_segments_c),dtype=float)

for m in tqdm.tqdm(ms,"Fit PWC parameters"):
    A, B, C, D = ABCD[m].T
    
    pca[m,:], bins[m] = smooth_fun_c(good_xs[m],A,n_segments_c)
    pcb[m,:], _    = smooth_fun_c(good_xs[m],B,n_segments_c)    
    pcc[m,:], _    = smooth_fun_c(good_xs[m],C,n_segments_c)    
    pcd[m,:], _    = smooth_fun_c(good_xs[m],D,n_segments_c)

    # gax = piecewisecubic((pca[m],bins[0]), good_xs[m])
    # gbx = piecewisecubic((pcb[m],bins[0]), good_xs[m])
    # gcx = piecewisecubic((pcc[m],bins[0]), good_xs[m])
    # gdx = piecewisecubic((pcd[m],bins[0]), good_xs[m])
    # fig.suptitle("At construction")
    # plt.plot(good_xs[0],A,c='r'); plt.plot(good_xs[0],gax,c='black',linewidth=3);  plt.show();
    # plt.plot(good_xs[0],B,c='g'); plt.plot(good_xs[0],gbx,c='black',linewidth=3);  plt.show();
    # plt.plot(good_xs[0],C,c='b'); plt.plot(good_xs[0],gcx,c='black',linewidth=3);  plt.show();
    # plt.plot(good_xs[0],D,c='black'); plt.plot(good_xs[0],gdx,c='black',linewidth=3);  plt.show();        

Gs = [(m,bins[m], np.array([pca[m],pcb[m],pcc[m],pcd[m]])) for m in range(nmat)]

hist_m = np.zeros((2,)+hist.shape,dtype=float)
P_m = np.zeros((2,)+hist.shape,dtype=float)

for m in ms:
    hist_m[m] = evaluate_2d(Gs[m],xs,vs)
hist_modeled = np.sum(hist_m,axis=0)

for m in ms:
    P_m[m]    = hist_m[m]/np.maximum(hist + (hist==0), hist_modeled)
    
P_modeled = np.minimum(np.sum(P_m,axis=0), 1)


    
##---- TODO: STICK THE DEBUG-PLOTTING FUNCTIONS SOMEWHERE CENTRAL
if (debug&7):
    plt.ion()
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    line1, = ax.plot(vs, np.zeros_like(hist[0]), 'g-')
    line3, = ax.plot(vs, np.zeros_like(hist[0]), 'b:')
    line4, = ax.plot(vs, np.zeros_like(hist[0]), 'r:')
    line2, = ax.plot(vs, hist[0],  'black',linewidth=4)
    
    plt.show()

    
if(debug==4):
    colors = ['b','r']
    lines  = [line3,line4]

    for i,x in enumerate(xs):
        line1.set_ydata(hist_modeled[i])
        line2.set_ydata(hist[i])
        ax.collections.clear()
        ax.fill_between(vs,hist_modeled[i],color='green',alpha=0.5)
        for m in ms:
            lines[m].set_ydata(hist_m[m,i])                
            ax.fill_between(vs,hist_m[m][i],color=colors[m],alpha=0.7)
        
        ax.set_title(f"x={x}")
        ax.relim()
        ax.autoscale_view()    
        fig.canvas.draw()
        fig.canvas.flush_events()


if (debug==8):
    fig = plt.figure(figsize=(10,10))
    axarr = fig.subplots(3,2)
    fig.suptitle(f'{sample} {region_mask}') 
    axarr[0,0].imshow(hist)
    axarr[0,0].set_title(f"{field}-field 2D Histogram")
    axarr[0,1].imshow(row_normalize(hist_modeled,hist.max(axis=1)))
    axarr[0,1].set_title("Reconstructed 2D Histogram")
    axarr[1,0].imshow(row_normalize(hist_m[0],hist.max(axis=1)))
    axarr[1,0].set_title("Material 1")
    axarr[1,1].imshow(row_normalize(hist_m[1],hist.max(axis=1)))
    axarr[1,1].set_title("Material 2")

    axarr[2,0].imshow(P_m[0])
    axarr[2,0].set_title("P(m0|x,v)")
    axarr[2,1].imshow(P_m[1])    
    axarr[2,1].set_title("P(m1|x,v)")    
    fig.tight_layout()

    output_dir = f"{hdf5_root}/processed/probabilities/{sample}/"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)        
    fig.savefig(f"{output_dir}/compute_probabilities_{field}_{region_mask}.png")
    plt.show()    


