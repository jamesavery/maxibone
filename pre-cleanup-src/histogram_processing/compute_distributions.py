import os, sys, tqdm, numpy as np, matplotlib.pyplot as plt, numpy.linalg as la, scipy.ndimage as ndi, scipy.optimize as opt
sys.path.append(sys.path[0]+"/../")
from piecewise_cubic import piecewisecubic_matrix, piecewisecubic, smooth_fun
from config.paths import commandline_args, hdf5_root as hdf5_root
na = np.newaxis

hist_path = f"{hdf5_root}/processed/histograms/"
sample, suffix, n_segments = commandline_args({"sample":"<required>",
                                               "suffix":"<required>",
                                               "n_segments": 8})

f_hist   = np.load(f"{hist_path}/{sample}/bins{suffix}.npz")
f_labels = np.load(f"{hist_path}/{sample}/bins{suffix}_labeled.npz")

def material_points(labs,material_id):
    mask = labs==material_id    
    xs, ys = np.argwhere(mask).astype(float).T    
    return xs,ys

#TODO: Weight importance in piecewise cubic fitting

hist = f_hist["field_bins"][0][::4,::4]
#sums = np.sum(hist,axis=1)[:,na]
#hist = hist/(sums + (sums==0))
lab  = f_labels["edt"][::4,::4]

nmat    = lab.max()
(nx,nv) = hist.shape
xs = np.arange(nx)
vs = np.arange(nv)

# Compute start parameter approximation samples
labm = [ndi.grey_erosion(lab==m+1,5) for m in range(nmat)]
amx = np.sqrt(np.array([   np.max(labm[m]*hist,axis=1) for m in range(nmat)]))
cmx = np.array([np.argmax(labm[m]*hist,axis=1) for m in range(nmat)])


widths = np.abs(np.concatenate([ [(cmx[1]-cmx[0])/2], (cmx[1:]-cmx[:-1])/2 ]).astype(float))
#bmx    = np.sqrt(6)/(widths+(widths==0))
bmx    = 4/(widths+(widths==0))
dmx    = np.sqrt(np.ones_like(bmx)*1.7)

goodix  = amx>0

# Smooth piecewise polynomial representations
pca  = np.empty((nmat,2*n_segments,1),dtype=float)
pcb  = np.empty((nmat,2*n_segments,1),dtype=float)
pcc  = np.empty((nmat,2*n_segments,1),dtype=float)
pcd  = np.empty((nmat,2*n_segments,1),dtype=float)
bins = np.empty((n_segments,),dtype=float)

for m in tqdm.tqdm(range(nmat),"Initial PWC parameters"):
    pca[m], bins = smooth_fun(xs[amx[m]>0],amx[m][amx[m]>0],n_segments)
    pcb[m], _    = smooth_fun(xs[amx[m]>0],bmx[m][amx[m]>0],n_segments)
    pcc[m], _    = smooth_fun(xs[amx[m]>0],cmx[m][amx[m]>0],n_segments)
    pcd[m], _    = smooth_fun(xs[amx[m]>0],dmx[m][amx[m]>0],n_segments)

Gs = np.array([(pca[m],pcb[m],pcc[m],pcd[m]) for m in range(nmat)])


def evaluate_2d(G, bins, xs, vs, goodix):
    pca,pcb,pcc,pcd = G

    ax = np.zeros(xs.shape,dtype=float)
    ax[goodix] = piecewisecubic((pca,bins), xs[goodix])
    ax = ax[:,na]
    
    bx = piecewisecubic((pcb,bins), xs)[:,na]
    cx = piecewisecubic((pcc,bins), xs)[:,na]
    dx = piecewisecubic((pcd,bins), xs)[:,na]
    
#    return (ax*ax)*np.exp(-(bx*bx)*np.abs(vs[na,:]-cx)**(dx*dx))
    return (ax*ax)*np.exp(-(bx*bx)*np.abs(vs[na,:]-cx)**1.6)

    

def total_2d(Gs, bins, xs, vs, goodix):
    return np.sum([evaluate_2d(Gs[i],bins,xs,vs, goodix[i]) for i in range(nmat)],axis=0)

def energy(Gs_flat, *args):
    hist, bins, xs, vs, beta, goodix, G0 = args
    Gs = Gs_flat.reshape(G0.shape)
#    Gs[:,2] = G0[:,2]
    
    model = total_2d(Gs,bins,xs,vs,goodix)
    diff  = hist - model
    negative = diff[diff<0]

    image_length = np.prod(hist.shape)
    E1 = np.sum(diff**2)/image_length     # Integral of residual squared
    E2 = np.sum(negative**2)/image_length # Integral of overshoot squared

#    line1.set_ydata(model[1000//4])
    pltim.set_data(model)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    print(f"Energy E1 = {E1}, E2 = {E2}\n")
    return E1 #+ beta*E2


    
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
im = total_2d(Gs,bins,xs,vs,goodix)

pltim = ax.imshow(im)

#line1, = ax.plot(vs, im[1000//4], 'r-')
#line2, = ax.plot(vs, hist[1000//4], 'g') 
plt.show()

beta = 10
constants = hist, bins, xs, vs, beta, goodix, Gs

opt_result = opt.minimize(energy, Gs.flatten(), constants, method="SLSQP")

