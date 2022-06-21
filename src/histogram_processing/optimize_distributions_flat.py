import os, sys, tqdm, numpy as np, matplotlib.pyplot as plt, numpy.linalg as la, scipy.ndimage as ndi, scipy.optimize as opt, time
sys.path.append(sys.path[0]+"/../")
from piecewise_cubic import piecewisecubic_matrix, piecewisecubic, smooth_fun
from config.paths import commandline_args, hdf5_root as hdf5_root
from distributions import *
from helper_functions import *
na = np.newaxis

hist_path = f"{hdf5_root}/processed/histograms/"
sample, field, region_mask, stride, debug = commandline_args({"sample":"<required>",
                                                              "field":"edt",
                                                              "region_mask":"<required>",
                                                              "stride": 4,
                                                              "debug":8
})

f_hist   = np.load(f"{hist_path}/{sample}/bins-{region_mask}.npz")
f_labels = np.load(f"{hist_path}/{sample}/bins-{region_mask}_labeled.npz")

def row_normalize(A,r):
    return A/(r[:,na]+(r==0)[:,na])

def material_points(labs,material_id):
    mask = labs==material_id    
    xs, ys = np.argwhere(mask).astype(float).T    
    return xs,ys

#TODO: Weight importance in piecewise cubic fitting

field_id = {'edt':0,'gauss':1}; # TODO: Get from data

hist = f_hist["field_bins"][field_id[field]][::stride,::stride]
#sums = np.sum(hist,axis=1)[:,na]
#hist = hist/(sums + (sums==0))
lab  = f_labels[field][::stride,::stride]

nmat    = lab.max()
(nx,nv) = hist.shape
xs = np.arange(0,nx*stride,stride)
vs = np.arange(0,nv*stride,stride)

# Compute start parameter approximation samples
labm = [ndi.grey_erosion(lab==m+1,5) for m in range(nmat)]
amx = np.sqrt(np.array([   np.max(labm[m]*hist,axis=1) for m in range(nmat)]))
cmx = np.array([np.argmax(labm[m]*hist,axis=1) for m in range(nmat)])*stride


widths = np.abs(np.concatenate([ [(cmx[1]-cmx[0])/2], (cmx[1:]-cmx[:-1])/2 ]).astype(float))
#bmx    = np.sqrt(6)/(widths+(widths==0))
bmx    = 6/(widths+(widths==0))
dmx    = np.sqrt(np.ones_like(bmx)*2)

goodix  = amx>0


if (debug&7):
    plt.ion()
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    line1, = ax.plot(vs, np.zeros_like(hist[0]), 'g-')
    line3, = ax.plot(vs, np.zeros_like(hist[0]), 'b:')
    line4, = ax.plot(vs, np.zeros_like(hist[0]), 'r:')
    line2, = ax.plot(vs, hist[0],  'black',linewidth=4)
    
    plt.show()


def opt_bs(bs,*args):
    abcd0, vs = args
    n = len(abcd0)//4
    abcd = abcd0.copy()
    abcd[n:2*n] = bs
    ax.set_title(f"bs = {bs}")    
    return energy1d(abcd,args)

def opt_bds(bds,*args):
    abcd0, vs = args
    n = len(abcd0)//4
    abcd = abcd0.copy()
    abcd[n:2*n] = bds[:n]
    abcd[3*n:4*n] = bds[n:]    
    ax.set_title(f"bs = {bds[:n]}, ds = {bds[n:]}")    
    return energy1d(abcd,args)

    
def opt_all(abcd,*args):
    i, x, abcd0, vs, hist_x = args
    model     = np.sum(powers(vs,abcd), axis=0)
    residual  = hist_x-model

    dx = 1/len(vs)
    E1 = dx*np.sum(residual*residual)
    E2 = dx*np.sum((residual<0)*residual*residual)

    n = len(abcd)//4    
    A,B,C,D = abcd[:n], abcd[n:2*n], abcd[2*n:3*n], abcd[3*n:4*n]    
    Ecloseness = np.sum(1/(np.abs(C[1:]-C[:-1])+0.001))
    
#    print(np.round(E1,2), np.round(1e2*Ecloseness,2))
    if(debug==2):
        line1.set_ydata(model)
        ax.set_title(f"{x}: a = {np.round(A*A,1)}, b = {np.round(B*B,1)}, c = {np.round(C,1)}, d = {1+np.round(D*D,1)}")
        ax.relim()
        ax.autoscale_view()    
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    return E1 + 1e2*Ecloseness #+ 10*E2


good_xs = [[]] * nmat
good_is = [[]] * nmat
ABCDm   = [[]] * nmat
ABCD    = [] 
ABCD_ms = [] 
ABCD_xs = [] 
ABCD_is = [] 

# FLAT OPTIMIZATION
for i,x in enumerate(xs):       
    ms = np.array([m for m in range(nmat) if labm[m][i].max() > 0])
    n  = len(ms)
    if(n>0):
        abcd0 = np.array([amx[ms,i], bmx[ms,i], cmx[ms,i], dmx[ms,i]]).flatten()

        if (debug==1):
            model = powers(vs,abcd0)        
            line1.set_ydata(np.sum(model,axis=0))
            line2.set_ydata(hist[i])
            ax.relim()
            ax.autoscale_view()    
            fig.canvas.draw()
            fig.canvas.flush_events()

        if(debug==2):
            ax.set_title(f"x = {x}")
            line2.set_ydata(hist[i])

        constants  = i, x, abcd0, vs, hist[i]
        midpoints = np.maximum(
            np.concatenate([(cmx[ms,i][1:] + cmx[ms,i][:-1])/2, [vs.max()]]),
            0.9*cmx[ms,i])
            
        bounds = opt.Bounds(np.concatenate([0.3*amx[ms,i],
                                            0.1*bmx[ms,i],
                                            0.9*cmx[ms,i],
                                            0.5*np.ones(ms.shape)]),
                            np.concatenate([1.1*amx[ms,i],
                                            2.0*bmx[ms,i],
                                            midpoints,
                                            1.0*np.ones(ms.shape)]),
                            True)
        opt_result = opt.minimize(opt_all,abcd0,constants,bounds=bounds)

        abcd, success = opt_result['x'], opt_result['success']

        if success:
            print(f"Histogram at x={x} converged successfully for materials {ms}.")
            ABCD      += [abcd]
            ABCD_ms   += [ms]
            ABCD_xs   += [x]
            ABCD_is   += [i]
            
            for im,m in enumerate(ms):            
                good_xs[m] += [x]
                good_is[m] += [i]
                ABCDm[m]    += [abcd[im]]
            
        
        if(debug==4):
            colors = ['b','r']
            lines  = [line3,line4]
            model = powers(vs,abcd)
            n = len(abcd)//4    
            A,B,C,D = abcd[:n], abcd[n:2*n], abcd[2*n:3*n], abcd[3*n:4*n]    
        
            line1.set_ydata(np.sum(model,axis=0))
            line2.set_ydata(hist[i])
            ax.collections.clear()
            ax.fill_between(vs,np.sum(model,axis=0),color='green',alpha=0.5)
            for i,m in enumerate(ms):
                lines[m].set_ydata(model[i])                
                ax.fill_between(vs,model[i],color=colors[m],alpha=0.7)
                
            ax.set_title(f"x={x}: a = {np.round(A*A,1)}, 1/b = {np.round(1/(B*B),3)}, c = {np.round(C,1)}, d = {1.3+np.round(D*D,1)}")
            ax.relim()
            ax.autoscale_view()    
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.savefig(f"opt_debug-{i:03}.png")

hist_modeled = np.zeros_like(hist)
hist_m = np.zeros((2,)+hist.shape,dtype=float)
            
for i,gi in enumerate(ABCD_is):
    abcd = ABCD[i]
    ms   = ABCD_ms[i]
    model = powers(vs,abcd)
    hist_modeled[gi] = np.sum(model,axis=0)
    hist_m[ms,gi] = model

if (debug&8):
    fig = plt.figure(figsize=(10,10))
    axarr = fig.subplots(2,2)
    fig.suptitle(f'{sample} {region_mask}') # or plt.suptitle('Main title')
    axarr[0,0].imshow(row_normalize(hist,hist.max(axis=1)))
    axarr[0,0].set_title(f"{field}-field 2D Histogram")
    axarr[0,1].imshow(row_normalize(hist_modeled,hist.max(axis=1)))
    axarr[0,1].set_title("Remodeled 2D Histogram")
    axarr[1,0].imshow(row_normalize(hist_m[0],hist.max(axis=1)))
    axarr[1,0].set_title("Material 1")
    axarr[1,1].imshow(row_normalize(hist_m[1],hist.max(axis=1)))
    axarr[1,1].set_title("Material 2")
    fig.tight_layout()
    fig.savefig(f"{hdf5_root}/processed/histograms/{sample}/hist_vs_modeled_{field}_{region_mask}.png")
    plt.show()

update_hdf5(f"{hdf5_root}/processed/histograms/{sample}.h5",
            group_name=f"{region_mask}/{field}",
            datasets={"xs":xs, "vs":vs,
                      "good_xs0":np.array(good_xs[0]),
                      "good_xs1":np.array(good_xs[1]),                      
                      "ABCD0":   np.array(ABCD[0]),
                      "ABCD1":   np.array(ABCD[1])
            })        

