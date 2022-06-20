#!/usr/bin/env python3
import os,sys,h5py,numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm
sys.path.append(sys.path[0]+"/../../")
from config.paths import *
from helper_functions import h5meta_info_volume_matched

def imshow(image,filename=None,xlabel=None,xticks=None,ylabel=None,yticks=None,plotlabel=None,cmap='RdYlBu',figsize=(20,10)):

    fig = plt.figure(figsize=figsize)
        
    ax = fig.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks[0])        
        ax.set_xticklabels(xticks[1])
    if yticks is not None:
        ax.set_yticks(yticks[0])
        ax.set_yticklabels(yticks[1])
    if plotlabel is not None:
        ax.set_title(plotlabel)
        
    ax.imshow(image,cmap=cmap)

    if filename is not None:
        print(f"Writing image to {filename}")        
        fig.savefig(filename)
    else:
        plt.show()


def show_section(sample,filename=None,axesnames=('y','z'), bbox=(0, None, None), scale=1, nticks=6, figsize=None):
    axesid={'z':0,'y':1,'x':2}
    axes = (axesid[axesnames[0]],axesid[axesnames[1]])
    
    (Nz,Ny,Nx), subvolume_nzs, voxel_size_1x = h5meta_info_volume_matched(sample)
    (nz,ny,nx) = (Nz//scale,Ny//scale,Nx//scale)
    voxel_size = voxel_size_1x * scale        
    voxels = np.memmap(f"{binary_root}/voxels/{scale}x/{sample}.uint16",dtype=np.uint16,mode='r').reshape(nz,ny,nx)

    zs = np.round(np.linspace(0,nz*voxel_size,nz),2)
    ys = np.round(np.linspace(0,ny*voxel_size,ny),2) 
    xs = np.round(np.linspace(0,nx*voxel_size,nx),2)

    axeslabels = "z/μm", "y/μm", "x/μm"    
    axesspaces = [zs,ys,xs]

    cut = int(round(bbox[0]/voxel_size))
    if bbox[1] is not None:
        start, end = bbox[1]
        ab_range = slice(int(round(start/voxel_size)), end if end < 0 else int(round(end/voxel_size)))
    else:
        ab_range = slice(None)
        
    if bbox[2] is not None:
        start, end = bbox[2]
        or_range = slice(int(round(start/voxel_size)), end if end < 0 else int(round(end/voxel_size)))
    else:
        or_range = slice(None)

    
    if axes==(0,1):             # TODO: Dynamic axes
        image = voxels[ab_range,or_range,cut].T
    if axes==(1,0):      
        image = voxels[or_range,ab_range,cut]
    if axes==(0,2):        
        image = voxels[ab_range,cut,or_range].T
    if axes==(2,0):        
        image = voxels[or_range,cut,ab_range]            
    if axes==(1,2):
        image = voxels[cut,ab_range,or_range].T
    if axes==(2,1):
        image = voxels[cut,or_range,ab_range]        

    del voxels

    abscissa, ordinate = axesspaces[axes[0]][ab_range], axesspaces[axes[1]][or_range]
    
    
    xtix   = np.linspace(0,len(abscissa)-1,nticks,dtype=int)
    ytix   = np.linspace(0,len(ordinate)-1,nticks,dtype=int) 
    xticks = (xtix, abscissa[xtix])
    yticks = (ytix, ordinate[ytix])    

    imshow(image,filename=filename,xlabel=axeslabels[axes[0]],ylabel=axeslabels[axes[1]],xticks=xticks,yticks=yticks,figsize=figsize)




if __name__ == "__main__":
    sample, scale, abscissa, ordinate, cut, ab_from, ab_to, or_from, or_to = \
        commandline_args({"sample":"<required>", "scale":4,
                          "abscissa":"x", "ordinate":"y",
                          "cut":3000,
                          "ab_from":0, "ab_to":-1,
                          "or_from":0, "or_to":-1
        })
                                                                               
    
    (Nz,Ny,Nx), subvolume_nzs, voxel_size_1x = h5meta_info_volume_matched(sample)
    (nz,ny,nx) = (Nz//scale,Ny//scale,Nx//scale)
    voxel_size = voxel_size_1x * scale

    print(f"Extracting region from {(nz*voxel_size,ny*voxel_size,nx*voxel_size)} μm^3 image")

    show_section("770c_pag",scale=scale,axesnames=(abscissa,ordinate),bbox=(cut,(ab_from,ab_to),(or_from,or_to)),figsize=None)


    
