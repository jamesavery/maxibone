import numpy as np
import os
from   matplotlib import cm
import contextlib
import imageio

def normalize(data, vmin=None, vmax=None):
    vmin, vmax = data.min(), data.max()
    return (data - vmin) / (vmax - vmin) 

# Assumes data is already in range [0,1]
def to_byte(data,mask):
    return (data*255*(mask==False)).astype(np.uint8);


def write_video(filename,Ftxx,mask=None,fps=24,colormap=cm.viridis):
    (nt,ny,nx) = Ftxx.shape;

    if(mask is None):
        mask = np.zeros(Ftxx.shape[1:],dtype=np.bool);

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)

    vmin, vmax = Ftxx.reshape(-1).min(), Ftxx.reshape(-1).max();
    writer = imageio.get_writer(
        filename, fps=fps, quality=10, macro_block_size=None)

    for i in range(nt):
        img_rgbaf = colormap(normalize(Ftxx[i],vmin,vmax));
        frame=to_byte(img_rgbaf[:,:,:3],mask[:,:,None]);
        writer.append_data(frame)
    writer.close()


def write_double_video(filename,Ftxx1,Ftxx2,mask=None,fps=24,colormap=cm.viridis):
    assert(Ftxx1.shape == Ftxx2.shape);
    (nt,ny,nx) = Ftxx1.shape;

    Ftxx = np.empty((nt,ny,2*nx),dtype=Ftxx1.dtype);
    Ftxx[:,:,:nx] = Ftxx1;
    Ftxx[:,:,nx:] = Ftxx2;

    write_video(filename,Ftxx,mask,fps,colormap)


