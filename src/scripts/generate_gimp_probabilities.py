import os,sys, pathlib, h5py, numpy as np
sys.path.append(sys.path[0]+"/../")
import pybind_kernels.histograms as histograms
from config.paths import binary_root, hdf5_root_fast as hdf5_root, commandline_args
from tqdm import tqdm
import matplotlib.pyplot as plt, matplotlib.cm as cm
from histogram_processing.piecewise_quadratic import piecewisequadratic, piecewisequadratic_matrix, smooth_fun
from PIL import Image
from numpy import newaxis as NA
import helper_functions


# TODO: Til fælles fil.
def save_probabilities(Ps, sample, region,value_ranges,prob_method):
    output_path = f'{hdf5_root}/processed/probabilities/{sample}.h5'
    helper_functions.update_hdf5(
        output_path,
        group_name = f'{prob_method}/{region}',
        datasets = { 'value_ranges' : value_ranges },
        attributes = {}
    )
    for name, P0, P1, pc, valid_range, threshes, new_threshes in Ps:
        helper_functions.update_hdf5(
            output_path,
            group_name = f'{prob_method}/{region}/{name}',
            datasets = {
                f'c0': P0,
                f'c1': P1,
                f'pc_coefs':   pc[0],
                f'pc_borders': pc[1],
                f'threshes': threshes,
                f'new_threshes': new_threshes
            },
            attributes = {
                f'valid_start': valid_range[0],
                f'valid_end': valid_range[1],
            }
        )



if __name__ == "__main__":
   sample, region, debug = commandline_args({"sample":"<required>",
                                             "region":"bone_region", "debug":0})

   field_names = ["edt","gauss"]

   try:
      f = h5py.File(f"{hdf5_root}/hdf5-byte/msb/{sample}.h5","r");
   except Exception as e:
      print(f"Couldn't open{hdf5_root}/hdf5-byte/msb/{sample}.h5")
      print(f"Reason: {e}")
      sys.exit(-1)

   
   n_subvolumes = len(f["subvolume_dimensions"][:])
   
   image_dir = f"{hdf5_root}/processed/probabilities/images/{sample}"
   bins_dir  = f"{hdf5_root}/processed/histograms/{sample}/"


   for b in tqdm(range(n_subvolumes),desc=f"{region} block in 0,...,{n_subvolumes}",leave=False):
       Ps = []       
       for field_name in tqdm(field_names,desc=f"Field in {field_names}",leave=False):
         bin_path   = f"{bins_dir}/bins-{region}{b}.npz"
         image_path = f"{image_dir}/fb-{field_name}-{region}{b}-gimp.png"

         try:
            image = np.asarray(Image.open(image_path)).copy()
         except Exception as e:
            print(f"Couldn't open {image_path}")
            print(f"Reason: {e}")
            sys.exit(-1)
            
         try:
            bin_file = np.load(bin_path)
            i = list(bin_file["field_names"]).index(field_name)
            hist2d = bin_file["field_bins"][i]
         except Exception as e:
            print(f"Couldn't open {bin_path}")
            print(f"Reason: {e}")
            sys.exit(-1)
            
         n_rows, n_cols, _ = image.shape
            
         thresholds = np.argwhere(image[...,0] > image[...,1])
         xs, ys = thresholds.T
            
         pc = smooth_fun(xs[::4],ys[::4],50,1e3)
         all_xs = np.arange(n_rows+1)
         smooth_thresholds = piecewisequadratic(pc,all_xs,extrapolation="linear")

         if debug:
            for i in range(n_rows):
               image[i,(int(smooth_thresholds[i])-1):(int(smooth_thresholds[i])+1),1] = 0xff;

            Image.fromarray(image).save(f'{image_dir}/fb-{field_name}-{region}{b}-gimp-t.png')


         ts = np.round(smooth_thresholds).astype(int)

         P0 = np.zeros(hist2d.shape, dtype=np.float32)
         P1 = np.zeros(hist2d.shape, dtype=np.float32)
         
         for i, row in enumerate(hist2d):
            mx = max(1,np.float32(row.max()))
            P0[i,:ts[i]] = row[:ts[i]].astype(np.float32) / mx
            P1[i,ts[i]:] = row[ts[i]:].astype(np.float32) / mx

         Ps.append( (field_name,P0,P1,pc,(0,n_rows-1),thresholds,smooth_thresholds) )

       save_probabilities(Ps, sample, region+str(b),bin_file['value_ranges'],"gimp_separation")


            
            
