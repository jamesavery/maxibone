#!/bin/bash
sample   = $0
nblocks  = $1
step     = $2

case $step in 
# Get ESRF data files from ERDA
    1)
	python3 io_modules/cache_esrf2013.py $sample
	;;

# Construct uint16 HDF5 split into MSB and LSB
    2)
	python3 scripts/generate-byte-hdf5.py $sample
	;;

# Compute volume matching shifts
    3)
	python3 scripts/volume_matcher.py $sample
	;;
# Generate uint16 binary blobs
    4)
	python3 scripts/h5tobin.py $sample
	;;
    
# Rescale to 2, 4, 8, 16, 32
    5) 
    python3 preprocess/rescale-cupy-bin.py $sample
    ;;
    
# Segment implant by thresholding at scales 2,...,32
# 1x is too big for memory, needs new implementation of CC  (Needs 292GB to do ndi.label)

    6)
	for scale in 32 16 8 4 2; do python3 segmentation/segment-implant-cc.py $sample $scale; done;
	;;
    

# Get FoR-stuff
    7)
	for scale in 32 16 8; do python3 segmentation/implant-FoR.py $sample $scale; done;
	;;
    
# Compute solid implant 

    8)
	for scale in 4 2; do python3 segmentation/implant-data.py $sample $scale; done;
	;;

# Compute the fields
    9)
	for scale in 32 16 8 4 2; do python3 generate_gauss_c.py 40 50 $sample $scale; done;
	;;

# Compute the 2D histograms

    10)
	for b $(seq 0 $nblocks); do python3 histogram_processing/compute_histograms.py $sample 0 $b 1 -bone_region$b bone_region; done
	;;

    11)
	for b $(seq 0 $nblocks); do python3 histogram_processing/optimize_distributions_flat.py $sample bone_region$b edt 4 0; done
	;;

    12)
	for b $(seq 0 $nblocks); do python3 histogram_processing/compute_probabilities_flat.py $sample bone_region$b edt 10 0; done
	;;    

    13)
	python3 scripts/segment-from-distributions.py $sample 0 0 bone_region optimized_distributions
	;;

    14) 
	for m in 0 1; do python3 preprocess/rescale-cupy-bin.py $sample segmented/P$m ; done
	;;

    15)
	python3 segmentation/segment-blod-cc.py $sample
	;;

# Compute the classification probabilities
# Compute the segmentation probability 3D images
# Compute the blood network
# Compute the bone-implant-contact
# Compute the osteocyte network

esac
