sample="770c_pag"

python3.12 processing_steps/0300_volume_matcher.py $sample
python3.12 processing_steps/0400_h5tobin.py $sample
python3.12 processing_steps/0500_rescale_cupy_bin.py $sample
python3.12 processing_steps/0600_segment_implant_cc.py $sample 1
python3.12 processing_steps/0600_segment_implant_cc.py $sample 2
python3.12 processing_steps/0600_segment_implant_cc.py $sample 8
python3.12 processing_steps/0700_implant_FoR.py $sample 8
python3.12 processing_steps/0750_bone_region.py $sample 2
python3.12 processing_steps/0800_implant_data.py $sample 2
python3.12 processing_steps/0900_generate_gauss_c.py $sample 2
python3.12 processing_steps/0950_generate_edt.py $sample 2
python3.12 processing_steps/0975_combine_fields.py $sample 2
python3.12 processing_steps/1000_compute_histograms.py $sample 1 2 256 0 0 "-bone_region" "bone_region" 2 4096 4096
python3.12 processing_steps/1250_generate_otsu_probabilities.py $sample "bone_region"
python3.12 processing_steps/1300_segment_from_distributions.py $sample 1 0 128 "bone_region" "otsu_separation" 2 "gauss+edt" 2
python3.12 processing_steps/1500_segment_blood_cc.py $sample 0 gauss+edt
python3.12 processing_steps/1600_bic.py $sample
python3.12 processing_steps/1700_bic_stats.py $sample
python3.12 processing_steps/1800_healthy_bone.py $sample

# Remove temporary files
rm -rf /tmp/maxibone/*