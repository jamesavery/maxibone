# This file defines a variety of variables defined in both the run_workflows and 
# update_live_runner scripts. Most of this should never need to be changed with 
# those that will at the top


## Run Configuration options ----------------------------------------------------
# These may be altered between runner instances
# This is the base for all meow runner triggers, doesn't need to contain the actual data
VGRID = "meow_triggers"
# Where the pattern and recipe defintiions are stored. If updated whilst the runner is 
# running, then changes will be caught and the runner will update its internal state
MEOW_RUNNER_DATA = "runner_data"
# Where jobs are processed. If something goes wrong and nothing appears in output, this
# is where to investigate but otherwise can be ignored.
JOB_RUNNER_DATA = "runner_processing"
# All jobs should produce some output here. This may potentially get quite large if you
# keep running analysis again and again
OUTPUT_RUNNER_DATA = "runner_output"


## MEOW Experiment options ------------------------------------------------------
# These are variables integral to functionality in the experiment and should not
# ever need to be altered unless the structure of the experiment fundamentally 
# shifts
GEN_BYTE = "generate_byte_data"
VOL_INT = "volume_and_intensity_match"
GEN_SCA = "generate_scales"
IMP_ANL = "implant_analysis"
IMP_DIFF = "implant_diffusion"
IMP_EDT = "implant_edt"
CMP_HST = "compute_histogram"
CMP_RDG = "compute_ridges"
CMP_PRB = "compute_probabilities"
CMP_SEG = "compute_segmentation"
CMP_BA = "compute_bone_area"
RPT_CNT = "repeat_computation_with_constraints"

FLAG_DIRS = [
    "00_generate_byte_data",
    "01_volume_matcher",
    "02_generate_scales",
    "03_implant_analysis",
    "04_generate_implant_diffusion",
    "05_generate_implant_edt",
    "06_compute_histograms",
    "07_compute_ridges",
    "08_compute_probabilites",
    "09_compute_segmentation",
    "10_compute_bone_area",
    "11_repeat_with_constraints",
    "12_all_done"
]
