# This script is intended to start an instance of the MEOW WorkflowRunner, which will
# schedule the defined analysis below. As an event driven system it depends on flag files
# within the VGRID directory defined below. To start processing you should run this script
# which will start a daemon process that will continue until the shell command is stopped, 
# usually via ctrl C. No changes should be required here between runs despite changes in 
# the scripts in scripts, histogram_processing, preprocess etc.
#
# If you need to alter patterns or recipes whilst keeping the 
# WorkflowRunner running, you can use the accompanying update_live_runner.py script to do 
# so.

import os
import mig_meow as meow

from meow_variables import VGRID, MEOW_RUNNER_DATA, JOB_RUNNER_DATA, OUTPUT_RUNNER_DATA, \
    GEN_BYTE, VOL_INT, GEN_SCA, IMP_ANL, IMP_DIFF, IMP_EDT, CMP_HST, CMP_RDG, CMP_PRB, \
    CMP_SEG, CMP_BA, RPT_CNT, FLAG_DIRS
from config.paths import esrf_implants_root


## CONFIGURATION -----------------------------------------------------------------------
# This is the main section that you should need to interact with, as the rest of this 
# script is designed to be self contained as run from these options. You may also need to
# look at the later pattern definitions section if you wish to update script variable 
# values

# How many workers there are. Each takes a single job at a time
WORKERS = 1
# If this is true then when a new rule is created from a pattern and recipe, it will look
# for current files that would trigger it, were they created now 
RETRO_ACTIVE = True
# If true then output will be printed to stdout 
PRINT_LOGGING = True
# If true then output will be logged in a file in the 'mig_meow_logs' dir
FILE_LOGGING = False
# How long a worker will wait in seconds after finding an empty queue before it will check 
# again
WAIT_TIME = 1

## PATTERN DEFINTIONS -----------------------------------------------------------------------
# Each patttern here corresponds to one of the workflow scripts. The scripts themselves are
# called by notebooks in the 'notebooks' dir, which become the recipes attached to these patterns
# Variables should be updated here as these variables will override any defintions in the recipe
# notebooks. If you add/remove any variables from the scripts then both the notebooks and the
# defintions here should be updated.

# Generate byte data
p_00_generate_byte_data = meow.Pattern(GEN_BYTE)
p_00_generate_byte_data.add_single_input("trigger_file", "00_generate_byte_data/*")
p_00_generate_byte_data.add_output("output_flag", "meow_triggers/01_volume_matcher/{FILENAME}")
p_00_generate_byte_data.add_variable("chunk_length", 256)
p_00_generate_byte_data.add_variable("use_bohrium", True)
p_00_generate_byte_data.add_variable("xml_root", esrf_implants_root)
p_00_generate_byte_data.add_recipe(GEN_BYTE)

# Volume and intensity match
p_01_volumne_matcher = meow.Pattern(VOL_INT)
p_01_volumne_matcher.add_single_input("trigger_file", "01_volume_matcher/*")
p_01_volumne_matcher.add_output("output_flag", "meow_triggers/02_generate_scales/{FILENAME}")
p_01_volumne_matcher.add_variable("overlap", 10)
p_01_volumne_matcher.add_variable("max_shift", 150)
p_01_volumne_matcher.add_variable("generate_h5", False)
p_01_volumne_matcher.add_recipe(VOL_INT)

# Generate scales
p_02_generate_scales = meow.Pattern(GEN_SCA)
p_02_generate_scales.add_single_input("trigger_file", "02_generate_scales/*")
p_02_generate_scales.add_output("output_flag", "meow_triggers/03_implant_analysis/{FILENAME}")
p_02_generate_scales.add_variable("dataset", "voxels")
p_02_generate_scales.add_variable("output_relative_to_input", "..")
p_02_generate_scales.add_variable("chunk_size", 6*20)
p_02_generate_scales.add_variable("compression", "lzf")
p_02_generate_scales.add_recipe(GEN_SCA)

# Implant analysis
p_03_implant_analysis = meow.Pattern(IMP_ANL)
p_03_implant_analysis.add_single_input("trigger_file", "03_implant_analysis/*")
p_03_implant_analysis.add_output("output_flag", "meow_triggers/04_generate_implant_diffusion/{FILENAME}")
p_03_implant_analysis.add_recipe(IMP_ANL)

# Implant diffusion
p_04_implant_diffusion = meow.Pattern(IMP_DIFF)
p_04_implant_diffusion.add_single_input("trigger_file", "04_generate_implant_diffusion/*")
p_04_implant_diffusion.add_output("output_flag", "meow_triggers/05_generate_implant_edt/{FILENAME}")
p_04_implant_diffusion.add_variable("scale", 1)
p_04_implant_diffusion.add_variable("chunk_size", 400)
p_04_implant_diffusion.add_variable("padding", 100)
p_04_implant_diffusion.add_recipe(IMP_DIFF)

# Implant EDT
p_05_implant_edt = meow.Pattern(IMP_EDT)
p_05_implant_edt.add_single_input("trigger_file", "05_generate_implant_edt/*")
p_05_implant_edt.add_output("output_flag", "meow_triggers/06_compute_histograms/{FILENAME}")
p_05_implant_edt.add_variable("scale", 1)
p_05_implant_edt.add_variable("chunk_size", 200)
p_05_implant_edt.add_variable("padding", 28)
p_05_implant_edt.add_recipe(IMP_EDT)

# Compute histograms
p_06_compute_histogram = meow.Pattern(CMP_HST)
p_06_compute_histogram.add_single_input("trigger_file", "06_compute_histograms/*")
p_06_compute_histogram.add_output("output_flag", "meow_triggers/07_compute_ridges/{FILENAME}")
p_06_compute_histogram.add_recipe(CMP_HST)

# Compute ridges
p_07_compute_ridges = meow.Pattern(CMP_RDG)
p_07_compute_ridges.add_single_input("trigger_file", "07_compute_ridges/*")
p_07_compute_ridges.add_output("output_flag", "meow_triggers/08_compute_probabilites/{FILENAME}")
p_07_compute_ridges.add_variable("batch_k", "--batch")
p_07_compute_ridges.add_variable("batch_v", True)
p_07_compute_ridges.add_variable("config_k", "--config")
p_07_compute_ridges.add_variable("config_v", "config.json")
p_07_compute_ridges.add_variable("dry_run_k", "--dry_run")
p_07_compute_ridges.add_variable("dry_run_v", False)
p_07_compute_ridges.add_variable("output_k", "--output")
p_07_compute_ridges.add_variable("output_v", "output")
p_07_compute_ridges.add_variable("peaks_k", "--peaks")
p_07_compute_ridges.add_variable("peaks_v", False)
p_07_compute_ridges.add_variable("verbose_k", "--verbose")
p_07_compute_ridges.add_variable("verbose_v", False)
p_07_compute_ridges.add_recipe(CMP_RDG)

# Compute probabilities
p_08_compute_probabilities = meow.Pattern(CMP_PRB)
p_08_compute_probabilities.add_single_input("trigger_file", "08_compute_probabilites/*")
p_08_compute_probabilities.add_output("output_flag", "meow_triggers/09_compute_segmentation/{FILENAME}")
p_08_compute_probabilities.add_variable("material_id", 1)
p_08_compute_probabilities.add_variable("axis", "x")
p_08_compute_probabilities.add_variable("n_segments", 8)
p_08_compute_probabilities.add_recipe(CMP_PRB)

# Compute segmentation 
p_09_compute_segmentation = meow.Pattern(CMP_SEG)
p_09_compute_segmentation.add_single_input("trigger_file", "09_compute_segmentation/*")
p_09_compute_segmentation.add_output("output_flag", "meow_triggers/10_compute_bone_area/{FILENAME}")
p_09_compute_segmentation.add_recipe(CMP_SEG)

# Compute bone area
p_10_bone_area = meow.Pattern(CMP_BA)
p_10_bone_area.add_single_input("trigger_file", "10_compute_bone_area/*")
p_10_bone_area.add_output("output_flag", "meow_triggers/11_repeat_with_constraints/{FILENAME}")
p_10_bone_area.add_recipe(CMP_BA)

# Repeat Histogram with constraints
p_11_repeat_histogram_with_constraints = meow.Pattern(RPT_CNT)
p_11_repeat_histogram_with_constraints.add_single_input("trigger_file", "11_repeat_with_constraints/*")
p_11_repeat_histogram_with_constraints.add_output("output_flag", "meow_triggers/12_all_done/{FILENAME}")
p_11_repeat_histogram_with_constraints.add_recipe(RPT_CNT)


## WORKFLOW SETUP -----------------------------------------------------------------------
# Here all patterns and recipes are put into dictionaries to be passed to the runner
patterns = {
    # Generate byte data
    GEN_BYTE: p_00_generate_byte_data,

####    # Volume and intensity match
####    VOL_INT: p_01_volumne_matcher,
####
####    # Generate scales
####    GEN_SCA: p_02_generate_scales,
####
####    # Implant analysis
####    IMP_ANL: p_03_implant_analysis,
####
####    # Implant diffusion
####    IMP_DIFF: p_04_implant_diffusion,
####
####    # Implant EDT
####    IMP_EDT: p_05_implant_edt,
####
####    # Compute histograms
####    CMP_HST: p_06_compute_histogram,
####
####    # Compute ridges
####    CMP_RDG: p_07_compute_ridges,
####
####    # Compute probabilities
####    CMP_PRB: p_08_compute_probabilities,
####
####    # Compute segmentation 
####    CMP_SEG: p_09_compute_segmentation,
####
####    # Compute bone area
####    CMP_BA: p_10_bone_area,
####
####    # Repeat Histogram with constraints
####    RPT_CNT: p_11_repeat_histogram_with_constraints
}

recipes = {
    # Generate byte data
    GEN_BYTE: meow.register_recipe("notebooks/00_generate_byte_data.ipynb", GEN_BYTE),

    # Volume and intensity match
    VOL_INT: meow.register_recipe("notebooks/01_volume_matcher.ipynb", VOL_INT),

    # Generate scales
    GEN_SCA: meow.register_recipe("notebooks/02_generate_scales.ipynb", GEN_SCA),

    # Implant analysis
    IMP_ANL: meow.register_recipe("notebooks/03_implant_analysis.ipynb", IMP_ANL),

    # Implant diffusion
    IMP_DIFF: meow.register_recipe("notebooks/04_generate_implant_diffusion.ipynb", IMP_DIFF),

    # Implant EDT
    IMP_EDT: meow.register_recipe("notebooks/05_generate_implant_edt.ipynb", IMP_EDT),

    # Compute histograms
    CMP_HST: meow.register_recipe("notebooks/06_compute_histograms.ipynb", CMP_HST),

    # Compute ridges
    CMP_RDG: meow.register_recipe("notebooks/07_compute_ridges.ipynb", CMP_RDG),

    # Compute probabilities
    CMP_PRB: meow.register_recipe("notebooks/08_compute_probabilities.ipynb", CMP_PRB),

    # Compute segmentation 
    CMP_SEG: meow.register_recipe("notebooks/09_compute_segmentation.ipynb", CMP_SEG),

    # Compute bone area
    CMP_BA: meow.register_recipe("notebooks/10_compute_bone_area.ipynb", CMP_BA),

    # Repeat Histogram with constraints
    RPT_CNT: meow.register_recipe("notebooks/11_repeat_histogram_with_constraints.ipynb", RPT_CNT)
}


## FLAG DIRECTORY CREATION --------------------------------------------------------------
# All processing in this runner will take some flag files as their trigger for a new job
# Here we just create all the required directories to store those flags. These will all
# be within the directory defined by VGRID
for directory in FLAG_DIRS:
    flag_dir = os.path.join(VGRID, directory)
    if not os.path.exists(flag_dir):
        os.mkdir(flag_dir)


## WORKFLOW RUNNER -----------------------------------------------------------------------
# Start the actual runner itself. Configuring this should be done by the variables set up 
# in the configuration section
runner = meow.WorkflowRunner(
    VGRID,
    WORKERS,
    meow_data=MEOW_RUNNER_DATA,
    job_data=JOB_RUNNER_DATA,
    output_data=OUTPUT_RUNNER_DATA,
    patterns=patterns,
    recipes=recipes,
    daemon=True,
    start_workers=True,
    retro_active_jobs=RETRO_ACTIVE,
    print_logging=PRINT_LOGGING,
    file_logging=FILE_LOGGING,
    wait_time=WAIT_TIME
)
