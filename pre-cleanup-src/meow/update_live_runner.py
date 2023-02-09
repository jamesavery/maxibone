# This script can be used to update the pattern or recipe definitions in a live WorkflowRunner. 
# They can also be manually editted by a user, but this is not advised unless you really are 
# confident in what you're doing.

import mig_meow as meow

from meow_variables import GEN_BYTE, VOL_INT, GEN_SCA, IMP_ANL, IMP_DIFF, IMP_EDT, CMP_HST, \
    CMP_RDG, CMP_PRB, CMP_SEG, CMP_BA, RPT_CNT, MEOW_RUNNER_DATA
from config.paths import esrf_implants_root

## Configuration ----------------------------------------------------------------------------
# The following two lists define which patterns and recipes in the running WorkflowRunner will
# be updated. Any that should not be updated should be commented out. Updating extras ones that
# have not been changed won't break anything, but will mean additional processing will be 
# scheduled needlessly. Note that for patterns you will also need to update the pattern defintions
# below.

recipes_to_update = [
    GEN_BYTE,
    VOL_INT,
    GEN_SCA,
    IMP_ANL,
    IMP_DIFF,
    IMP_EDT,
    CMP_HST,
    CMP_RDG,
    CMP_PRB,
    CMP_SEG,
    CMP_BA,
    RPT_CNT
]

patterns_to_update = [
    GEN_BYTE,
    VOL_INT,
    GEN_SCA,
    IMP_ANL,
    IMP_DIFF,
    IMP_EDT,
    CMP_HST,
    CMP_RDG,
    CMP_PRB,
    CMP_SEG,
    CMP_BA,
    RPT_CNT
]

## Pattern Defintions ------------------------------------------------------------------------
# Generate byte data
if GEN_BYTE in patterns_to_update:
    p_00_generate_byte_data = meow.Pattern(GEN_BYTE)
    p_00_generate_byte_data.add_single_input("trigger_file", "00_generate_byte_data/*")
    p_00_generate_byte_data.add_output("output_flag", "meow_triggers/01_volume_matcher/{FILENAME}")
    p_00_generate_byte_data.add_variable("chunk_length", 256)
    p_00_generate_byte_data.add_variable("use_bohrium", True)
    p_00_generate_byte_data.add_variable("xml_root", esrf_implants_root)
    p_00_generate_byte_data.add_recipe(GEN_BYTE)
    meow.write_dir_pattern(p_00_generate_byte_data, directory=MEOW_RUNNER_DATA)

# Volume and intensity match
if VOL_INT in patterns_to_update:
    p_01_volumne_matcher = meow.Pattern(VOL_INT)
    p_01_volumne_matcher.add_single_input("trigger_file", "01_volume_matcher/*")
    p_01_volumne_matcher.add_output("output_flag", "meow_triggers/02_generate_scales/{FILENAME}")
    p_01_volumne_matcher.add_variable("overlap", 10)
    p_01_volumne_matcher.add_variable("max_shift", 150)
    p_01_volumne_matcher.add_variable("generate_h5", False)
    p_01_volumne_matcher.add_recipe(VOL_INT)
    meow.write_dir_pattern(p_01_volumne_matcher, directory=MEOW_RUNNER_DATA)

# Generate scales
if VOL_INT in patterns_to_update:
    p_02_generate_scales = meow.Pattern(GEN_SCA)
    p_02_generate_scales.add_single_input("trigger_file", "02_generate_scales/*")
    p_02_generate_scales.add_output("output_flag", "meow_triggers/03_implant_analysis/{FILENAME}")
    p_02_generate_scales.add_variable("dataset", "voxels")
    p_02_generate_scales.add_variable("output_relative_to_input", "..")
    p_02_generate_scales.add_variable("chunk_size", 6*20)
    p_02_generate_scales.add_variable("compression", "lzf")
    p_02_generate_scales.add_recipe(GEN_SCA)
    meow.write_dir_pattern(p_02_generate_scales, directory=MEOW_RUNNER_DATA)

# Implant analysis
if IMP_ANL in patterns_to_update:
    p_03_implant_analysis = meow.Pattern(IMP_ANL)
    p_03_implant_analysis.add_single_input("trigger_file", "03_implant_analysis/*")
    p_03_implant_analysis.add_output("output_flag", "meow_triggers/04_generate_implant_diffusion/{FILENAME}")
    p_03_implant_analysis.add_recipe(IMP_ANL)
    meow.write_dir_pattern(p_03_implant_analysis, directory=MEOW_RUNNER_DATA)

# Implant diffusion
if IMP_DIFF in patterns_to_update:
    p_04_implant_diffusion = meow.Pattern(IMP_DIFF)
    p_04_implant_diffusion.add_single_input("trigger_file", "04_generate_implant_diffusion/*")
    p_04_implant_diffusion.add_output("output_flag", "meow_triggers/05_generate_implant_edt/{FILENAME}")
    p_04_implant_diffusion.add_variable("scale", 1)
    p_04_implant_diffusion.add_variable("chunk_size", 400)
    p_04_implant_diffusion.add_variable("padding", 100)
    p_04_implant_diffusion.add_recipe(IMP_DIFF)
    meow.write_dir_pattern(p_04_implant_diffusion, directory=MEOW_RUNNER_DATA)

# Implant EDT
if IMP_EDT in patterns_to_update:
    p_05_implant_edt = meow.Pattern(IMP_EDT)
    p_05_implant_edt.add_single_input("trigger_file", "05_generate_implant_edt/*")
    p_05_implant_edt.add_output("output_flag", "meow_triggers/06_compute_histograms/{FILENAME}")
    p_05_implant_edt.add_variable("scale", 1)
    p_05_implant_edt.add_variable("chunk_size", 200)
    p_05_implant_edt.add_variable("padding", 28)
    p_05_implant_edt.add_recipe(IMP_EDT)
    meow.write_dir_pattern(p_05_implant_edt, directory=MEOW_RUNNER_DATA)

# Compute histograms
if CMP_HST in patterns_to_update:
    p_06_compute_histogram = meow.Pattern(CMP_HST)
    p_06_compute_histogram.add_single_input("trigger_file", "06_compute_histograms/*")
    p_06_compute_histogram.add_output("output_flag", "meow_triggers/07_compute_ridges/{FILENAME}")
    p_06_compute_histogram.add_recipe(CMP_HST)
    meow.write_dir_pattern(p_06_compute_histogram, directory=MEOW_RUNNER_DATA)

# Compute ridges
if CMP_RDG in patterns_to_update:
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
    meow.write_dir_pattern(p_07_compute_ridges, directory=MEOW_RUNNER_DATA)

# Compute probabilities
if CMP_PRB in patterns_to_update:
    p_08_compute_probabilities = meow.Pattern(CMP_PRB)
    p_08_compute_probabilities.add_single_input("trigger_file", "08_compute_probabilites/*")
    p_08_compute_probabilities.add_output("output_flag", "meow_triggers/09_compute_segmentation/{FILENAME}")
    p_08_compute_probabilities.add_variable("material_id", 1)
    p_08_compute_probabilities.add_variable("axis", "x")
    p_08_compute_probabilities.add_variable("n_segments", 8)
    p_08_compute_probabilities.add_recipe(CMP_PRB)
    meow.write_dir_pattern(p_08_compute_probabilities, directory=MEOW_RUNNER_DATA)

# Compute segmentation 
if CMP_SEG in patterns_to_update:
    p_09_compute_segmentation = meow.Pattern(CMP_SEG)
    p_09_compute_segmentation.add_single_input("trigger_file", "09_compute_segmentation/*")
    p_09_compute_segmentation.add_output("output_flag", "meow_triggers/10_compute_bone_area/{FILENAME}")
    p_09_compute_segmentation.add_recipe(CMP_SEG)
    meow.write_dir_pattern(p_09_compute_segmentation, directory=MEOW_RUNNER_DATA)

# Compute bone area
if CMP_BA in patterns_to_update:
    p_10_bone_area = meow.Pattern(CMP_BA)
    p_10_bone_area.add_single_input("trigger_file", "10_compute_bone_area/*")
    p_10_bone_area.add_output("output_flag", "meow_triggers/11_repeat_with_constraints/{FILENAME}")
    p_10_bone_area.add_recipe(CMP_BA)
    meow.write_dir_pattern(p_10_bone_area, directory=MEOW_RUNNER_DATA)

# Repeat Histogram with constraints
if RPT_CNT in patterns_to_update:
    p_11_repeat_histogram_with_constraints = meow.Pattern(RPT_CNT)
    p_11_repeat_histogram_with_constraints.add_single_input("trigger_file", "11_repeat_with_constraints/*")
    p_11_repeat_histogram_with_constraints.add_output("output_flag", "meow_triggers/12_all_done/{FILENAME}")
    p_11_repeat_histogram_with_constraints.add_recipe(RPT_CNT)
    meow.write_dir_pattern(p_11_repeat_histogram_with_constraints, directory=MEOW_RUNNER_DATA)


## Recipe Defintions -------------------------------------------------------------------------
# Generate byte data
if GEN_BYTE in recipes_to_update:
    recipe = meow.register_recipe("notebooks/00_generate_byte_data.ipynb", GEN_BYTE)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Volume and intensity match
if VOL_INT in recipes_to_update:
    recipe = meow.register_recipe("notebooks/01_volume_matcher.ipynb", VOL_INT)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Generate scales
if GEN_SCA in recipes_to_update:
    recipe = meow.register_recipe("notebooks/02_generate_scales.ipynb", GEN_SCA)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Implant analysis
if IMP_ANL in recipes_to_update:
    recipe = meow.register_recipe("notebooks/03_implant_analysis.ipynb", IMP_ANL)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Implant diffusion
if IMP_DIFF in recipes_to_update:
    recipe = meow.register_recipe("notebooks/04_generate_implant_diffusion.ipynb", IMP_DIFF)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Implant EDT
IMP_EDT: meow.register_recipe("notebooks/05_generate_implant_edt.ipynb", IMP_EDT)
if IMP_EDT in recipes_to_update:
    recipe = meow.register_recipe("notebooks/05_generate_implant_edt.ipynb", IMP_EDT)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Compute histograms
if CMP_HST in recipes_to_update:
    recipe = meow.register_recipe("notebooks/06_compute_histograms.ipynb", CMP_HST)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Compute ridges
if CMP_RDG in recipes_to_update:
    recipe = meow.register_recipe("notebooks/07_compute_ridges.ipynb", CMP_RDG)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Compute probabilities
if CMP_PRB in recipes_to_update:
    recipe = meow.register_recipe("notebooks/08_compute_probabilities.ipynb", CMP_PRB)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Compute segmentation 
if CMP_SEG in recipes_to_update:
    recipe = meow.register_recipe("notebooks/09_compute_segmentation.ipynb", CMP_SEG)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Compute bone area
if CMP_BA in recipes_to_update:
    recipe = meow.register_recipe("notebooks/10_compute_bone_area.ipynb", CMP_BA)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)

# Repeat Histogram with constraints
if RPT_CNT in recipes_to_update:
    recipe = meow.register_recipe("notebooks/11_repeat_histogram_with_constraints.ipynb", RPT_CNT)
    meow.write_dir_recipe(recipe, directory=MEOW_RUNNER_DATA)
