# MAXIBONE
Source code for SRμCT Bone tomography analysis project.

# How to build
## For GPU support, ensure the NVIDIA HPC SDK is in your path
## Clone the repository
```sh
git clone git@github.com:jamesavery/maxibone.git
```

## Determine python version
```sh
python --version
```
Verify that the first line of `src/Makefile` is set to the correct python version. Otherwise, change it to the correct version.

## Change to the source directory
```sh
cd maxibone/src/
```

## Install the Python generic dependencies
```sh
make pip_generic
```

## Install the Python dependencies for the GPU
```sh
make pip_cuda
```

## Build the project
```sh
make -j
```

## Edit the paths in `src/config/paths.py` to point to the correct data directories

## Start running the processing scripts
The scripts should be run from the `src` directory. Most of them only need the sample name as argument. For example:
```sh
python3 processing_steps/0400_h5tobin.py 770c_pag
```
Each script has a help message that can be displayed by running it with the `-h` flag:
```sh
python3 processing_steps/0400_h5tobin.py -h
```
Outputs:
```txt
usage: 0400_h5tobin.py [-h] [-c CHUNK_SIZE] [--disable-plotting] [-v VERBOSE] [--version] sample [sample_scale] [y_cutoff]

Converts a 16-bit HDF5 file to a 16-bit binary file. The binary file can be read faster than the HDF5 file, if it is stored on a fast disk.

positional arguments:
  sample                The name of the sample volume to be processed.
  sample_scale          The scale of the sample volume to be processed. Default is 1.
  y_cutoff              The y-coordinate to cut the volume at. Default is 0.

options:
  -h, --help            show this help message and exit
  -c CHUNK_SIZE, --chunk-size CHUNK_SIZE
                        The size of the z-axis of the chunks to be processed. Default is 64.
  --disable-plotting    Disable plotting the results of the script.
  -v VERBOSE, --verbose VERBOSE
                        Set the verbosity level of the script. Default is 1. Generally, 0 is no output, 1 is progress / some text output, 2 is helper/core function output, and 3 is extreme debugging.
  --version             Print the version of the script and exit.

For more information, please visit github.com/jamesavery/maxibone
```