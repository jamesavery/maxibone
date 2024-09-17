#! /usr/bin/python3
'''
This script downloads the contents of a volume from the ESRF 2013 dataset stored
on ERDA.
'''
import os, sys, pathlib, tqdm, fabric
sys.path.append(sys.path[0]+"/../")
from lib.py.helpers import commandline_args
from config.paths import esrf_data_sftp, esrf_data_local

if __name__ == "__main__":
    sample, experiment, verbose = commandline_args({
        "sample" : "<required>",
        "experiment" : "esrf_dental_implants_april_2013",
        "verbose" : 1
    })

    # Read the index file for the sample
    index_dir  = f"{esrf_data_local}/{experiment}/index/"
    with open(f"{index_dir}/{sample}.txt") as f:
        volume_xmls = f.readlines()

    # Download the contents of each volume
    for volume_xml in volume_xmls:
        volume_dir = os.path.dirname(volume_xml)
        local_directory = f"{esrf_data_local}/{experiment}/{volume_dir}"
        sftp_directory  = f"{esrf_data_sftp}/{experiment}/{volume_dir}"

        if verbose >= 1: print(f"Local: Creating directory {local_directory}")
        pathlib.Path(local_directory).mkdir(parents=True, exist_ok=True)

        with fabric.Connection('erda') as connection:
            if verbose >= 1: print("Connected to ERDA")

            with connection.sftp() as sftp:
                if verbose >= 1: print(f"SFTP: Attempting to chdir to {sftp_directory}")
                sftp.chdir(sftp_directory)

                if verbose >= 1: print(f"SFTP: Reading directory contents")
                files = sftp.listdir()

                if verbose >= 1: print(f"Local: Attempting to chdir to {local_directory}")
                os.chdir(local_directory)

                if verbose >= 1: print("SFTP: Downloading subvolume contents")
                for f in tqdm.tqdm(files):
                    sftp.get(f, f)

            connection.close()
