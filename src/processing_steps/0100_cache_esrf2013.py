#! /usr/bin/python3
'''
This script downloads the contents of a volume from the ESRF 2013 dataset stored
on ERDA.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

from config.paths import esrf_data_sftp, esrf_data_local
import fabric
from lib.py.commandline_args import default_parser
import tqdm

if __name__ == "__main__":
    argparser = default_parser(description=__doc__)
    argparser.add_argument('experiment', action='store', type=str, default='esrf_dental_implants_april_2013', nargs='?',
        help='The experiment name to download. Default is "esrf_dental_implants_april_2013".')
    args = argparser.parse_args()

    # Read the index file for the sample
    index_dir  = f"{esrf_data_local}/{args.experiment}/index/"
    with open(f"{index_dir}/{args.sample}.txt") as f:
        volume_xmls = f.readlines()

    # Download the contents of each volume
    for volume_xml in volume_xmls:
        volume_dir = os.path.dirname(volume_xml)
        local_directory = f"{esrf_data_local}/{args.experiment}/{volume_dir}"
        sftp_directory  = f"{esrf_data_sftp}/{args.experiment}/{volume_dir}"

        if args.verbose >= 1: print(f"Local: Creating directory {local_directory}")
        pathlib.Path(local_directory).mkdir(parents=True, exist_ok=True)

        with fabric.Connection('erda') as connection:
            if args.verbose >= 1: print("Connected to ERDA")

            with connection.sftp() as sftp:
                if args.verbose >= 1: print(f"SFTP: Attempting to chdir to {sftp_directory}")
                sftp.chdir(sftp_directory)

                if args.verbose >= 1: print(f"SFTP: Reading directory contents")
                files = sftp.listdir()

                if args.verbose >= 1: print(f"Local: Attempting to chdir to {local_directory}")
                os.chdir(local_directory)

                if args.verbose >= 1: print("SFTP: Downloading subvolume contents")
                for f in tqdm.tqdm(files):
                    sftp.get(f, f)

            connection.close()
