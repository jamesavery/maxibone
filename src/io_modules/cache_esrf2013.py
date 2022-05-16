import os, sys, pathlib, tqdm, fabric
sys.path.append(sys.path[0]+"/../")
from config.paths import commandline_args, esrf_data_sftp, esrf_data_local

if __name__ == "__main__":
    sample, experiment = commandline_args({"sample":"<required>",
                                           "experiment":"esrf_dental_implants_april_2013"})
    
    index_dir  = f"{esrf_data_local}/{experiment}/index/";
    with open(f"{index_dir}/{sample}.txt") as f:
        volume_xmls = f.readlines()

    run_dir = os.getcwd()
    for volume_xml in volume_xmls:
        volume_dir = os.path.dirname(volume_xml)
        local_directory = f"{esrf_data_local}/{experiment}/{volume_dir}"
        sftp_directory  = f"{esrf_data_sftp}/{experiment}/{volume_dir}"
        print(f"Local: Creating directory {local_directory}")
        pathlib.Path(local_directory).mkdir(parents=True, exist_ok=True)

        with fabric.Connection('erda') as connection:
            print("Connected to ERDA")
            with connection.sftp() as sftp:
                print(f"SFTP: Attempting to chdir to {sftp_directory}")
                sftp.chdir(sftp_directory)
                print(f"SFTP: Reading directory contents")
                files = sftp.listdir()
                print(f"Local: Attempting to chdir to {local_directory}")                
                os.chdir(local_directory)
                print("SFTP: Downloading subvolume contents")
                for f in tqdm.tqdm(files):
                    sftp.get(f,f)

            connection.close()
    
