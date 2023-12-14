# OpenSfM.py
# Shi Zhang, Zhizhou Gu
# OpenSfM command line reconstruction

import os
import subprocess


def run_sfm_reconstruction(dataset_path):
    # Run the SfM reconstruction command
    cmd = f'OpenSfM/bin/opensfm reconstruct {dataset_path}'
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    # Define the path to your dataset
    dataset_name = 'lund'  # Replace with your dataset name
    opensfm_root = './OpenSfM'  # Replace with your OpenSfM installation path
    dataset_path = os.path.join(opensfm_root, 'data', dataset_name)

    # Run SfM reconstruction
    run_sfm_reconstruction(dataset_path)


if __name__ == "__main__":
    main()
