# sfm_dense_reconstruction.py

import os
import subprocess


def run_dense_reconstruction(dataset_path):
    # Run the undistort and compute_depthmaps commands
    commands = [
        'undistort',
        'compute_depthmaps'
    ]

    for command in commands:
        cmd = f'OpenSfM/bin/opensfm {command} {dataset_path}'
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


def main(dataset_name):
    # Define the path to your dataset
    opensfm_root = './OpenSfM'  # Replace with your OpenSfM installation path
    dataset_path = os.path.join(opensfm_root, 'data', dataset_name)

    # Run dense reconstruction
    run_dense_reconstruction(dataset_path)


if __name__ == "__main__":
    main()
