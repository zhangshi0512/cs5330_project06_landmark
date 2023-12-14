# OpenSfM.py
# Shi Zhang, Zhizhou Gu
# OpenSfM command line feature extraction and matching

import os
import subprocess


def run_opensfm_commands(dataset_path):
    # Define OpenSfM commands to run
    commands = [
        'extract_metadata',
        'detect_features',
        'match_features',
        'create_tracks'
    ]

    # Execute each command in order
    for command in commands:
        # TBD: change hard code
        cmd = f'OpenSfM/bin/opensfm {command} {dataset_path}'
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


def main():
    # Define the path to your dataset
    dataset_name = 'lund'  # Replace with your dataset name
    opensfm_root = './OpenSfM'  # Replace with your OpenSfM installation path
    dataset_path = os.path.join(opensfm_root, 'data', dataset_name)

    # Copy config file from the example dataset
    example_config = os.path.join(
        opensfm_root, 'data', dataset_name, 'config.yaml')
    dataset_config = os.path.join(dataset_path, 'config.yaml')
    if not os.path.exists(dataset_config):
        os.makedirs(os.path.dirname(dataset_config), exist_ok=True)
        shutil.copy(example_config, dataset_config)

    # Run OpenSfM commands
    run_opensfm_commands(dataset_path)


if __name__ == "__main__":
    main()
