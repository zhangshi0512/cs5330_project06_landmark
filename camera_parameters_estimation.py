# camera_parameters_estimation.py
# Shi Zhang
# This file is for estimating camera parameters

import cv2
import numpy as np
import csv
import os


def load_feature_matches(csv_file):
    matches = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            image1, image2, x1, y1, x2, y2 = row
            if (image1, image2) not in matches:
                matches[(image1, image2)] = []
            matches[(image1, image2)].append(
                (float(x1), float(y1), float(x2), float(y2)))
    return matches


def estimate_camera_parameters(matches, intrinsic_parameters):
    camera_params = {}
    for (image1, image2), match_points in matches.items():
        # Assuming all points lie on a plane with Z=0
        object_points = np.array(
            [[x, y, 0] for x, y, _, _ in match_points], dtype=np.float32)
        image_points = np.array(
            [[x, y] for _, _, x, y in match_points], dtype=np.float32)

        # Estimating pose
        _, rvec, tvec = cv2.solvePnP(
            object_points, image_points, intrinsic_parameters, None)

        camera_params[image1] = {'rvec': rvec, 'tvec': tvec}
        camera_params[image2] = {'rvec': rvec, 'tvec': tvec}

    return camera_params


def save_camera_params_to_file(camera_params, filename):
    with open(filename, 'w') as file:
        for image, params in camera_params.items():
            rvec, tvec = params['rvec'], params['tvec']
            file.write(
                f"{image},{rvec[0][0]},{rvec[1][0]},{rvec[2][0]},{tvec[0][0]},{tvec[1][0]},{tvec[2][0]}\n")


def main():
    csv_file = 'feature_matches.csv'
    matches = load_feature_matches(csv_file)

    # Example intrinsic parameters
    focal_length = 1.0
    principal_point = (0.5, 0.5)
    intrinsic_parameters = np.array([[focal_length, 0, principal_point[0]],
                                     [0, focal_length, principal_point[1]],
                                     [0, 0, 1]])

    # Estimate camera parameters
    camera_params = estimate_camera_parameters(matches, intrinsic_parameters)

    # Save camera parameters to a file
    camera_params_file = 'camera_parameters.csv'
    save_camera_params_to_file(camera_params, camera_params_file)


if __name__ == '__main__':
    main()
