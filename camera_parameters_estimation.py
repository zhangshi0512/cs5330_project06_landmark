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
        object_points = np.array(
            [[x, y, 0] for x, y, _, _ in match_points], dtype=np.float32)
        image_points = np.array(
            [[x, y] for _, _, x, y in match_points], dtype=np.float32)

        # RANSAC
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, intrinsic_parameters, None)

        # RANSAC refinement
        # Choose a threshold for the minimum inliers
        if inliers is None or len(inliers) < 5:
            print(
                f"Not enough inliers for image pair: {image1} - {image2}. Skipping.")
            continue

        # Refine using inliers
        object_points = object_points[inliers]
        image_points = image_points[inliers]
        _, rvec, tvec = cv2.solvePnP(
            object_points, image_points, intrinsic_parameters, None)

        # Reprojection error
        proj_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, intrinsic_parameters, None)
        error = np.mean(np.linalg.norm(proj_points - image_points, axis=1))

        # Rotation matrix and translation vector
        r_mat, _ = cv2.Rodrigues(rvec)

        camera_params[image1] = {'r_mat': r_mat, 'tvec': tvec.reshape(-1, 1)}
        camera_params[image2] = {'r_mat': r_mat, 'tvec': tvec.reshape(-1, 1)}

    return camera_params


def save_camera_params_to_file(camera_params, filename):
    with open(filename, 'w') as file:
        for image, params in camera_params.items():
            r_mat, tvec = params['r_mat'], params['tvec']
            rvec, _ = cv2.Rodrigues(r_mat)
            file.write(
                f"{image},{rvec[0][0]},{rvec[1][0]},{rvec[2][0]},{tvec[0][0]},{tvec[1][0]},{tvec[2][0]}\n")


def main():
    csv_file = 'feature_matches.csv'
    matches = load_feature_matches(csv_file)

    # Approximated intrinsic parameters
    focal_length = w  # use the width of the image from the feature_matching.py
    principal_point = (w / 2, h / 2)
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
