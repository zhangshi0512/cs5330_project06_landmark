# camera_parameters_estimation.py
# Shi Zhang
# This file is for estimating camera parameters

import cv2
import numpy as np
import open3d as o3d
import csv
import os

# Function to estimate intrinsic camera parameters


def estimate_intrinsic_parameters(matches, dimensions):
    image_names = []
    for (image1, image2), match_points in matches.items():
        if image1 not in image_names:
            image_names.append(image1)
        if image2 not in image_names:
            image_names.append(image2)

    unique_images = set(image_names)
    num_images = len(unique_images)

    image_points = {}
    for (image1, image2), match_points in matches.items():
        if image1 not in image_points:
            image_points[image1] = []
        if image2 not in image_points:
            image_points[image2] = []

        for x1, y1, x2, y2 in match_points:
            image_points[image1].append((x1, y1))
            image_points[image2].append((x2, y2))

    image_points_list = [np.array(
        image_points[img], dtype=np.float32).reshape(-1, 1, 2) for img in image_names]

    # Initialize object points for each image
    object_points_list = []
    for img in image_names:
        num_points = len(image_points[img])
        object_points_3d = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            # Assuming a 10x10 grid, modify as needed
            object_points_3d[i] = [i % 10, i // 10, 0]

        object_points_list.append(object_points_3d)

    camera_matrices = {}
    dist_coeffs = {}

    for idx, img_name in enumerate(image_names):
        img_points = image_points_list[idx]
        obj_points = object_points_list[idx]

        w, h = dimensions[img_name]
        camera_matrix = np.array(
            [[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        dist_coeff = np.zeros((4, 1), dtype=np.float32)

        print(f"Processing image: {img_name}")
        print(f"Number of object points: {len(obj_points)}")
        print(f"Number of image points: {len(img_points)}")
        print(f"Shape of image points for current image: {img_points.shape}")

        _, camera_matrix, dist_coeff, _, _ = cv2.calibrateCamera(
            [obj_points], [img_points], (w, h), camera_matrix, dist_coeff,
            flags=cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO
        )

        camera_matrices[img_name] = camera_matrix
        dist_coeffs[img_name] = dist_coeff

    return camera_matrices, dist_coeffs


def load_feature_matches(csv_file):
    matches = {}
    dimensions = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            # Unpack all values from each row
            image1, image2, x1, y1, x2, y2, w1, h1, w2, h2 = row
            if (image1, image2) not in matches:
                matches[(image1, image2)] = []
                dimensions[image1] = (int(w1), int(h1))
                dimensions[image2] = (int(w2), int(h2))
            matches[(image1, image2)].append(
                (float(x1), float(y1), float(x2), float(y2)))
    return matches, dimensions


def estimate_camera_parameters(matches, dimensions):
    camera_params = {}

    for (image1, image2), match_points in matches.items():
        w1, h1 = dimensions[image1]
        w2, h2 = dimensions[image2]

        # Use the average of the dimensions from both images
        avg_width = (w1 + w2) / 2
        avg_height = (h1 + h2) / 2

        # Create the Open3D PinholeCameraIntrinsic object
        intrinsic = o3d.camera.PinholeCameraIntrinsic(int(avg_width), int(avg_height),
                                                      avg_width, avg_width,
                                                      avg_width / 2, avg_height / 2)

        # Convert Open3D intrinsic to OpenCV matrix for solvePnP
        intrinsic_parameters = np.array(intrinsic.intrinsic_matrix)

        object_points = np.array(
            [[x, y, 0] for x, y, _, _ in match_points], dtype=np.float32)
        image_points = np.array(
            [[x, y] for _, _, x, y in match_points], dtype=np.float32)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, intrinsic_parameters, None)

        if inliers is None or len(inliers) < 5:
            print(
                f"Not enough inliers for image pair: {image1} - {image2}. Skipping.")
            continue

        object_points = object_points[inliers.ravel()]
        image_points = image_points[inliers.ravel()]

        _, rvec, tvec = cv2.solvePnP(
            object_points, image_points, intrinsic_parameters, None)

        proj_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, intrinsic_parameters, None)
        error = np.mean(np.linalg.norm(
            proj_points.squeeze() - image_points, axis=1))

        r_mat, _ = cv2.Rodrigues(rvec)
        camera_params[image1] = {'r_mat': r_mat, 'tvec': tvec}
        camera_params[image2] = {'r_mat': r_mat, 'tvec': tvec}

    return camera_params


def save_camera_params_to_file(camera_params, filename):
    with open(filename, 'w') as file:
        for image, params in camera_params.items():
            r_mat, tvec = params['r_mat'], params['tvec']
            rvec, _ = cv2.Rodrigues(r_mat)
            file.write(
                f"{image},{rvec[0][0]},{rvec[1][0]},{rvec[2][0]},{tvec[0][0]},{tvec[1][0]},{tvec[2][0]}\n")


def save_intrinsic_params_to_file(intrinsic_matrices, dist_coeffs, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Camera Matrix', 'Distortion Coefficients'])
        for image, intrinsic in intrinsic_matrices.items():
            dist = dist_coeffs[image]
            writer.writerow(
                [image, intrinsic.flatten().tolist(), dist.flatten().tolist()])


def main():
    csv_file = 'feature_matches.csv'
    matches, dimensions = load_feature_matches(csv_file)

    # Estimate camera parameters
    camera_params = estimate_camera_parameters(matches, dimensions)

    # Estimate intrinsic parameters
    intrinsic_matrices, dist_coeffs = estimate_intrinsic_parameters(
        matches, dimensions)

    # Save camera parameters to a file
    camera_params_file = 'camera_parameters.csv'
    save_camera_params_to_file(camera_params, camera_params_file)

    # Save the intrinsic parameters and distortion coefficients
    intrinsic_params_file = 'intrinsic_parameters.csv'
    save_intrinsic_params_to_file(
        intrinsic_matrices, dist_coeffs, intrinsic_params_file)


if __name__ == '__main__':
    main()
