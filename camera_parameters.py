# camera_parameters.py
# Shi Zhang
# This file is for camera parameters process

# import statements
import cv2
import numpy as np
import csv


def load_camera_parameters(filename):
    # Loads camera parameters from a specified CSV file.
    # The parameters include rotation vectors and translation vectors for each image.
    # Converts the rotation vectors to rotation matrices.
    # Returns a dictionary mapping each image to its corresponding camera parameters.
    camera_params = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            if len(row) != 7:
                print(f"Skipping invalid row: {row}")
                continue
            image, rvec1, rvec2, rvec3, tvec1, tvec2, tvec3 = row
            rvec = np.array([float(rvec1), float(rvec2), float(rvec3)])
            tvec = np.array([float(tvec1), float(tvec2), float(tvec3)])

            # Convert rotation vector to rotation matrix
            r_mat, _ = cv2.Rodrigues(rvec)

            camera_params[image] = {'r_mat': r_mat, 'tvec': tvec}
    return camera_params


def load_feature_matches(csv_file):
    # Reads feature matches and image dimensions from a CSV file.
    # Organizes the matches into a dictionary indexed by image pairs.
    # Also extracts and stores the dimensions of each image.
    # Returns a tuple containing the matches dictionary and a dimensions dictionary.
    matches = {}
    dimensions = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            print(row)
            image1, image2, x1, y1, x2, y2, w1, h1, w2, h2 = row
            if (image1, image2) not in matches:
                matches[(image1, image2)] = []
                dimensions[image1] = (float(w1), float(h1))
                dimensions[image2] = (float(w2), float(h2))
            matches[(image1, image2)].append(
                (float(x1), float(y1), float(x2), float(y2)))
    return matches, dimensions
