# camera_parameters.py

import csv
import numpy as np


def load_camera_parameters(filename):
    camera_params = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image, *params = row
            camera_params[image] = np.array(
                params, dtype=np.float).reshape(3, 2)
    return camera_params


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
