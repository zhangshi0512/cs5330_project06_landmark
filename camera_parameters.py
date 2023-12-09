# camera_parameters.py

import csv
import numpy as np


def load_camera_parameters(filename):
    camera_params = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) != 7:
                print(f"Skipping invalid row: {row}")
                continue
            image, rvec1, rvec2, rvec3, tvec1, tvec2, tvec3 = row
            camera_params[image] = np.array([
                [float(rvec1), float(rvec2), float(rvec3)],
                [float(tvec1), float(tvec2), float(tvec3)]
            ])
    return camera_params


def load_feature_matches(csv_file):
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
