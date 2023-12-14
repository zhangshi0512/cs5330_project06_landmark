
# OpenSfM.py
# Zhizhou Gu
# Work In Progress: Using OpenSfM python bind to implement reconstruct pipeline
import opensfm
from opensfm import features
import cv2
import numpy as np
import os
import pdb

def read_and_feature_detection(dir_path):
    file_names = os.listdir(dir_path)

    for file_name in file_names:
        # Load an image using OpenCV
        image = cv2.imread(dir_path + '/' + file_name)

        # The 'image' variable is now a numpy array/matrix representing the image
        # print(image)
        
        f = features.extract_features_sift(image, {'sift_edge_threshold': 6, 'sift_peak_threshold': 0.02, 'feature_root': True }, 8)        
        # print('f', f)

        breakpoint()
        
        print('f[0]', f[0])
        print('f[1]', f[1])

        # opensfm.pyfeatures.akaze(image)

read_and_feature_detection('./images/lund')