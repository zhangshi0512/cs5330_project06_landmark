# preprocess.py
# shi zhang
# This file resize the image for uniformity
# and apply filters

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (800, 600))  # Resize for uniformity
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return gray_image

# Example usage
image = load_and_preprocess_image('path_to_image.jpg')
plt.imshow(image, cmap='gray')
plt.show()

