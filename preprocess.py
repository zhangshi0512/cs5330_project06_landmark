# preprocess.py
# shi zhang
# This file resize the image for uniformity
# and apply filters

import os
import cv2
import numpy as np


def preprocess_image(image):
    # Compute new dimensions while preserving aspect ratio
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = int(800)
    new_height = int(new_width / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Optionally apply histogram equalization
    # gray_image = cv2.equalizeHist(gray_image)

    return gray_image


def process_images_in_folder(folder_path, output_folder=None):
    print(f"Processing images in folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {file_path}...")
            image = cv2.imread(file_path)
            processed_image = preprocess_image(image)

            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, processed_image)
                print(f"Saved processed image to {output_path}")

            # Uncomment below lines to view the images
            cv2.imshow("Processed Image", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    process_images_in_folder('images/basketball/',
                             'preprocessed_images/basketball/')
