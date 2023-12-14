# preprocess.py
# shi zhang
# This file resize the image for uniformity
# and apply filters

# import statements
import os
import cv2
import numpy as np


def preprocess_image(image):
    # This function preprocesses a given image. It first resizes the image to a fixed width (800 pixels)
    # while maintaining the aspect ratio. Then, it converts the image to grayscale. This function can optionally
    # apply histogram equalization to the grayscale image for enhanced contrast.
    # Compute new dimensions while preserving aspect ratio

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
    # This function processes all JPEG images in a given folder. It reads each image, applies the preprocessing
    # steps defined in 'preprocess_image', and saves the processed images to an output folder if specified.
    # This function is useful for batch processing of images for tasks like image analysis or machine learning input preparation.
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

            # To view the images
            cv2.imshow("Processed Image", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    process_images_in_folder('images/blaine_house/',
                             'preprocessed_images/blaine_house/')
