# feature_matching.py
# Shi Zhang
# This file is for feature detection and matching

import cv2
import os
import csv


def detect_and_match_features(image1, image2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Feature matching
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test and store good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches, keypoints1, keypoints2


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((filename, img))
    return images


def main():
    folder_path = 'preprocessed_images/blain_house/'
    images = load_images_from_folder(folder_path)

    # Prepare to save matches
    matches_file = open('feature_matches.csv', 'w', newline='')
    csv_writer = csv.writer(matches_file)
    csv_writer.writerow(['Image1', 'Image2', 'KeyPoint1_X',
                        'KeyPoint1_Y', 'KeyPoint2_X', 'KeyPoint2_Y'])

    # Perform feature matching between all pairs of images
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            image1_name, image1 = images[i]
            image2_name, image2 = images[j]
            matches, keypoints1, keypoints2 = detect_and_match_features(
                image1, image2)

            # Write match details to CSV
            for match in matches:
                pt1 = keypoints1[match.queryIdx].pt
                pt2 = keypoints2[match.trainIdx].pt
                csv_writer.writerow(
                    [image1_name, image2_name, pt1[0], pt1[1], pt2[0], pt2[1]])

    matches_file.close()


if __name__ == '__main__':
    main()
