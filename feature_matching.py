# feature_matching.py
# Shi Zhang
# This file is for feature detection and matching

import cv2
import os
import csv
import matplotlib.pyplot as plt

# Initialize SIFT detector
sift = cv2.SIFT_create()


def detect_and_match_features(image1, image2, sift):
    # Detect and compute keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Check if SIFT found descriptors
    if descriptors1 is None or descriptors2 is None:
        print(f"Not enough features in one of the images. Skipping this pair.")
        return [], [], []

    # FLANN-based matcher parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Matching descriptor vectors using FLANN matcher
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Store all good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)

    # Draw top matches
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                  good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show detected matches
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('Feature Matches')
    plt.axis('off')
    plt.show()

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

    # Extract image dimensions for intrinsic parameters approximation
    sample_image_path = os.path.join(folder_path, images[0][0])
    sample_image = cv2.imread(sample_image_path)
    h, w = sample_image.shape[:2]
    focal_length = w  # Approximate focal length

    all_matches = []  # Initialize the list to store all matches

    # Perform feature matching between all pairs of images
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            image1_name, image1 = images[i]
            image2_name, image2 = images[j]
            matches, keypoints1, keypoints2 = detect_and_match_features(
                image1, image2, sift)

            # Process matches and store them in all_matches
            for match in matches:
                pt1 = keypoints1[match.queryIdx].pt
                pt2 = keypoints2[match.trainIdx].pt
                all_matches.append(
                    [image1_name, image2_name, pt1[0], pt1[1], pt2[0], pt2[1]])

    # Write all matches to CSV at once
    with open('feature_matches.csv', 'w', newline='') as matches_file:
        csv_writer = csv.writer(matches_file)
        csv_writer.writerow(['Image1', 'Image2', 'KeyPoint1_X',
                            'KeyPoint1_Y', 'KeyPoint2_X', 'KeyPoint2_Y'])
        csv_writer.writerows(all_matches)


if __name__ == '__main__':
    main()
