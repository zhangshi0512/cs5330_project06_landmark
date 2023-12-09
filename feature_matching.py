# feature_matching.py
# Shi Zhang
# This file is for feature detection and matching

import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

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

    print(
        f"Found {len(keypoints1)} and {len(keypoints2)} keypoints in the images respectively.")

    # Convert descriptors to float32 for FLANN
    if descriptors1.dtype != np.float32:
        descriptors1 = descriptors1.astype(np.float32)
    if descriptors2.dtype != np.float32:
        descriptors2 = descriptors2.astype(np.float32)

    # FLANN-based matcher parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    # Matching descriptor vectors using FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    print(f"Found {len(matches)} raw matches.")

    # Print out distances of raw matches
    for i, (m, n) in enumerate(matches):
        print(
            f"Match {i}: Distance 1 - {m.distance}, Distance 2 - {n.distance}")
        if i == 10:  # Just print the first 10 to check the range
            break

    # Store all good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"{len(good_matches)} matches passed Lowe's ratio test.")

    """
    distance_threshold = 0.7
    good_matches = [m for m in good_matches if m.distance < distance_threshold]

    print(f"{len(good_matches)} matches passed the distance threshold.")
    """

    # Apply RANSAC to filter out outliers
    if len(good_matches) > 4:  # Minimum number of matches to find the homography
        ptsA = np.float32(
            [keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32(
            [keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(
            ptsA, ptsB, cv2.RANSAC, 3.0, maxIters=2000)
        if mask is not None:
            matchesMask = mask.ravel().tolist()
            good_matches = [gm for gm, mask in zip(
                good_matches, matchesMask) if mask]
            print(f"{sum(matchesMask)} matches survived the RANSAC filter.")
        else:
            print("RANSAC homography could not be computed. All matches discarded.")
            matchesMask = None
    else:
        print("Not enough good matches to apply RANSAC.")
        matchesMask = None

    # Draw top matches
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                  good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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
    folder_path = 'preprocessed_images/basketball/'
    images = load_images_from_folder(folder_path)

    all_matches = []  # Initialize the list to store all matches

    # Perform feature matching between all pairs of images
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            image1_name, image1 = images[i]
            image2_name, image2 = images[j]

            h1, w1 = image1.shape[:2]  # Height and width of image1
            h2, w2 = image2.shape[:2]  # Height and width of image2

            matches, keypoints1, keypoints2 = detect_and_match_features(
                image1, image2, sift)

            # Process matches and store them in all_matches
            for match in matches:
                pt1 = keypoints1[match.queryIdx].pt
                pt2 = keypoints2[match.trainIdx].pt
                all_matches.append(
                    [image1_name, image2_name, pt1[0], pt1[1], pt2[0], pt2[1], w1, h1, w2, h2])

    # Write all matches and dimensions to CSV at once
    with open('feature_matches.csv', 'w', newline='') as matches_file:
        csv_writer = csv.writer(matches_file)
        csv_writer.writerow(['Image1', 'Image2', 'KeyPoint1_X', 'KeyPoint1_Y',
                            'KeyPoint2_X', 'KeyPoint2_Y', 'Width1', 'Height1', 'Width2', 'Height2'])
        csv_writer.writerows(all_matches)


if __name__ == '__main__':
    main()
