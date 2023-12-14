# feature_matching.py
# Shi Zhang
# This file is for feature detection and matching

# import statements
import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Initialize SIFT with custom parameters
nfeatures = 15000  # Increase for more features
contrastThreshold = 0.02  # Decrease to retain more features with lower contrast
edgeThreshold = 6  # Decrease to retain more features that are edge-like
sigma = 1.6  # Typically left at default

sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrastThreshold,
                       edgeThreshold=edgeThreshold, sigma=sigma)


def draw_title(img, title, font_scale=1, font=cv2.FONT_HERSHEY_SIMPLEX, y_offset=30):
    # Adds a title to an image at a specified position.
    # The title is centrally aligned with a specified font scale and color.
    text_size = cv2.getTextSize(title, font, font_scale, 1)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    cv2.putText(img, title, (text_x, y_offset),
                font, font_scale, (0, 255, 0), 2)


def detect_and_match_features(image1, image2, sift, pair_name, save_path):
    # Detects and matches features between two images using SIFT and FLANN-based matcher.
    # It filters good matches based on Lowe's ratio test and draws top matches for visualization.
    # The resulting image with matches is saved to the specified path.
    # Returns the good matches and keypoints for both images.

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
    FLANN_INDEX_KDTREE = 1  # KD-Tree algorithm
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=9)
    search_params = dict(checks=150)

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
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    print(f"{len(good_matches)} matches passed Lowe's ratio test.")

    # Apply RANSAC to filter out outliers
    if len(good_matches) > 4:  # Minimum number of matches to find the homography
        ptsA = np.float32(
            [keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32(
            [keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(
            ptsA, ptsB, cv2.RANSAC, 4.0, maxIters=2000)
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

    # Adding title to the image
    draw_title(img_matches, 'Feature Matches')

    # Resize image if necessary (similar to adjusting figure size)
    desired_width = 1200  # Example width, adjust as needed
    scale_ratio = desired_width / img_matches.shape[1]
    img_matches = cv2.resize(img_matches, None, fx=scale_ratio, fy=scale_ratio)

    # Save the image
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_file = os.path.join(save_path, f"matches_{pair_name}.png")
    cv2.imwrite(output_file, img_matches)

    print(f"Saved matching visualization to {output_file}")

    return good_matches, keypoints1, keypoints2


def load_images_from_folder(folder):
    # Loads all grayscale images from a specified folder.
    # Returns a list of tuples containing the filename and the image.
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((filename, img))
    return images


def main():
    # Main function to process images for feature detection and matching.
    # It loads images from a folder, performs feature matching, and saves the results.
    folder_path = 'preprocessed_images/british_museum/'
    save_path = 'output/feature_matching/british_museum/'
    images = load_images_from_folder(folder_path)

    all_matches = []  # Initialize the list to store all matches

    # Perform feature matching between all pairs of images
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            image1_name, image1 = images[i]
            image2_name, image2 = images[j]

            # Generate a unique name for each pair of images
            pair_name = f"{image1_name}_vs_{image2_name}"

            h1, w1 = image1.shape[:2]  # Height and width of image1
            h2, w2 = image2.shape[:2]  # Height and width of image2

            matches, keypoints1, keypoints2 = detect_and_match_features(
                image1, image2, sift, pair_name, save_path)

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
