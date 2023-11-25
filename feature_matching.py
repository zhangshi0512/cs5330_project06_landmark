# feature_matching.py
# shi zhang
# This file is for feature detection and matching

import cv2

def detect_and_match_features(image1, image2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Feature matching
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # Draw matches
    matched_image = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)
    return matched_image

# Load images
image1 = cv2.imread('image1.jpg', 0)  # 0 for grayscale
image2 = cv2.imread('image2.jpg', 0)

# Match features
result = detect_and_match_features(image1, image2)
cv2.imshow('Feature Matches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
