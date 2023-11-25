# cs5330_project06_landmark

CS5330 Pattern Recognition & Computer Vision

NEU 2023 Fall

Instructor: Bruce Maxwell

Team: Shi Zhang, ZhiZhou Gu

## Tasks

1. Data Preprocessing:

Select a subset of images from the dataset.
Preprocess images (resizing, filtering, etc.) using OpenCV.

2. Feature Detection and Matching:

Implement feature detection algorithms (like SIFT, SURF) to find key points in images.
Match these features across different images of the same landmark.

3. Estimating Camera Parameters:

Estimate the camera parameters and pose for each image.
This might involve understanding intrinsic and extrinsic camera parameters.

4. 3D Reconstruction:

Use triangulation methods to estimate 3D points from matched 2D points.
Generate a point cloud representing the 3D scene.

5. Optimization and Refinement:

Apply bundle adjustment to refine camera parameters and 3D point estimates.
Clean up the point cloud for noise and outliers.

6. Visualization and Analysis:

Visualize the 3D reconstruction using appropriate tools (e.g., PCL, MeshLab).
Analyze the accuracy and quality of the reconstruction.
