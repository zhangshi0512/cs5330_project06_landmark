# cs5330_project06_landmark

CS5330 Pattern Recognition & Computer Vision

NEU 2023 Fall

Instructor: Bruce Maxwell

Team: Shi Zhang, ZhiZhou Gu

# 3D Scene Reconstruction from Images

## Overview

This project aims to reconstruct 3D scenes from a collection of 2D images. It involves several steps from preprocessing images to visualizing the reconstructed 3D scene.

## Tasks

### 1. Data Preprocessing:

- Select a subset of images from the dataset.
- Preprocess images using OpenCV.
  - Requirements:
    - Resize images for uniformity.
    - Convert images to grayscale.
    - Apply optional filters like edge detection.

### 2. Feature Detection and Matching:

- Implement feature detection algorithms to find key points in images.
- Match these features across different images of the same landmark.
  - Techniques:
    - Use algorithms like SIFT or SURF for feature detection.
    - Employ feature matching methods like FLANN or BFMatcher.

### 3. Estimating Camera Parameters:

- Estimate the camera parameters and pose for each image.
  - Requirements:
    - Determine intrinsic parameters (like focal length, principal point).
    - Estimate extrinsic parameters (like rotation, translation vectors).

### 4. 3D Reconstruction:

- Use triangulation methods to estimate 3D points from matched 2D points.
- Generate a point cloud representing the 3D scene.
  - Techniques:
    - Implement triangulation using the camera parameters and feature matches.
    - Assemble the 3D points into a point cloud.

### 5. Optimization and Refinement:

- Apply bundle adjustment to refine camera parameters and 3D point estimates.
- Clean up the point cloud for noise and outliers.
  - Techniques:
    - Use optimization libraries like OpenCV or g2o for bundle adjustment.
    - Implement noise reduction and outlier removal in the point cloud.

### 6. Visualization and Analysis:

- Visualize the 3D reconstruction using appropriate tools.
- Analyze the accuracy and quality of the reconstruction.
  - Tools:
    - Use PCL (Point Cloud Library) or MeshLab for visualization.
    - Evaluate the reconstruction quality against ground truth data, if available.

## Acknowledgement

### Reference Tutorial

[From 2D to 3D: 4 Ways to Make a 3D Reconstruction from Imagery](https://www.youtube.com/watch?v=7uDXIIRpv80)

[3D Scene Reconstruction from Multiple Views](https://www.youtube.com/watch?v=9eXcCEJ6qoE)

[3D Reconstruction Playlist](https://www.youtube.com/playlist?list=PLRW7gspIIsxvRmzO1n9t-B90mpArhmIG9)

[2D to 3D surface reconstruction from images and point clouds](https://www.youtube.com/playlist?list=PL3OV2Akk7XpDjlhJBDGav08bef_DvIdH2)

### Reference Paper

[Google Landmarks Dataset v2 -- A Large-Scale Benchmark for Instance-Level Recognition and Retrieval, Tobias Weyand, Andre Araujo, Bingyi Cao, Jack Sim](https://arxiv.org/abs/2004.01804)

[Optimization of Rank Losses for Image Retrieval, Elias Ramzi, Nicolas Audebert, Clément Rambour, André Araujo, Xavier Bitot, Nicolas Thome](https://arxiv.org/abs/2309.08250)

[Project: Camera Rectification and Structure from Motion, CIS 580, Machine Perception, Spring 2017](https://www.cis.upenn.edu/~cis580/Spring2017/Projects/project2/CIS580_Project2_v3.pdf)

[3D Reconstruction from Multiple Images, Shawn McCann](https://cvgl.stanford.edu/teaching/cs231a_winter1415/prev/projects/CS231a-FinalReport-sgmccann.pdf)

[Structure from Motion, A demo using the tools in the MATLAB Computer Vision toolbox](https://cs-courses.mines.edu/csci508/schedule/10/SfMDemo.pdf)

## Project Running Instructions

### Project Setup

Our team used Visual Studio Code for this project.

Before running this project, you will need to install several libraries.

Open the command prompt, and navigate to the python folder in your local environment, normally like: C:\Users\AppData\Local\Programs\Python\Python310>

Then run following commands

```markdown
pip install opencv-python numpy scipy matplotlib
pip install open3d
```

### Project Running

#### Task 1

Run preprocess.py, it should go through image files in a sub-folder under 'images' folder and convert each image to the same size grayscale images saved under a sub-folder under 'preprocessed_images' folder.

#### Task 2

Run feature_matching.py, it should load images from 'preprocessed_images' folder, implement feature detection algorithms to find key points in images, then save the image features as a csv file, like 'feature_matches.csv'.

#### Task 3

Run camera_parameters_estimation.py, it should first read features from the 'feature_matches.csv', and estimate the camera parameters and pose for each image, and save the camera parameters as a csv file, like 'camera_parameters.csv'.

#### Task 4 - 6

Run reconstruction_visualization.py, it should load camera parameters and feature matches, triangulate points, then create and visualize the point cloud.
