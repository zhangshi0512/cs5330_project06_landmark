# reconstruction_visualization.py

import cv2
import csv
import numpy as np
import open3d as o3d
from camera_parameters import load_camera_parameters, load_feature_matches


def load_intrinsic_parameters(filename):
    intrinsic_matrices = {}
    dist_coeffs = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            image, intrinsic_flat, dist_flat = row
            # Convert string representations back to numpy arrays
            intrinsic_matrix = np.fromstring(
                intrinsic_flat.strip("[]"), sep=', ').reshape(3, 3)
            dist_coeff = np.fromstring(dist_flat.strip("[]"), sep=', ')
            intrinsic_matrices[image] = intrinsic_matrix
            dist_coeffs[image] = dist_coeff
    return intrinsic_matrices, dist_coeffs


def triangulate_points(matches, camera_params, intrinsic_matrices, dist_coeffs):
    all_points_3d = []

    for (img1, img2), match_pts in matches.items():
        print(f"Processing pair: {img1} and {img2}")

        # Check if both images have camera parameters and intrinsic parameters
        if img1 not in camera_params or img2 not in camera_params:
            print(
                f"Camera parameters for {img1} or {img2} are missing. Skipping.")
            continue

        if img1 not in intrinsic_matrices or img2 not in intrinsic_matrices:
            print(
                f"Intrinsic parameters for {img1} or {img2} are missing. Skipping.")
            continue

        # Get camera matrices (extrinsic parameters: rotation and translation)
        extrinsic1 = np.hstack(
            (camera_params[img1]['r_mat'], camera_params[img1]['tvec'].reshape(-1, 1)))
        extrinsic2 = np.hstack(
            (camera_params[img2]['r_mat'], camera_params[img2]['tvec'].reshape(-1, 1)))

        # Get intrinsic parameters and distortion coefficients
        intrinsic1 = intrinsic_matrices[img1]
        intrinsic2 = intrinsic_matrices[img2]
        dist1 = dist_coeffs[img1]
        dist2 = dist_coeffs[img2]

        # Extracting matched points and converting them to the correct format
        pts1 = np.float32([pt[0:2] for pt in match_pts])
        pts2 = np.float32([pt[2:4] for pt in match_pts])

        # Undistort points
        pts1_undistorted = cv2.undistortPoints(
            np.expand_dims(pts1, axis=1), intrinsic1, dist1)
        pts2_undistorted = cv2.undistortPoints(
            np.expand_dims(pts2, axis=1), intrinsic2, dist2)

        # Triangulate points
        P1 = intrinsic1 @ extrinsic1
        P2 = intrinsic2 @ extrinsic2
        pts_4d_hom = cv2.triangulatePoints(
            P1, P2, pts1_undistorted, pts2_undistorted)

        # Convert homogeneous coordinates to 3D points
        pts_3d = pts_4d_hom[:3] / pts_4d_hom[3]
        all_points_3d.append(pts_3d.T)

    # Combine all 3D points from each image pair
    points_3d = np.concatenate(
        all_points_3d, axis=0) if all_points_3d else np.empty((0, 3))

    # Filter out invalid points (removing NaNs and infinite values)
    valid_points = points_3d[~np.isnan(points_3d).any(
        axis=1) & ~np.isinf(points_3d).any(axis=1)]
    print(f"Total valid 3D points: {len(valid_points)}")

    return valid_points


def create_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    if points.size > 0:
        point_cloud.points = o3d.utility.Vector3dVector(points)
        print("Creating point cloud with points:", points.shape)
    else:
        print("No points available to create point cloud.")
    return point_cloud


def visualize_point_cloud(point_cloud):
    # Remove outliers
    # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.02)
    # cl, ind = point_cloud.remove_statistical_outlier(
    #    nb_neighbors=20, std_ratio=2.0)
    point_cloud, ind = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)
    print("Attempting to visualize point cloud...")
    o3d.visualization.draw_geometries([point_cloud])


def main():
    camera_params_file = 'camera_parameters.csv'
    intrinsic_params_file = 'intrinsic_parameters.csv'
    feature_matches_file = 'feature_matches.csv'

    # Load camera parameters
    camera_params = load_camera_parameters(camera_params_file)

    # Load intrinsic parameters
    intrinsic_matrices, dist_coeffs = load_intrinsic_parameters(
        intrinsic_params_file)

    # Load feature matches
    matches, _ = load_feature_matches(feature_matches_file)

    # Triangulate points
    points = triangulate_points(
        matches, camera_params, intrinsic_matrices, dist_coeffs)

    # Create and visualize the point cloud
    point_cloud = create_point_cloud(points)
    visualize_point_cloud(point_cloud)


if __name__ == '__main__':
    main()
