# reconstruction_visualization.py

import cv2
import numpy as np
import open3d as o3d
from camera_parameters import load_camera_parameters, load_feature_matches


def triangulate_points(matches, camera_params):
    # Initialize a list to store arrays of 3D points for each image pair
    all_points_3d = []

    # Convert rotation and translation vectors to camera matrices
    camera_matrices = {}
    for img, params in camera_params.items():
        rvec, tvec = params[0], params[1].reshape(
            3, 1)  # Correcting the reshaping
        # Ensuring correct shape for Rodrigues function
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        P = np.hstack((R, tvec))  # Camera matrix [R|t]
        camera_matrices[img] = P

    # Iterate over each pair of images and their corresponding matches
    for (img1, img2), match_pts in matches.items():
        if img1 not in camera_matrices or img2 not in camera_matrices:
            continue  # Skip if camera matrices for the images are not available

        # Extracting matched points
        pts1 = np.float32([pt[0:2] for pt in match_pts]).T
        pts2 = np.float32([pt[2:4] for pt in match_pts]).T

        # Get camera matrices for the pair of images
        P1 = camera_matrices[img1]
        P2 = camera_matrices[img2]

        # Triangulate points
        pts_4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
        # Convert to Euclidean coordinates
        # pts_3d_hom = pts_4d_hom / pts_4d_hom[3]
        pts_3d_hom = pts_4d_hom / np.array(pts_4d_hom[3])

        # Append these points to the list
        all_points_3d.append(pts_3d_hom[:3].T)

    # Aggregate all 3D points from each image pair
    if all_points_3d:
        points_3d = np.concatenate(all_points_3d, axis=0)
    else:
        points_3d = np.empty((0, 3))

    # Filter valid points (remove NaNs and infinites)
    valid_points = points_3d[~np.isnan(points_3d).any(
        axis=1) & ~np.isinf(points_3d).any(axis=1)]

    return valid_points


def create_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    if points.size > 0:
        point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


def visualize_point_cloud(point_cloud):
    # Remove outliers
    # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.02)
    # cl, ind = point_cloud.remove_statistical_outlier(
    #    nb_neighbors=20, std_ratio=2.0)
    point_cloud, ind = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)

    o3d.visualization.draw_geometries([point_cloud])


def main():
    camera_params_file = 'camera_parameters.csv'
    feature_matches_file = 'feature_matches.csv'

    # Load camera parameters
    camera_params = load_camera_parameters(camera_params_file)

    # Load feature matches and dimensions (if needed)
    matches, dimensions = load_feature_matches(
        feature_matches_file)  # Unpack the tuple correctly

    # Triangulate points
    points = triangulate_points(matches, camera_params)

    # Create and visualize the point cloud
    point_cloud = create_point_cloud(points)
    visualize_point_cloud(point_cloud)


if __name__ == '__main__':
    main()
