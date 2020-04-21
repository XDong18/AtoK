import os
import json
import math
import numpy as np
from typing import Union
from shutil import copyfile
import matplotlib.pyplot as plt
import imageio

import cv2
import copy
import argparse
import open3d as o3d
import pyntcloud
from tqdm import tqdm

import argoverse
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.calibration import (
    point_cloud_to_homogeneous,
    project_lidar_to_undistorted_img
)
from argoverse.utils.frustum_clipping import (
    generate_frustum_planes, 
    cuboid_to_2d_frustum_bbox
)
# from filter_ground_pts import filter_ground_pts_polar_grid_mean_var
from aggregate import aggregate
from os.path import join, split, exists
import json
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
from filter_ground_pts import filter_ground_pts_polar_grid_mean_var

def mkdir(path):
    if not exists(path):
        os.makedirs(path)

def segment_lidar_idx_from_box_cube(
        lidar_data: np.ndarray, 
        corners_velo: np.ndarray,
        conf_range: float=0.0) -> np.ndarray:

    # Cam to Velo coordinates
    # Cam:  X->right; Y->down; Z->front
    # Velo: X->front; Y->left; Z->up
    # Refer to Figure 3 of Argoverse (Arxiv:1911.02620)

    # Get boundary value
    lower_bound = np.min(corners_velo, axis=0)
    upper_bound = np.max(corners_velo, axis=0)

    # Scale the confidence range
    bound_diff = upper_bound - lower_bound
    upper_bound = (1 + conf_range) * bound_diff + lower_bound
    lower_bound = (- conf_range) * bound_diff + lower_bound

    # Get point cloud in range
    selected_bool = (lidar_data > lower_bound) \
                  & (lidar_data < upper_bound)

    # Convert to index
    selected_idx = np.where(selected_bool.sum(1) == 3)

    return selected_idx

def parse_args():
    parser = argparse.ArgumentParser(description='Convert data format from Argoverse to KITTI',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_split', choices=['train', 'val', 'test', 'sample'], 
                        help='Which data split to use in testing')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Show command without running')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='Show label')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite the output files')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print the output logs')
    
    args = parser.parse_args()

    return args

args = parse_args()

ROOT_DIR= '/shared/xudongliu/data/argoverse-tracking/'
data_dir = os.path.join(ROOT_DIR, args.data_split)
goal_dir = os.path.join(ROOT_DIR, 'argo_track', args.data_split)
obj_dir = os.path.join(goal_dir, 'obj_noground')
oid_dir = os.path.join(goal_dir, 'oid_noground')
trans_dir = os.path.join(goal_dir, 'trans_noground')
corners_dir = os.path.join(goal_dir, 'corners_noground') # TODO name '_noground"
if not os.path.exists(goal_dir):
    os.makedirs(goal_dir)

if not os.path.exists(obj_dir):
    os.makedirs(obj_dir)

mkdir(oid_dir)
mkdir(trans_dir)
mkdir(corners_dir)

CAMERA_KEY=['ring_front_center',
 'ring_front_left',
 'ring_front_right',
 'ring_rear_left',
 'ring_rear_right',
 'ring_side_left',
 'ring_side_right']

CAMS = CAMERA_KEY[0:1]

im_w=1920
im_h=1200

def tprint(string: str, verbose: bool=False):
    if verbose:
        tqdm.write(string)

def main():
    tprint('\nLoading files...', True)
    argoverse_loader= ArgoverseTrackingLoader(data_dir)
    dl= SimpleArgoverseTrackingDataLoader(data_dir=data_dir, labels_dir=data_dir)
    tprint('\nTotal number of logs: {}'.format(len(argoverse_loader)), True)
    argoverse_loader.print_all()
    tprint('\n')

    total_number = 0
    curr_number = 0
    for log_key in argoverse_loader.log_list:
        path, dirs, files = next(os.walk(os.path.join(data_dir, log_key, 'lidar')))
        total_number= total_number+len(files)

    total_number *= len(CAMS)
    # print(total_number)
    tprint('Total number of files: {}. Translation starts...'.format(total_number), True)
    tprint('Progress:', True)

    for cam_id, cam_key in enumerate(CAMS):
        tprint(f"Camera: {cam_id} {cam_key}", True)
        
        for log_id, log_key in enumerate(tqdm(sorted(argoverse_loader.log_list))):
            argoverse_data = argoverse_loader.get(log_key)
            calibration_data = argoverse_data.get_calibration(cam_key)
            tprint(f"Log: {log_id} {log_key} {len(argoverse_data.lidar_timestamp_list)}", True)
            if args.dry_run:
                continue

            trk_id_dict = {}
            class_dict = {}
            trans_target_dict = {}
            mkdir(os.path.join(obj_dir, str(log_id).zfill(4)))
            mkdir(join(oid_dir, str(log_id).zfill(4)))
            mkdir(join(trans_dir, str(log_id).zfill(4)))
            mkdir(join(corners_dir, str(log_id).zfill(4)))

            for indx, timestamp in enumerate(argoverse_data.lidar_timestamp_list):
                lidar_data = filter_ground_pts_polar_grid_mean_var(argoverse_data.get_lidar(indx)) #TODO no ground points
                label_object_list= argoverse_data.get_label_object(indx)

                for detected_object in label_object_list:
                    classes= detected_object.label_class.lower()

                    if detected_object.track_id in trk_id_dict:
                        tid = trk_id_dict[detected_object.track_id]
                    else:
                        trk_id_dict[detected_object.track_id] = len(trk_id_dict)
                        tid = trk_id_dict[detected_object.track_id]
                        class_dict[tid] = classes
                        trans_target_dict[tid] = SE3(rotation=quat2rotmat(detected_object.quaternion), translation=detected_object.translation)
                
                    sub_obj_dir = join(obj_dir, str(log_id).zfill(4), str(tid).zfill(6))
                    mkdir(sub_obj_dir)
                    sub_trans_dir = join(trans_dir, str(log_id).zfill(4), str(tid).zfill(6))
                    mkdir(sub_trans_dir)
                    sub_corners_dir = join(corners_dir, str(log_id).zfill(4), str(tid).zfill(6))
                    mkdir(sub_corners_dir)

                    obj_file_path = os.path.join(sub_obj_dir, str(indx).zfill(6) + '.pcd')
                    trans_file_path = os.path.join(sub_trans_dir, str(indx).zfill(6) + '.npy')
                    corners_file_path = os.path.join(sub_corners_dir, str(indx).zfill(6) + '.pcd')

                    center= detected_object.translation
                    corners_ego_frame=detected_object.as_3d_bbox()
                    selected_lidar_idx = segment_lidar_idx_from_box_cube(lidar_data, corners_ego_frame)
                    selected_lidar_data = lidar_data[selected_lidar_idx]
                    selected_lidar_data = calibration_data.project_ego_to_cam(selected_lidar_data) #TODO
                    lidar_rgb = np.zeros((len(selected_lidar_data), 3))
                    # selected_lidar_data = np.hstack([selected_lidar_data[:, :2][:, ::-1], selected_lidar_data[:, 2:]])

                    if len(selected_lidar_data)==0:
                        continue
                    
                    corners_cam_frame = calibration_data.project_ego_to_cam(corners_ego_frame) #TODO 
                    corners_rgb = np.zeros((len(corners_ego_frame), 3))
                    corners_rgb[:,0] = 1

                    # selected_lidar_data = np.concatenate((selected_lidar_data, corners_ego_frame), axis=0) #TODO ego-> camera
                    # lidar_rgb = np.concatenate((lidar_rgb, corners_rgb), axis=0)

                    lidar_xyz = o3d.geometry.PointCloud()
                    lidar_xyz.points = o3d.utility.Vector3dVector(selected_lidar_data)
                    lidar_xyz.colors = o3d.utility.Vector3dVector(lidar_rgb)
                    # if not os.path.isfile(obj_file_path) or args.overwrite:
                    o3d.io.write_point_cloud(os.fspath(obj_file_path), lidar_xyz)

                    corners_xyz = o3d.geometry.PointCloud()
                    corners_xyz.points = o3d.utility.Vector3dVector(corners_cam_frame) #TODO
                    corners_xyz.colors = o3d.utility.Vector3dVector(corners_rgb)
                    o3d.io.write_point_cloud(os.fspath(corners_file_path), corners_xyz)

                    curr_se3 = SE3(rotation=quat2rotmat(detected_object.quaternion), translation=detected_object.translation)
                    curr_trans_se3 = curr_se3.inverse().right_multiply_with_se3(trans_target_dict[tid])
                    np.save(trans_file_path, curr_trans_se3.transform_matrix)


                curr_number += 1

            with open(join(oid_dir, str(log_id).zfill(4), 'trk_id_dict.json'), 'w') as f:
                json.dump(trk_id_dict, f)
            with open(join(oid_dir, str(log_id).zfill(4), 'class_dict.json'), 'w') as f:
                json.dump(class_dict, f)

    tprint(f'Translation finished, processed {curr_number} files', True)

if __name__=='__main__':
    main()
