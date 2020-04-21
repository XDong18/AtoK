"""The code to translate Argoverse dataset to KITTI dataset format"""
# Argoverse-to-KITTI Adapter
# Author: Yiyang Zhou 
# Email: yiyang.zhou@berkeley.edu

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
from filter_ground_pts import filter_ground_pts_polar_grid_mean_var
from aggregate import aggregate


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

####CONFIGURATION#################################################
# Root directory
ROOT_DIR= '/shared/xudongliu/data/argoverse-tracking/'

# Setup directories
data_dir = os.path.join(ROOT_DIR, args.data_split)
goal_dir = os.path.join(ROOT_DIR, 'argo_track_aggre5', args.data_split)
oxts_dir = os.path.join(goal_dir, 'oxts')
velo_dir = os.path.join(goal_dir, 'velodyne')
img_dir = os.path.join(goal_dir, 'image_02')
cali_dir = os.path.join(goal_dir, 'calib')
label_dir = os.path.join(goal_dir, 'label_02')
velo_reduce_dir = os.path.join(goal_dir, 'velodyne_reduced')
inst_seg_dir = os.path.join(goal_dir, 'instance_segment')
pc_dir = os.path.join(goal_dir, 'pc')
t_pc_dir = os.path.join(goal_dir, 't_pc')
if not os.path.exists(goal_dir):
    os.makedirs(goal_dir)
if not os.path.exists(oxts_dir):
    os.mkdir(oxts_dir)
if not os.path.exists(velo_dir):
    os.mkdir(velo_dir)
if not os.path.exists(img_dir):
    os.mkdir(img_dir)
if not os.path.exists(cali_dir):
    os.mkdir(cali_dir)
if not os.path.exists(label_dir):
    os.mkdir(label_dir)
if not os.path.exists(velo_reduce_dir):
    os.mkdir(velo_reduce_dir)
if not os.path.exists(inst_seg_dir):
    os.mkdir(inst_seg_dir)
if not os.path.exists(pc_dir):
    os.mkdir(pc_dir)
if not os.path.exists(t_pc_dir):
    os.mkdir(t_pc_dir)

# Camera keys
CAMERA_KEY=['ring_front_center',
 'ring_front_left',
 'ring_front_right',
 'ring_rear_left',
 'ring_rear_right',
 'ring_side_left',
 'ring_side_right']

CAMS = CAMERA_KEY[0:1]

# Maximum thresholding distance for labelled objects
# (Object beyond this distance will not be labelled)
im_w=1920
im_h=1200
color_dir = {'vehicle':(0,0,255), 'large_vehicle':(255,0,0), 'bus':(0,255,0), 'trailer':(255,255,255)}
####################################################################
"""
Your original file directory is:
argodataset
└── argoverse-tracking <----------------------------ROOT_DIR
    └── train <-------------------------------------data_dir
        └── 0ef28d5c-ae34-370b-99e7-6709e1c4b929
        └── 00c561b9-2057-358d-82c6-5b06d76cebcf
        └── ...
    └── validation
        └──5c251c22-11b2-3278-835c-0cf3cdee3f44
        └──...
    └── test
        └──8a15674a-ae5c-38e2-bc4b-f4156d384072
        └──...
        

"""

_PathLike = Union[str, "os.PathLike[str]"] # ???
def load_ply(ply_fpath: _PathLike) -> np.ndarray:
    """Load a point cloud file from a filepath.
    Args:
        ply_fpath: Path to a PLY file
    Returns:
        arr: Array of shape (N, 3)
    """

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]

    return np.concatenate((x, y, z), axis=1)

def tprint(string: str, verbose: bool=False):
    if verbose:
        tqdm.write(string)

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

def segment_lidar_idx_not_from_box_cube(
        lidar_data: np.ndarray, 
        corners_velo: np.ndarray,
        conf_range: float=0.0) -> np.ndarray:

    # Cam to Velo coordinates
    # Cam:  X->right; Y->down; Z->front
    # Velo: X->front; Y->left; Z->up
    # Refer to Figure 3 of Argoverse (Arxiv:1911.02620)

    # Get boundary value
    lower_bound = np.array([-1e10, -1e10, -1e10])
    upper_bound = np.array([1e10, 1e10, 1e10])

    # Scale the confidence range
    # bound_diff = upper_bound - lower_bound
    # upper_bound = (1 + conf_range) * bound_diff + lower_bound
    # lower_bound = (- conf_range) * bound_diff + lower_bound

    # Get point cloud in range
    selected_bool = (lidar_data > lower_bound) \
                  & (lidar_data < upper_bound)

    # Convert to index
    selected_idx = np.where(selected_bool.sum(1) == 3)

    return selected_idx


def get_color_from_tid(tid: int, classes) -> (int, int, int):
    # NOTE: the color is in RGB format
    #       and COCO treats (0, 0, 0) as ignore, tid cannot be 0
    tid += 1
    return color_dir[classes]

def get_tid_from_color(color: (int, int, int)) -> int:
    # NOTE: the color is in RGB format
    #       and COCO treats (0, 0, 0) as ignore, tid cannot be 0
    return int((color[0] << 16) + (color[1] << 8) + color[2]) - 1

def segment_tid_to_lidar_color(
        lidar_rgb: np.ndarray,
        lidar_idx: np.ndarray,
        tid: int,
        classes: str) -> np.ndarray:
    color = get_color_from_tid(tid, classes)
    lidar_rgb[lidar_idx, 0] = color[0]
    lidar_rgb[lidar_idx, 1] = color[1]
    lidar_rgb[lidar_idx, 2] = color[2] 
    return lidar_rgb

def get_tid_from_lidar_color(
        lidar_rgb_part: np.ndarray) -> int:
    median_rgb = np.median(lidar_rgb_part, axis=0)
    tid = get_tid_from_color(median_rgb)
    return tid

def project_segment_to_img(
        img: np.ndarray,
        lidar_rgb: np.ndarray,
        lidar_vuz: np.ndarray) -> np.ndarray:
    '''
    lidar_vuz should have the same dimension (v, u, z) as lidar_rgb (v, u, 3)
    '''
    valid_point = np.logical_and.reduce(lidar_vuz > 0, axis=1)
    valid_point &= np.logical_and.reduce(
            lidar_vuz[:, :2] < np.array([img.shape[0]-1, img.shape[1]-1]), axis=1)
    valid_idx = lidar_vuz[np.where(valid_point)].astype(int)
    assert valid_idx is not None, 'No point image projection'
    # for i, pos in enumerate(valid_idx):
    #     cv2.circle(img, (pos[1], pos[0]), 1, (lidar_rgb, pos[2], pos[2]), 4)
    img[valid_idx[:, 0], valid_idx[:, 1]] = lidar_rgb[valid_point]


    return img, valid_idx

def convert_file(input_path, output_path, verbose=False):
    tprint(f"convert from {input_path} into {output_path}", verbose)
    os.system(f"convert {input_path} {output_path}") 

def convert_file_mogrify(input_folder, output_folder, input_format='jpg', target_format='png', verbose=False):
    tprint(f"convert from {input_folder}/*.{input_format} into format {target_format} in {output_folder}", verbose)
    os.system(f"mogrify -path {output_folder} -format {target_format} {input_folder}/*.{input_format}")

def main():
    tprint('\nLoading files...', True)

    # Check the number of logs(one continuous trajectory)
    argoverse_loader= ArgoverseTrackingLoader(data_dir)
    dl= SimpleArgoverseTrackingDataLoader(data_dir=data_dir, labels_dir=data_dir)
    tprint('\nTotal number of logs: {}'.format(len(argoverse_loader)), True)
    argoverse_loader.print_all()
    tprint('\n')

    # count total number of files
    total_number = 0
    curr_number = 0
    for log_key in argoverse_loader.log_list:
        path, dirs, files = next(os.walk(os.path.join(data_dir, log_key, 'lidar')))
        total_number= total_number+len(files)
    total_number *= len(CAMS)

    tprint('Total number of files: {}. Translation starts...'.format(total_number), True)
    tprint('Progress:', True)

    for cam_id, cam_key in enumerate(CAMS):
        tprint(f"Camera: {cam_id} {cam_key}", True)
        for log_id, log_key in enumerate(tqdm(sorted(argoverse_loader.log_list))):
            argoverse_data = argoverse_loader.get(log_key)
            tprint(f"Log: {log_id} {log_key} {len(argoverse_data.lidar_timestamp_list)}", True)
            if args.dry_run:
                continue
            trk_id_dict = {}

            # Create the log folder
            if not os.path.exists(os.path.join(img_dir, str(log_id).zfill(4))):
                os.mkdir(os.path.join(img_dir, str(log_id).zfill(4)))
            if not os.path.exists(os.path.join(label_dir, str(log_id).zfill(4))):
                os.mkdir(os.path.join(label_dir, str(log_id).zfill(4)))
            if not os.path.exists(os.path.join(velo_dir, str(log_id).zfill(4))):
                os.mkdir(os.path.join(velo_dir, str(log_id).zfill(4)))
            if not os.path.exists(os.path.join(velo_reduce_dir, str(log_id).zfill(4))):
                os.mkdir(os.path.join(velo_reduce_dir, str(log_id).zfill(4)))
            if not os.path.exists(os.path.join(inst_seg_dir, str(log_id).zfill(4))):
                os.mkdir(os.path.join(inst_seg_dir, str(log_id).zfill(4)))
            if not os.path.exists(os.path.join(t_pc_dir, str(log_id).zfill(4))):
                os.mkdir(os.path.join(t_pc_dir, str(log_id).zfill(4)))


            # Recreate the calibration file content 
            with open(os.path.join(cali_dir, str(log_id).zfill(4) + '.txt'),'w+') as f:
                calibration_data = argoverse_data.get_calibration(cam_key)
                log_calib_data = dl.get_log_calibration_data(log_key)
                planes = generate_frustum_planes(
                            calibration_data.K.copy(), 
                            cam_key)
                L3='P2: '
                for j in calibration_data.K.reshape(1,12)[0]:
                    L3= L3+ str(j)+ ' '
                L3=L3[:-1]

                L6= 'Tr_velo_to_cam: '
                for k in calibration_data.extrinsic.reshape(1,16)[0][0:12]:
                    L6= L6+ str(k)+ ' '
                L6=L6[:-1]

                file_content='P0: 0 0 0 0 0 0 0 0 0 0 0 0\n' \
                        'P1: 0 0 0 0 0 0 0 0 0 0 0 0\n' \
                        '{}\n' \
                        'P3: 0 0 0 0 0 0 0 0 0 0 0 0\n' \
                        'R0_rect: 1 0 0 0 1 0 0 0 1\n' \
                        '{}\n' \
                        'Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0\n'.format(L3,L6)

                f.write(file_content)


            # Get Camera pose
            with open(os.path.join(oxts_dir, str(log_id).zfill(4) + '.txt'), 'w+') as fpose:

                # Loop through the each lidar frame (10Hz)
                # to copy and reconfigure all images, lidars, 
                # calibration files, and label files.
                all_list_len = len(argoverse_data.lidar_timestamp_list)  
                all_lidar_data_list = \
                    [filter_ground_pts_polar_grid_mean_var(argoverse_data.get_lidar(index)) for index in range(all_list_len)]
                
                for indx, timestamp in enumerate(argoverse_data.lidar_timestamp_list):

                    # Save lidar file into .bin format under the new directory 
                    # lidar_data = argoverse_data.get_lidar(indx)
                    # lidar_data = filter_ground_pts_polar_grid_mean_var(lidar_data)
                    '''
                    the begin of aggredate data
                    '''
                    num_ag = 5
                    if indx<int(num_ag/2) :
                        lidar_data_list = [all_lidar_data_list[indx]]
                        timestamp_list = [argoverse_data.lidar_timestamp_list[indx]]
                        for fram_idx in range(num_ag):
                            if fram_idx==indx:
                                continue
                            lidar_data_list.append(all_lidar_data_list[fram_idx])
                            timestamp_list.append(argoverse_data.lidar_timestamp_list[fram_idx])
                        # lidar_data_list = [argoverse_data.get_lidar(indx), argoverse_data.get_lidar(indx+1), argoverse_data.get_lidar(indx+2)]
                        # timestamp_list = [argoverse_data.lidar_timestamp_list[indx], argoverse_data.lidar_timestamp_list[indx+1], argoverse_data.lidar_timestamp_list[indx+2]]
                    elif indx>all_list_len-1-int(num_ag/2):
                        lidar_data_list = [all_lidar_data_list[indx]]
                        timestamp_list = [argoverse_data.lidar_timestamp_list[indx]]
                        for fram_idx in range(all_list_len-1-num_ag, all_list_len):
                            if fram_idx==indx:
                                continue
                            lidar_data_list.append(all_lidar_data_list[fram_idx])
                            timestamp_list.append(argoverse_data.lidar_timestamp_list[fram_idx])
                        # lidar_data_list = [argoverse_data.get_lidar(indx), argoverse_data.get_lidar(indx-1), argoverse_data.get_lidar(indx-2)]
                        # timestamp_list = [argoverse_data.lidar_timestamp_list[indx], argoverse_data.lidar_timestamp_list[indx-1], argoverse_data.lidar_timestamp_list[indx-2]]
                    else:
                        lidar_data_list = [all_lidar_data_list[indx]]
                        timestamp_list = [argoverse_data.lidar_timestamp_list[indx]]
                        for fram_idx in range(indx-int(num_ag/2), indx+int(num_ag/2)+1):
                            if fram_idx==indx:
                                continue
                            lidar_data_list.append(all_lidar_data_list[fram_idx])
                            timestamp_list.append(argoverse_data.lidar_timestamp_list[fram_idx])
                        # lidar_data_list = [argoverse_data.get_lidar(indx-1), argoverse_data.get_lidar(indx), argoverse_data.get_lidar(indx+1)]
                        # timestamp_list = [argoverse_data.lidar_timestamp_list[indx], argoverse_data.lidar_timestamp_list[indx-1], argoverse_data.lidar_timestamp_list[indx+1]]
                    
                    aggregated_lidar_data = aggregate(lidar_data_list, timestamp_list, data_dir, log_key)
                    '''
                    the end of aggredate data
                    '''
                    lidar_data = aggregated_lidar_data # the change of lidar data

                    target_lidar_file_path = os.path.join(velo_dir, str(log_id).zfill(4), str(indx).zfill(6) + '.bin')
                    aug_lidar_file_path = os.path.join(velo_reduce_dir, str(log_id).zfill(4), str(indx).zfill(6) + '.ply')
                    inst_seg_file_path = os.path.join(inst_seg_dir, str(log_id).zfill(4), str(indx).zfill(6) + '.png')

                    lidar_data_augmented = np.hstack([lidar_data, np.zeros([lidar_data.shape[0], 1])])
                    lidar_data_augmented = lidar_data_augmented.astype('float32')
                    lidar_data_augmented.tofile(target_lidar_file_path)

                    instance_img = np.zeros([im_h, im_w, 3])
                    # (x, y, z) to (u, v, z)
                    lidar_uvz = calibration_data.project_ego_to_image(lidar_data)
                    lidar_vuz = np.hstack([lidar_uvz[:, :2][:, ::-1], lidar_uvz[:, 2:]])
                    lidar_rgb = np.zeros(lidar_data.shape)
                    #print('test1')
                    lidar_xyzrgb = o3d.geometry.PointCloud()
                    lidar_xyzrgb.points = o3d.utility.Vector3dVector(lidar_data)

                    pc_lidar_rgb = np.stack((lidar_vuz[:,2], lidar_vuz[:,2], lidar_vuz[:,2]), axis = -1)
                    pc_img = np.zeros([im_h, im_w, 3])
                    pc_valid_point = np.logical_and.reduce(lidar_vuz > 0, axis=1)
                    pc_valid_point &= np.logical_and.reduce(
                            lidar_vuz[:, :2] < np.array([pc_img.shape[0]-1, pc_img.shape[1]-1]), axis=1)
                    pc_valid_idx = lidar_vuz[np.where(pc_valid_point)].astype(int)
                    pc_img[pc_valid_idx[:, 0], pc_valid_idx[:, 1]] = pc_lidar_rgb[pc_valid_point]
                    t_pc_file_path = os.path.join(t_pc_dir, str(log_id).zfill(4), str(indx).zfill(6) + '.png')
                    # print(t_pc_file_path, pc_img)
                    cv2.imwrite(t_pc_file_path, pc_img)

                    # for i, pos in enumerate(pc_valid_idx):
                    #     cv2.circle(pc_img, (pos[1], pos[0]), 1, (int(pos[2]), int(pos[2]), int(pos[2])), 4)
                    # pc_file_path = os.path.join(pc_dir, str(log_id).zfill(4), str(indx).zfill(6) + '.png')
                    # cv2.imwrite(pc_file_path, pc_img)

                    pose = argoverse_data.get_pose(indx)
                    if pose is None:
                        logger.warning("`pose` is missing at index %d", i)
                        continue
                    # NOTE: It should be GPS location and rotation vector etc.
                    rot, _ = cv2.Rodrigues(pose.rotation)
                    loc = pose.translation
                    locrot = np.hstack([loc, rot.flatten()])
                    oxts_path = ' '.join([str(item) for item in locrot]) + '\n'
                    fpose.write(oxts_path)

                    # Save the image file into .png format under the new directory 
                    cam_file_path = argoverse_data.image_list_sync[cam_key][indx]
                    target_cam_file_path = os.path.join(img_dir, str(log_id).zfill(4), str(indx).zfill(6) + '.png')
                    if args.vis: img = imageio.imread(cam_file_path)
                    #print('test2')
                    if not os.path.isfile(target_cam_file_path) or args.overwrite:
                        convert_file(cam_file_path, target_cam_file_path, args.verbose)
                    #print('test3')
                    label_object_list= argoverse_data.get_label_object(indx)
                    with open(os.path.join(label_dir, str(log_id).zfill(4), str(indx).zfill(6) + '.txt'),'w+') as f:
                        for detected_object in label_object_list:
    
                            # Filter target classes
                            classes= detected_object.label_class.lower()
                            if classes not in ['vehicle', 'large_vehicle', 'bus', 'trailer']:
                                continue

                            # Generate tracking id
                            if detected_object.track_id in trk_id_dict:
                                tid = trk_id_dict[detected_object.track_id]
                            else:
                                trk_id_dict[detected_object.track_id] = len(trk_id_dict)
                                tid = trk_id_dict[detected_object.track_id]
                            #tprint(tid, get_color_from_tid(tid), args.verbose)

                            occulusion= round(detected_object.occlusion/25)
                            height= detected_object.height
                            length= detected_object.length
                            width= detected_object.width
                            truncated = 0

                            # all points in ego frame 
                            center= detected_object.translation
                            corners_ego_frame=detected_object.as_3d_bbox()

                            # all points in the camera frame
                            center_cam_frame= calibration_data.project_ego_to_cam(np.array([center]))
                            corners_cam_frame= calibration_data.project_ego_to_cam(corners_ego_frame) 

                            points_h = point_cloud_to_homogeneous(corners_ego_frame).T
                            _, uv_cam, _, _ = project_lidar_to_undistorted_img(points_h, copy.deepcopy(log_calib_data), cam_key)
                            image_bbox = cuboid_to_2d_frustum_bbox(uv_cam.T, planes, calibration_data.K[:3,:3])
                            

                            if image_bbox is not None: 

                                # the center coordinates in cam frame we need for KITTI 
                                selected_lidar_idx = segment_lidar_idx_from_box_cube(lidar_data, corners_ego_frame)
                                lidar_rgb = segment_tid_to_lidar_color(lidar_rgb, selected_lidar_idx, tid, classes)
                                
                                #tid_obtained = get_tid_from_lidar_color(lidar_rgb[selected_lidar_idx])

                                x1,y1,x2,y2 = image_bbox

                                x1 = min(x1, im_w-1)
                                x2 = min(x2, im_w-1)
                                y1 = min(y1, im_h-1)
                                y2 = min(y2, im_h-1)

                                x1 = max(x1, 0)
                                x2 = max(x2, 0)
                                y1 = max(y1, 0)
                                y2 = max(y2, 0)

                                if args.vis:
                                    tprint(f"{log_id} - {indx}: {x1}, {y1}, {x2}, {y2}", args.verbose)
                                    plt.plot([x1,x1],[y1,y2], 'r')
                                    plt.plot([x1,x2],[y1,y1], 'r')
                                    plt.plot([x1,x2],[y2,y2], 'r')
                                    plt.plot([x2,x2],[y1,y2], 'r')

                                # for the orientation, we choose point 1 and point 5 for application 
                                p1= corners_cam_frame[1]
                                p5= corners_cam_frame[5]
                                dz=-(p1[2]-p5[2])
                                dx=p1[0]-p5[0]
                                # the orientation angle of the car
                                angle= math.atan2(dz,dx)
                                beta= math.atan2(center_cam_frame[0][0],center_cam_frame[0][2])
                                alpha = angle - beta
                                alpha = alpha % (2*math.pi) - math.pi
                                line= '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n'.format(
                                    indx, tid, classes, 
                                    round(truncated,2), occulusion, round(alpha,2), 
                                    round(x1,2), round(y1,2),
                                    round(x2,2), round(y2,2),
                                    round(height,2), round(width,2), round(length,2), 
                                    round(center_cam_frame[0][0],2), round(center_cam_frame[0][1],2),
                                    round(center_cam_frame[0][2],2), round(angle,2))                

                                f.write(line)
                                tprint(f"{log_id} ({log_key}) - {indx}: {classes} {image_bbox}, {center_cam_frame}", args.verbose)
                        if args.vis:
                            plt.imshow(img)
                            plt.show()
                        
                        curr_number += 1
                    if not os.path.isfile(inst_seg_file_path) or args.overwrite:
                        instance_img, valid_idx = project_segment_to_img(instance_img, lidar_rgb, lidar_vuz)
                        cv2.imwrite(inst_seg_file_path, instance_img)
                        
                    if not os.path.isfile(aug_lidar_file_path) or args.overwrite:
                        lidar_xyzrgb.colors = o3d.utility.Vector3dVector(lidar_rgb)
                        o3d.io.write_point_cloud(os.fspath(aug_lidar_file_path), lidar_xyzrgb)


    tprint(f'Translation finished, processed {curr_number} files', True)

if __name__ == '__main__':
    main()
