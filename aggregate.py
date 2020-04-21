import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import imageio
import numpy as np

from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
from argoverse.utils.camera_stats import (
    CAMERA_LIST,
    RING_CAMERA_LIST,
    RING_IMG_HEIGHT,
    RING_IMG_WIDTH,
    STEREO_CAMERA_LIST,
    STEREO_IMG_HEIGHT,
    STEREO_IMG_WIDTH,
)
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
from typing_extensions import Literal


def aggregate(lidar_pts_list, timestamp_list, dataset_dir, log_id):
    # print(timestamp_list[0], dataset_dir, log_id)
    aggregate_pts = lidar_pts_list[0]
    target_city_ego_se3 = get_city_SE3_egovehicle_at_sensor_t(timestamp_list[0], dataset_dir, log_id)
    for idx, lidar_pts in enumerate(lidar_pts_list[1:]):
        city_ego_se3 = get_city_SE3_egovehicle_at_sensor_t(timestamp_list[idx], dataset_dir, log_id)
        trans_se3 = city_ego_se3.inverse().right_multiply_with_se3(target_city_ego_se3)
        transformed_pts = trans_se3.transform_point_cloud(lidar_pts)
        aggregate_pts = np.concatenate((aggregate_pts, transformed_pts), axis=0)

    return aggregate_pts
