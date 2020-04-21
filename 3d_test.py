import open3d as o3d
import numpy as np
import copy
import os
from os.path import join, split, exists
import copy


fn = "/shared/xudongliu/data/argoverse-tracking/argo_track/sample/obj/0000/000000/000000.pcd"
fn2 = "/shared/xudongliu/data/argoverse-tracking/argo_track/sample/obj/0000/000000/000020.pcd"
ptc =  o3d.io.read_point_cloud(fn)
ptc2 =  o3d.io.read_point_cloud(fn2)
# print(len(np.asarray(ptc2.points)))
new_points = np.asarray(ptc.points)[:len(np.asarray(ptc2.points))]
new_ptc = o3d.geometry.PointCloud()
new_ptc.points = o3d.utility.Vector3dVector(new_points)


threshold = 1
trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
evaluation = o3d.registration.evaluate_registration(ptc2, new_ptc, threshold, trans_init)
print(evaluation)
reg_p2p = o3d.registration.registration_icp(
        ptc2, new_ptc, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print(reg_p2p.transformation)
