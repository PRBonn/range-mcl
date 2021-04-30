#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: compute the normal for each point in a point cloud

import open3d as o3d
from utils import *

try:
  from c_gen_normal_map import gen_normal_map
except:
  print("You are currently using python library, which could be slow.")
  print("If you want to use fast C library, please Export PYTHONPATH=<path-to-range-image-library>")
  from utils import gen_normal_map


def compute_normals_range(cloud, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
  """ compute normals for each point using range image-based method.
  """
  range_image, vertex_map = range_projection_o3d(cloud, fov_up, fov_down, proj_H, proj_W, max_range)
  normal_map = gen_normal_map(range_image, vertex_map, proj_H, proj_W)
  cloud.points = o3d.utility.Vector3dVector(vertex_map.reshape(proj_H * proj_W, 3))
  cloud.normals = o3d.utility.Vector3dVector(normal_map.reshape(proj_H * proj_W, 3))
  return cloud


def compute_normals_o3d(cloud, voxel_size=0.1, max_nn=100):
  """ compute normals for each point using open3d.
  """
  params = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10.0,
                                                max_nn=max_nn)
  cloud.estimate_normals(params)
  cloud.orient_normals_towards_camera_location()
  return cloud


def normal_test(scan_path, proj_H=64, proj_W=900, use_range=False):
  """ test the normal computations.
  """
  # load the point cloud
  curren_points = load_vertex(scan_path)

  # get rid of invalid points
  dist = np.linalg.norm(curren_points[:, :3], 2, axis=1)
  curren_points = curren_points[(dist < 50) & (dist > 2)]

  # convert into open3d format and preprocess the point cloud
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(curren_points[:, :3])
  
  # test normal computation
  if use_range:
    cloud = compute_normals_range(pcd, proj_H=proj_H, proj_W=proj_W)
  else:
    cloud = compute_normals_o3d(pcd)
  
  # visualization
  o3d.visualization.draw_geometries([cloud])


if __name__ == '__main__':
  scan_path = '/path/to/scan.bin'
  normal_test(scan_path)
