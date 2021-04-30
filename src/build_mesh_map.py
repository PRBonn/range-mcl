#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script can be used to create mesh maps using LiDAR scans with GT poses.

import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import tqdm
from copy import deepcopy

from utils import load_files, load_poses, load_calib, load_vertex
from map_building.simplify_ground_mesh import pcd_ground_seg_open3d, mesh_simplify
from map_building.compute_normals import compute_normals_range


def preprocess_cloud(pcd, voxel_size=0.1,
                     crop_x=30, crop_y=30, crop_z=5,
                     downsample=False):
  """ preprocess the point cloud, including downsampling and cropping.
  """
  # downsample the point cloud if needed
  cloud = pcd.voxel_down_sample(voxel_size) if downsample else deepcopy(pcd)
  
  # crop point cloud with a box
  bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-crop_x, -crop_y, -crop_z),
                                             max_bound=(+crop_x, +crop_y, +crop_z))
  
  return cloud.crop(bbox)


def run_poisson(pcd, depth, min_density):
  """ run Poisson reconstruction on a local point cloud to get a local mesh.
  """
  if not pcd.has_normals():
    print("PointCloud doesn't have normals")
  o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
  mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=depth)
  
  # Post-process the mesh
  if min_density:
    vertices_to_remove = densities < np.quantile(densities, min_density)
    mesh.remove_vertices_by_mask(vertices_to_remove)
  
  # Return mesh
  mesh.compute_vertex_normals()
  return mesh


def main(config):
  """ This script can be used to create mesh maps using LiDAR scans with GT poses.
  It assumes you have the data in the kitti-like format like:

  data
  └── sequences
      └── 00
          ├── calib.txt
          ├── poses.txt
          └── velodyne
              ├── 000000.bin
              ├── 000001.bin
              └── ...

  How to run it and check a quick example:
  $ ./build_gt_map.py /path/to/config.yaml
  """
  # load scans and poses
  scan_folder = config['scan_folder']
  scan_paths = load_files(scan_folder)

  # load poses
  pose_file = config['pose_file']
  poses =load_poses(pose_file)
  inv_frame0 = np.linalg.inv(poses[0])

  # load calibrations
  # Note that if your poses are already in the LiDAR coordinate system, you
  # just need to set T_cam_velo as a 4x4 identity matrix
  calib_file = config['calib_file']
  T_cam_velo = load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)

  # convert poses into LiDAR coordinate system
  new_poses = []
  for pose in poses:
    new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
  new_poses = np.array(new_poses)
  gt_poses = new_poses
  
  # Use the whole sequence if -1 is specified
  n_scans = len(scan_paths) if config['n_scans'] == -1 else config['n_scans']
  
  # init mesh map
  mesh_file = config['mesh_file']
  if os.path.exists(mesh_file):
    exit(print('The mesh map already exists at:', mesh_file))
  global_mesh = o3d.geometry.TriangleMesh()
  cloud_map = o3d.geometry.PointCloud()
  
  # counter for local map
  count = 1
  local_map_size = config['local_map_size']
  
  # config for range images
  range_config = config['range_image']

  for idx in tqdm(range(n_scans)):
    # load the point cloud
    curren_points = load_vertex(scan_paths[idx])
    
    # get rid of invalid points
    dist = np.linalg.norm(curren_points[:, :3], 2, axis=1)
    curren_points = curren_points[(dist < range_config['max_range']) & (dist > range_config['min_range'])]
    
    # convert into open3d format and preprocess the point cloud
    local_cloud = o3d.geometry.PointCloud()
    local_cloud.points = o3d.utility.Vector3dVector(curren_points[:, :3])
    
    # estimated normals
    local_cloud = compute_normals_range(local_cloud, range_config['fov_up'], range_config['fov_down'],
                                  range_config['height'], range_config['width'], range_config['max_range'])
    
    # preprocess point clouds
    local_cloud = preprocess_cloud(local_cloud, config['voxel_size'],
                                   config['crop_x'], config['crop_y'], config['crop_z'],
                                   downsample=True)
    
    # integrate the local point cloud
    local_cloud.transform(gt_poses[idx])
    cloud_map += local_cloud
    
    if idx > 0:
      # if the car stops, we don't count the frame
      relative_pose = np.linalg.inv(gt_poses[idx - 1]).dot(gt_poses[idx])
      traj_dist = np.linalg.norm(relative_pose[:3, 3])
      if traj_dist > 0.2:
        count += 1
    
      # build a local mesh map
      if count % local_map_size == 0:
        # segment the ground
        ground, rest = pcd_ground_seg_open3d(cloud_map, config)
        
        # build the local poisson mesh
        mesh = run_poisson(ground+rest, depth=config['depth'], min_density=config['min_density'])
        
        # simply the ground to save space
        mesh = mesh_simplify(mesh, config)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # integrate the local mesh into global mesh
        global_mesh += mesh
        
        # re-init cloud map
        cloud_map = o3d.geometry.PointCloud()
    
  # save the mesh map
  print("Saving mesh to " + mesh_file)
  o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
  o3d.io.write_triangle_mesh(mesh_file, global_mesh)

  # visualize the mesh map
  if config['visualize']:
    o3d.visualization.draw_geometries([global_mesh])


if __name__ == "__main__":
  # load config file
  config_filename = '../config/build_map.yml'
  
  if yaml.__version__>='5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
  main(config)
