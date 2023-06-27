#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: some functions for MCL initialization

import numpy as np
import open3d as o3d
from utils import euler_angles_from_rotation_matrix

np.random.seed(0)


def init_particles_uniform(map_size, numParticles):
  """ Initialize particles uniformly.
    Args:
      map_size: size of the map.
      numParticles: number of particles.
    Return:
      particles.
  """
  [x_min, x_max, y_min, y_max] = map_size
  particles = []
  rand = np.random.rand
  for i in range(numParticles):
    x = (x_max - x_min) * rand(1) + x_min
    y = (y_max - y_min) * rand(1) + y_min
    # theta = 2 * np.pi * rand(1)
    theta = -np.pi + 2 * np.pi * rand(1)
    weight = 1
    particles.append([x, y, theta, weight])
  
  return np.array(particles)


def gen_coords_given_poses(poses, resolution=0.2, submap_size=2):
  """ Generate the road coordinates given the map poses.
    Args:
      poses: poses used to build the map.
      resolution: size of the grids for the initialization.
      submap_size: size of the submap for the initialization.
    Return:
      coords: coordinates of road grids for initialize particles.
  """
  submap_coords = []
  for x_coord in np.arange(-submap_size, submap_size, resolution):
    for y_coord in np.arange(-submap_size, submap_size, resolution):
      submap_coords.append([x_coord, y_coord])
  
  coords = []
  for pose in poses:
    center = pose[:2, 3]
    coords.append(submap_coords + center)
  
  coords = np.array(coords).reshape(-1, 2)
  coords_3d = np.zeros((coords.shape[0], coords.shape[1] + 1))
  coords_3d[:, :2] = coords
  
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(coords_3d)
  downpcd = pcd.voxel_down_sample(voxel_size=resolution).points
  coords = np.array(downpcd)[:, :2]
  
  min_x = int(np.round(np.min(coords[:, 0])))
  max_x = int(np.round(np.max(coords[:, 0])))
  min_y = int(np.round(np.min(coords[:, 1])))
  max_y = int(np.round(np.max(coords[:, 1])))
  
  return [min_x, max_x, min_y, max_y], coords


def init_particles_given_coords(numParticles, coords, init_weight=1.0):
  """ Initialize particles uniformly given the road coordinates.
    Args:
      numParticles: number of particles.
      coords: road coordinates.
      init_weight: initialization weight.
    Return:
      particles.
  """
  particles = []
  rand = np.random.rand
  args_coords = np.arange(len(coords))
  selected_args = np.random.choice(args_coords, numParticles)
  
  for i in range(numParticles):
    x = coords[selected_args[i]][0]
    y = coords[selected_args[i]][1]
    # theta = 2 * np.pi * rand(1)
    theta = -np.pi + 2 * np.pi * rand(1)
    particles.append([x, y, theta, init_weight])
  
  return np.array(particles, dtype=float)


def init_particles_pose_tracking(numParticles, init_pose, noises=[10.0, 10.0, np.pi/3.0], init_weight=1.0):
  """ Initialize particles with a noisy initial pose.
    Here, we use ground truth pose with noises defaulted as [±5 meters, ±5 meters, ±π/6 rad]
    to mimic a non-accurate GPS information as a coarse initial guess of the global pose.
    Args:
      numParticles: number of particles.
      init_pose: initial pose.
      noises: range of noises.
      init_weight: initialization weight.
    Return:
      particles.
  """
  particles = []
  rand = np.random.rand
  init_x = init_pose[0, 3]
  init_y = init_pose[1, 3]
  init_yaw = euler_angles_from_rotation_matrix(init_pose[:3, :3])[2]
  
  for i in range(numParticles):
    x = init_x + noises[0] * rand(1) - noises[0] / 2
    y = init_y + noises[1] * rand(1) - noises[1] / 2
    theta = init_yaw + noises[2] * rand(1) - noises[2] / 2
    particles.append([x, y, theta, init_weight])
  
  return np.array(particles, dtype=float)


