#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script plots the final localization results and can visualize offline given the results.

import os
import sys
import yaml
import utils
import numpy as np
import matplotlib.pyplot as plt
from visualizer import Visualizer


def plot_traj_result(results, poses, numParticles=1000, grid_res=0.2, start_idx=0,
                     ratio=0.8, converge_thres=5, eva_thres=100):
  """ Plot the final localization trajectory.
    Args:
      results: localization results including particles in every timestamp.
      poses: ground truth poses.
      numParticles: number of particles.
      grid_res: the resolution of the grids.
      start_idx: the start index.
      ratio: the ratio of particles used to estimate the poes.
      converge_thres: a threshold used to tell whether the localization converged or not.
      eva_thres: a threshold to check the estimation results.
  """
  # get ground truth xy and yaw separately
  gt_location = poses[start_idx:, :2, 3]
  gt_heading = []
  for pose in poses:
    gt_heading.append(utils.euler_angles_from_rotation_matrix(pose[:3, :3])[2])
  gt_heading = np.array(gt_heading)[start_idx:]
  
  estimated_traj = []
  
  for frame_idx in range(start_idx, len(poses)):
    particles = results[frame_idx]
    # collect top 80% of particles to estimate pose
    idxes = np.argsort(particles[:, 3])[::-1]
    idxes = idxes[:int(ratio * numParticles)]
    
    partial_particles = particles[idxes]
    
    if np.sum(partial_particles[:, 3]) == 0:
      continue
      
    normalized_weight = partial_particles[:, 3] / np.sum(partial_particles[:, 3])

    estimated_traj.append(partial_particles[:, :3].T.dot(normalized_weight.T))

  estimated_traj = np.array(estimated_traj)
  
  # generate statistics for location (x, y)
  diffs_xy = np.array(estimated_traj[:, :2] * grid_res - gt_location)
  diffs_dist = np.linalg.norm(diffs_xy, axis=1)  # diff in euclidean

  # generate statistics for yaw
  diffs_heading = np.minimum(abs(estimated_traj[:, 2] - gt_heading),
                             2. * np.pi - abs(estimated_traj[:, 2] - gt_heading)) * 180. / np.pi

  # check if every eva_thres success converged
  if len(diffs_dist) > eva_thres and np.all(diffs_dist[eva_thres::eva_thres] < converge_thres):
    # calculate location error
    diffs_location = diffs_dist[eva_thres:]
    mean_location = np.mean(diffs_location)
    mean_square_error = np.mean(diffs_location * diffs_location)
    rmse_location = np.sqrt(mean_square_error)

    mean_heading = np.mean(diffs_heading)
    mean_square_error_heading = np.mean(diffs_heading * diffs_heading)
    rmse_heading = np.sqrt(mean_square_error_heading)

    # print('rmse_location: ', rmse_location)
    # print('rmse_heading: ', rmse_heading)
  
  # plot results
  fig = plt.figure(figsize=(16, 10))
  ax = fig.add_subplot(111)

  ax.plot(poses[:, 0, 3], poses[:, 1, 3], c='r', label='ground_truth')
  ax.plot(estimated_traj[:, 0] * grid_res, estimated_traj[:, 1] * grid_res, label='weighted_mean_80%')
  plt.show()


def vis_offline(results, poses, map_poses, mapsize, numParticles=1000, grid_res=0.2, start_idx=0):
  """ Visualize localization results offline.
    Args:
      results: localization results including particles in every timestamp.
      poses: ground truth poses.
      map_poses: poses used to generate the map.
      mapsize: size of the map.
      numParticles: number of particles.
      grid_res: the resolution of the grids.
      start_idx: the start index.
  """
  plt.ion()
  visualizer = Visualizer(mapsize, poses, map_poses,
                          numParticles=numParticles,
                          grid_res=grid_res, strat_idx=start_idx)
  for frame_idx in range(start_idx, len(poses)):
    particles = results[frame_idx]
    visualizer.update(frame_idx, particles)
    visualizer.fig.canvas.draw()
    visualizer.fig.canvas.flush_events()


def save_loc_result(frame_idx, map_size, poses, particles, est_poses, results_folder):
  """ Save the intermediate plots of localization results.
    Args:
      frame_idx: index of the current frame.
      map_size: size of the map.
      poses: ground truth poses.
      particles: current particles.
      est_poses: pose estimates.
      results_folder: folder to store the plots
  """
  # collect top 80% of particles to estimate pose
  idxes = np.argsort(particles[:, 3])[::-1]
  idxes = idxes[:int(0.8 * len(particles))]

  partial_particles = particles[idxes]

  normalized_weight = partial_particles[:, 3] / np.sum(partial_particles[:, 3])
  estimated_xy = partial_particles[:, :2].T.dot(normalized_weight.T)
  est_poses.append(estimated_xy)
  est_traj = np.array(est_poses)
  # plot results
  fig = plt.figure(figsize=(16, 10))
  ax = fig.add_subplot(111)

  ax.scatter(particles[:, 0], particles[:, 1], c=particles[:, 3], cmap='Blues', s=1)
  ax.plot(poses[:frame_idx + 1, 0, 3], poses[:frame_idx + 1, 1, 3], c='r', label='ground_truth')
  ax.plot(est_traj[:, 0], est_traj[:, 1], label='weighted_mean_80%')

  ax.axis('square')
  ax.set(xlim=map_size[:2], ylim=map_size[2:])

  plt.xlabel('x [m]')
  plt.ylabel('y [m]')
  plt.legend(loc='upper right')
  plt.savefig(os.path.join(results_folder, str(frame_idx).zfill(6)))
  plt.close()


if __name__ == '__main__':
  pass