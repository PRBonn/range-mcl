#!/usr/bin/env python3
# Brief: This script evaluates the localization results

import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from utils import load_poses, load_calib, euler_angles_from_rotation_matrix


def get_estimates(particles, start_idx, particle_select_ratio=0.8):
  """ Generate the estimates given the particles """
  estimates = np.zeros((len(particles), 3))
  
  # collect top 80% of particles to estimate pose
  for idx in range(start_idx, len(particles)):
    indexes = np.argsort(particles[idx, :, 3])[::-1]
    indexes = indexes[:int(particle_select_ratio * len(indexes))]
    selected_particles = particles[idx, indexes]
    
    # normalise weights
    normalized_weight = selected_particles[:, 3] / np.sum(selected_particles[:, 3])
    
    # compute estimated x, y and yaw
    estimates[idx] = selected_particles[:, :3].T.dot(normalized_weight.T)
  
  return estimates


if __name__ == '__main__':
  # load config file
  config_filename = '../config/evaluation.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  config = yaml.safe_load(open(config_filename))

  # load setups
  plot_loc_traj = config['plot_loc_traj']
  save_evaluation_results = config['save_evaluation_results']
  
  # load ground truth files
  pose_file = config['pose_file']
  calib_file = config['calib_file']

  # load ground truth poses
  poses = np.array(load_poses(pose_file))
  inv_frame0 = np.linalg.inv(poses[0])

  # load calibrations
  T_cam_velo = load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)

  # convert poses in LiDAR coordinate system
  new_poses = []
  for pose in poses:
    new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
  new_poses = np.array(new_poses)
  gt_poses = new_poses

  gt_xy_raw = gt_poses[:, :2, 3]
  gt_yaw = []

  for pose in gt_poses:
    gt_yaw.append(euler_angles_from_rotation_matrix(pose[:3, :3])[2])
  gt_yaw_raw = np.array(gt_yaw)

  # load localization result
  result_file = config['result_file']
  results = np.load(result_file)
  
  # load parameters
  grid_resolution = config['grid_resolution']  # meters
  converge_thres = config['converge_thres']  # meters
  particle_select_ratio = config['particle_select_ratio']  # use the top 80 percent to estimate the pose
  interval = config['convergence_interval']  # use the top 80 percent to estimate the pose

  # init evaluation results
  success_converge = False
  rmse_location = -1
  rmse_yaw = -1

  particles = results['particles']
  start_idx = results['start_idx']
  numParticles = results['start_idx']

  # load result
  if 'estimates' in results.files and len(results['estimates']) > 0:
    estimates = results['estimates']
  else:
    estimates = get_estimates(particles, start_idx, particle_select_ratio)

  # check if the evaluation was done already
  if 'success_converge' in results.files:
    print('This result was evaluated already.')
    print('success_converge: ', results['success_converge'])
    print('rmse_location: ', results['rmse_location'])
    print('rmse_yaw: ', results['rmse_yaw'])

  else:
    gt_xy = gt_xy_raw[start_idx:]
    gt_yaw = gt_yaw_raw[start_idx:]
    
    # generate statistics for location (x, y)
    estimate_xy = estimates[start_idx:, :2] * grid_resolution
    diffs_xy = np.array(estimate_xy * grid_resolution - gt_xy[:])
    diffs_dist = np.linalg.norm(diffs_xy, axis=1)  # diff in euclidean

    # generate statistics for yaw
    diffs_yaw = np.minimum(abs(estimates[start_idx:, 2] - gt_yaw[start_idx:]),
                           abs(2. * np.pi - abs(estimates[start_idx:, 2] - gt_yaw[start_idx:]))) * 180. / np.pi
    
    # check if every interval success converged
    if np.all(diffs_dist[start_idx+interval::interval] < converge_thres):
      success_converge = True
      
      # calculate rmse for location
      diffs_xy_interval = diffs_dist[start_idx+interval:]
      mean_location = np.mean(diffs_xy_interval)
      mean_square_error = np.mean(diffs_xy_interval * diffs_xy_interval)
      rmse_location = np.sqrt(mean_square_error)

      # calculate rmse for yaw
      diffs_yaw_interval = diffs_yaw[interval:]
      mean_yaw = np.mean(diffs_yaw_interval)
      mean_square_error_yaw = np.mean(diffs_yaw_interval * diffs_yaw_interval)
      rmse_yaw = np.sqrt(mean_square_error_yaw)
      
      print('finished: ', result_file)
  
    print('success_converge: ', success_converge)
    print('rmse_location: ', rmse_location)
    print('rmse_yaw: ', rmse_yaw)

    if save_evaluation_results:
      np.savez_compressed(result_file.replace('.npz', '_updated.npz'),
                          particles=particles,
                          start_idx=start_idx,
                          numParticles=numParticles,
                          estimates=estimates,
                          success_converge=success_converge,
                          rmse_location=rmse_location,
                          rmse_yaw=rmse_yaw)

  if plot_loc_traj:
    offset = 20  # a small offset avoid showing the non-converged part
    plt.plot(gt_xy_raw[start_idx+offset:, 0], gt_xy_raw[start_idx+offset:, 1], 'r', label='gt')
    plt.plot(estimates[start_idx+offset:, 0], estimates[start_idx+offset:, 1], 'b', label='estimates')
    plt.legend()
    plt.axis('equal')
    plt.show()

