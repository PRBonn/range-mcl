#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the main file for range-image-based Monte Carlo localization.

import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from map_module import MapModule
from motion_model import motion_model, gen_commands
from resample_module import resample
from sensor_model import SensorModel
from initialization import gen_coords_given_poses, init_particles_given_coords, init_particles_pose_tracking
from utils import load_poses_kitti

from visualizer import Visualizer
from vis_loc_result import plot_traj_result, save_loc_result


if __name__ == '__main__':
  # load config file
  config_filename = '../config/localization.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  config = yaml.safe_load(open(config_filename))
  
  # load parameters
  start_idx = config['start_idx']
  grid_res = config['grid_res']
  numParticles = config['numParticles']
  reduced_num = numParticles
  visualize = config['visualize']
  result_path = config['result_path']
  save_result = config['save_result']
  
  # load input data
  scan_folder = config['scan_folder']
  map_file = config['map_file']
  map_pose_file = config['map_pose_file']
  map_calib_file = config['map_calib_file']
  pose_file = config['pose_file']
  calib_file = config['calib_file']
  
  # load poses
  map_poses = load_poses_kitti(map_pose_file, map_calib_file)
  poses = load_poses_kitti(pose_file, calib_file)
  
  # initialize mesh map module
  print('Load mesh map and initialize map module...')
  map_module = MapModule(map_poses, map_file)
  
  # initialize particles
  print('Monte Carlo localization initializing...')
  map_size, road_coords = gen_coords_given_poses(map_poses)
  if config['pose_tracking']:
    particles = init_particles_pose_tracking(numParticles, poses[start_idx])
  else:
    particles = init_particles_given_coords(numParticles, road_coords)
  
  # initialize sensor model
  sensor_model = SensorModel(map_module, scan_folder, config['range_image'])
  if config['range_image']['render_instanced']:
    update_weights = sensor_model.update_weights_instanced
  else:
    update_weights = sensor_model.update_weights
  
  # generate odom commands
  commands = gen_commands(poses)
  
  # initialize a visualizer
  if visualize:
    plt.ion()
    visualizer = Visualizer(map_size, poses, map_poses, grid_res=grid_res, strat_idx=start_idx)
  else:  # store intermediate plots in a local folder
    local_folder = 'loc_plots'
    if not os.path.exists(local_folder):
      os.mkdir(local_folder)
    est_poses = []
  
  # Starts range-mcl
  is_initial = True
  results = np.full((len(poses), numParticles, 4), 0, np.float32)
  time_counter = []
  for frame_idx in range(start_idx, len(poses)):
    if visualize:
      visualizer.update(frame_idx, particles)
      visualizer.fig.canvas.draw()
      visualizer.fig.canvas.flush_events()
    else:
      save_loc_result(frame_idx, map_size, poses, particles, est_poses, local_folder)
    
    start = time.time()
    
    # motion model
    particles = motion_model(particles, commands[frame_idx])
    
    # only update the weight when the car moves
    if commands[frame_idx, 1] > 0.2 or is_initial:
      is_initial = False
      
      # range-image-based sensor model
      particles, reduced_num = update_weights(particles, frame_idx)
      
      # resampling
      particles = resample(particles)
    
    cost_time = np.round(time.time() - start, 10)
    print('finished frame ' + str(frame_idx) + ' with time of: ', cost_time, 's')
    if sensor_model.is_converged:
      time_counter.append(cost_time)
    results[frame_idx, :reduced_num] = particles
  
  print('Average runtime after convergence:', np.mean(time_counter))
  
  # saving the localization results
  if save_result:
    if not os.path.exists(os.path.dirname(result_path)):
      os.mkdir(os.path.dirname(result_path))
    np.savez_compressed(result_path, particles=results, start_idx=start_idx, numParticles=numParticles)
    offset = 20  # add a small offset to avoid showing the meaning less estimations before convergence
    plot_traj_result(results, poses, grid_res=grid_res, numParticles=numParticles, start_idx=start_idx+offset)
    print('save the localization results at:', result_path)
