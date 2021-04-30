#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the sensor model for correlation-based Monte Carlo localization.

import os
import numpy as np
import OpenGL.GL as gl
import matplotlib.pyplot as plt
from map_renderer import MapRenderer, MapRenderer_instanced
from utils import load_files, load_vertex, range_projection, rotation_matrix_from_euler_angles


class SensorModel():
  """
  Brief: This class is the implementation of using correlation of range images as the sensor model for
         localization. In this sensor model we discretize the environment and generate a virtual frame for each grid
         after discretization. We estimate the similarity between the current frame and the grid virtual frames using
         the correlation.
  Initialization Input:
      mapsize: The size of the given map
      grid_coords: coordinates of virtual frames
      depth_image_paths: paths of range images
      grid_res: The resolution of the grids, default as 0.2 meter
  """
  
  def __init__(self, map_module, scan_folder, params):
    # load the map module.
    self.map_module = map_module
    
    # initialize the map renderer with the appropriate parameter.
    self.params = params
    self.max_instance = params['max_instance']
    if params['render_instanced']:
      self.renderer = MapRenderer_instanced(self.params)
    else:
      self.renderer = MapRenderer(self.params)
    self.renderer.set_mesh(self.map_module.mesh)
    
    # specify query scan paths
    self.scan_paths = load_files(scan_folder)
    
    self.is_converged = False
  
  def update_weights(self, particles, frame_idx):
    """ This function update the weight for each particle using the difference
    between current range image and the synthetic rendering for each particle.
    Old version where we render range image for each particle individually
    To use old version one need to import MapRenderer from renderer.py
    Input:
        particles: each particle has four properties [x, y, theta, weight]
        frame_idx: the index of the current frame
    Output:
        particles ... same particles with updated particles(i).weight
    """
    # load current scan and compute the histogram
    current_path = self.scan_paths[frame_idx]
    current_vertex = load_vertex(current_path)
    current_range, _, _, _ = range_projection(current_vertex,
                                              fov_up=self.params["fov_up"],
                                              fov_down=self.params["fov_down"],
                                              proj_H=self.params["height"],
                                              proj_W=self.params["width"],
                                              max_range=self.params["max_range"])
    # self.save_depth_image('current_frame', current_range, frame_idx, 0)
    
    scores = np.ones(len(particles)) * 0.00001
    
    tiles_collection = []
    
    for idx in range(len(particles)):
      particle = particles[idx]
      
      # first check whether the particle is inside the map or not
      if particle[0] < self.map_module.map_boundaries[0] or \
          particle[0] > self.map_module.map_boundaries[1] or \
          particle[1] < self.map_module.map_boundaries[2] or \
          particle[1] > self.map_module.map_boundaries[3]:
        continue
      
      # get tile index given particle position
      tile_idx = self.map_module.get_tile_idx([particle[0], particle[1]])
      if not self.map_module.tiles[tile_idx].valid:
        continue
      if tile_idx not in tiles_collection:
        tiles_collection.append(tile_idx)
      
      # get tile vertices start point and size
      start = self.map_module.tiles[tile_idx].vertices_buffer_start
      size = self.map_module.tiles[tile_idx].vertices_buffer_size
      
      # particle pose
      particle_pose = np.identity(4)  # init
      particle_pose[0, 3] = particle[0]  # particle[0]
      particle_pose[1, 3] = particle[1]  # particle[1]
      particle_pose[2, 3] = self.map_module.tiles[tile_idx].z  # use tile z
      particle_pose[:3, :3] = rotation_matrix_from_euler_angles(particle[2], degrees=False)  # rotation
      
      # generate synthetic range image
      self.renderer.render_with_tile(particle_pose, start, size)
      particle_depth = self.renderer.get_depth_map()
      
      # update the weight
      diff = abs(particle_depth - current_range)
      scores[idx] = np.exp(-0.5 * np.mean(diff[current_range > 0]) ** 2 / (2.0 ** 2))
    
    # normalization
    particles[:, 3] = particles[:, 3] * scores
    particles[:, 3] = particles[:, 3] / np.max(particles[:, 3])
    
    # check convergence using supporting tile map idea
    if len(tiles_collection) < 2 and not self.is_converged:
      self.is_converged = True
      print('Converged!')
      # cutoff redundant particles and leave only num of particles
      idxes = np.argsort(particles[:, 3])[::-1]
      particles = particles[idxes[:100]]
    
    return particles, len(particles)
  
  def update_weights_instanced(self, particles, frame_idx):
    """ This function update the weight for each particle using the difference
    between current range image and the synthetic rendering for each particle.
    Here, we use instance rendering to accelerate the sensor model
    Input:
        particles: each particle has four properties [x, y, theta, weight]
        frame_idx: the index of the current frame
    Output:
        particles ... same particles with updated particles(i).weight
    """
    
    # load current scan and compute the histogram
    current_path = self.scan_paths[frame_idx]
    current_vertex = load_vertex(current_path)
    current_range, _, _, _ = range_projection(current_vertex,
                                              fov_up=self.params["fov_up"],
                                              fov_down=self.params["fov_down"],
                                              proj_H=self.params["height"],
                                              proj_W=self.params["width"],
                                              max_range=self.params["max_range"])
    # self.save_depth_image('current_frame', current_range, frame_idx, 0)
    scores = np.ones(len(particles)) * 0.00001
    
    tiles_collection = []  # for counter number of tiles
    tiles_mask = np.ones(len(particles)) * -1  # for clustering
    
    for idx in range(len(particles)):
      particle = particles[idx]
      
      # first check whether the particle is inside the map or not
      if particle[0] < self.map_module.map_boundaries[0] or \
          particle[0] > self.map_module.map_boundaries[1] or \
          particle[1] < self.map_module.map_boundaries[2] or \
          particle[1] > self.map_module.map_boundaries[3]:
        continue
      
      # get tile index given particle position
      tile_idx = self.map_module.get_tile_idx([particle[0], particle[1]])
      if not self.map_module.tiles[tile_idx].valid:
        continue
      tiles_mask[idx] = tile_idx
      if tile_idx not in tiles_collection:
        tiles_collection.append(tile_idx)
    
    # we render all particles lies in the same tile instancely once
    for tile_idx in tiles_collection:
      # get tile vertices start point and size
      start = self.map_module.tiles[tile_idx].vertices_buffer_start
      size = self.map_module.tiles[tile_idx].vertices_buffer_size
      
      # collect poses of particles in the same tile
      mask = np.argwhere(tiles_mask == tile_idx)
      particles_in_tile = particles[mask]
      num_particles_in_tile = len(particles_in_tile)
      
      for interval_idx in range(int(num_particles_in_tile / self.max_instance) + 1):
        particles_in_tile_ = particles_in_tile[interval_idx * self.max_instance:
                                               (interval_idx + 1) * self.max_instance]
        num_particles_in_tile_ = len(particles_in_tile_)
        particle_poses = []
        for particle_idx in range(num_particles_in_tile_):
          particle = particles_in_tile_[particle_idx, 0]
          particle_pose = np.identity(4)
          particle_pose[0, 3] = particle[0]  # particle[0]
          particle_pose[1, 3] = particle[1]  # particle[1]
          particle_pose[2, 3] = self.map_module.tiles[tile_idx].z  # use tile z
          particle_pose[:3, :3] = rotation_matrix_from_euler_angles(particle[2], degrees=False)  # rotation
          particle_poses.append(particle_pose)
        
        # generate synthetic range image
        self.renderer.render_instanced(particle_poses, start, size)
        particle_depth = self.renderer.get_instance_depth_map()
        # update the weight
        scores_ = []
        for particle_idx in range(num_particles_in_tile_):
          diff = abs(particle_depth[particle_idx] - current_range)
          scores_.append(np.exp(-0.5 * np.mean(diff[current_range > 0]) ** 2 / (2.0 ** 2)))
        
        indices = mask[interval_idx * self.max_instance:(interval_idx + 1) * self.max_instance]
        if len(indices) > 1:
          scores[indices.squeeze()] = scores_
    
    # normalization
    particles[:, 3] = particles[:, 3] * scores
    particles[:, 3] = particles[:, 3] / np.max(particles[:, 3])
    
    # check convergence using supporting tile map idea
    if len(tiles_collection) < 2 and not self.is_converged:
      self.is_converged = True
      print('Converged!')
      # cutoff redundant particles and leave only num of particles
      idxes = np.argsort(particles[:, 3])[::-1]
      particles = particles[idxes[:100]]
    
    return particles, len(particles)
  
  def save_depth_image(self, folder_name, current_range, frame_idx, idx):
    """ Saving renderings for debugging """
    fig = plt.figure(frameon=False)  # frameon=False, suppress drawing the figure background patch.
    fig.set_size_inches(9, 0.64)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(current_range, aspect='equal')
    fig.savefig(os.path.join(folder_name, str(frame_idx).zfill(6) + '_' + str(idx) + '.png'))
    plt.close()


def test_map_render():
  """ debugging """
  map_file = '/path/to/scan/map'
  scan_folder = '/path/to/scan/folder'
  correlation_sensor = SensorModel(map_file, scan_folder)
  
  particles = np.zeros((100, 4))
  particles[:, 0] = np.arange(100) - 50
  particles[:, 3] = np.ones(len(particles))
  
  for frame_id in range(1):
    correlation_sensor.update_weights(particles, frame_id)


if __name__ == '__main__':
  # test_map_render()
  pass
