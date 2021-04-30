#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: functions to simplify the ground mesh.

import copy
import numpy as np

from time_utils import timeit


@timeit
def pcd_ground_seg_pca(scan, th=0.80, z_offset=-1.1):
  """ Perform PCA over PointCloud to segment ground.
  """
  pcd = copy.deepcopy(scan)
  _, covariance = pcd.compute_mean_and_covariance()
  eigen_vectors = np.linalg.eig(covariance)[1]
  k = eigen_vectors.T[2]

  # magnitude of projecting each face normal to the z axis
  normals = np.asarray(scan.normals)
  points = np.asarray(scan.points)
  mag = np.linalg.norm(np.dot(normals, k).reshape(-1, 1), axis=1)
  ground = pcd.select_by_index(np.where((mag >= th) & (points[:, 2] < z_offset))[0])
  rest = pcd.select_by_index(np.where((mag >= th) & (points[:, 2] < z_offset))[0], invert=True)

  # Also remove the faces that are looking downwards
  up_normals = np.asarray(ground.normals)
  orientation = np.dot(up_normals, k)
  ground = ground.select_by_index(np.where(orientation > 0.0)[0])

  ground.paint_uniform_color([1.0, 0.0, 0.0])
  rest.paint_uniform_color([0.0, 0.0, 1.0])
  
  return ground, rest


@timeit
def pcd_ground_seg_open3d(scan, config):
  """ Open3D also supports segmententation of geometric primitives from point clouds using RANSAC.
  """
  pcd = copy.deepcopy(scan)

  ground_model, ground_indexes = scan.segment_plane(distance_threshold=config['distance_threshold'],
                                                    ransac_n=config['ransac_n'],
                                                    num_iterations=config['num_iterations'])
  ground_indexes = np.array(ground_indexes)

  ground = pcd.select_by_index(ground_indexes)
  rest = pcd.select_by_index(ground_indexes, invert=True)

  ground.paint_uniform_color(config['ground_color'])
  rest.paint_uniform_color(config['rest_color'])

  return ground, rest


@timeit
def mesh_simplify(mesh, config):
  """ simplify the ground meshes using simplify_vertex_clustering and filter_smooth_simple.
  """
  mesh_gnd = copy.deepcopy(mesh)
  mesh_rest = copy.deepcopy(mesh)
  
  # triangles.shape = n_t x 3 x 3,
  # where n_t is the number of triangles,
  # the first 3 is the three vertices
  # and the second three is the 3d coordinates of the vertices
  triangles = np.asarray(mesh.triangles, dtype=np.int32)
  
  # colors.shape = n_v x 3, where n_v is the number of vertices, 3 channel contain RGB
  colors = np.asarray(mesh.vertex_colors)
  rearranged_colors = colors[triangles]
  
  gnd_idx = np.argwhere((rearranged_colors[:, 0, 0] > 0.5) |
                        (rearranged_colors[:, 1, 0] > 0.5) |
                        (rearranged_colors[:, 2, 0] > 0.5))
  
  rest_idx = np.ones(len(rearranged_colors), np.bool)
  rest_idx[gnd_idx] = 0
  
  mesh_gnd.remove_triangles_by_index(rest_idx)
  mesh_rest.remove_triangles_by_index(gnd_idx)

  mesh_gnd = mesh_gnd.simplify_vertex_clustering(config['simplify_resolution'])
  mesh_gnd = mesh_gnd.filter_smooth_simple(number_of_iterations=config['number_of_iterations'])
  
  mesh = mesh_gnd + mesh_rest
  mesh = mesh.remove_duplicated_triangles()
  
  return mesh


def get_mesh_size(mesh):
  """ functions to compute the size of mesh.
  """
  size = -1
  triangles = np.array(mesh.triangles)
  vertices = np.array(mesh.vertices)
  size += triangles.size * triangles.itemsize
  size += vertices.size * vertices.itemsize
  return size


def get_mesh_size_kb(mesh):
  """ functions to compute the size of mesh in KB.
  """
  return np.floor(get_mesh_size(mesh) / 1024.0)


def get_mesh_size_mb(mesh):
  """ functions to compute the size of mesh in MB.
  """
  return np.floor(get_mesh_size_kb(mesh) / 1024.0)


if __name__ == '__main__':
  pass
