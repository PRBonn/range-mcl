#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: resample the particles

from utils import *


def resample(particles):
  """ Re-sampling module,
  here we use the classical low variance re-sampling.
  """
  weights = particles[:, 3]
  
  # normalize the weights
  weights = weights / sum(weights)
  
  # compute effective number of particles
  eff_N = 1 / sum(weights ** 2)
  
  # resample
  new_particles = np.zeros(particles.shape)
  i = 0
  if eff_N < len(particles)*3.0/4.0:
    r = np.random.rand(1) * 1.0/len(particles)
    c = weights[0]
    for idx in range(len(particles)):
      u = r + idx/len(particles)
      while u > c:
        if i >= len(particles) - 1:
          break
        i += 1
        c += weights[i]
      new_particles[idx] = particles[i]
  else:
    new_particles = particles
    
  return new_particles


def limit(particles, map_size):
  kld_z = 0.01
  precomputed_epsilon_factor = 0.02
  
  x_min = round(map_size[0])
  x_max = round(map_size[1])
  y_min = round(map_size[2])
  y_max = round(map_size[3])
  # create a grid files lookup table
  grid_paths_lut = np.full(((y_max - y_min)*2, (x_max - x_min))*2, 0, dtype=int)

  coords = particles[:, :2]
  
  map_xs = np.round(y_max - coords[:, 1]).astype(int)
  map_ys = np.round(coords[:, 0] - x_min).astype(int)

  grid_paths_lut[map_xs, map_ys] = 1
  cells_with_data = np.count_nonzero(grid_paths_lut)
  
  common = 2.0 / (9 * (cells_with_data - 1))
  to_be_cubed = 1 - common - np.sqrt(common) * kld_z
  result = ((cells_with_data - 1) / precomputed_epsilon_factor) * to_be_cubed * to_be_cubed * to_be_cubed
  
  return int(result)
  
  
def test_sampling():
  """ debugging """
  # near particles
  particles = np.array([[1, 0, 0, 1], [0, 0, 0, 0.5], [0, 0, 0, 0.1],
                        [0, 0, 0, 0.3], [0, 1, 0, 1], [0, 0, 0, 0.5]])
  
  particles = resample(particles)
  print(particles)


def test_limit():
  """ debugging """
  # near particles
  particles = np.array([[1, 0, 0, 1], [0, 0, 0, 0.5], [0, 0, 0, 0.1],
                        [0, 0, 0, 0.3], [0, 1, 0, 1], [0, 0, 0, 0.5]])
  mapsize = [-5, 5, -10, 10]
  
  k = limit(particles, mapsize)
  print(k)
  
  # far particles
  particles = np.array([[1, 0, 0, 1], [2, 0, 0, 0.5], [3, 0, 0, 0.1],
                        [-1, 0, 0, 0.3], [-2, 1, 0, 1], [-3, 0, 0, 0.5]])

  k = limit(particles, mapsize)
  print(k)
  
  
if __name__ == '__main__':
  # test_sampling()
  # test_limit()
  pass