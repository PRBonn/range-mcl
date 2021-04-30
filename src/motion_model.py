#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the motion model for overlap-based Monte Carlo localization.

from utils import *


def motion_model(particles, u, real_command=False, duration=0.1):
  """ MOTION performs the sampling from the proposal.
  distribution, here the rotation-translation-rotation motion model

  input:
     particles: the particles as in the main script
     u: the command in the form [rot1 trasl rot2] or real odometry [v, w]
     noise: the variances for producing the Gaussian noise for
     perturbating the motion,  noise = [noiseR1 noiseTrasl noiseR2]

  output:
     the same particles, with updated poses.

  The position of the i-th particle is given by the 3D vector
  particles(i).pose which represents (x, y, theta).

  Assume Gaussian noise in each of the three parameters of the motion model.
  These three parameters may be used as standard deviations for sampling.
  """
  num_particles = len(particles)
  if not real_command:
    # noise in the [rot1 trasl rot2] commands when moving the particles
    MOTION_NOISE = [0.01, 0.05, 0.01]
    r1Noise = MOTION_NOISE[0]
    transNoise = MOTION_NOISE[1]
    r2Noise = MOTION_NOISE[2]

    rot1 = u[0] + r1Noise * np.random.randn(num_particles)
    tras1 = u[1] + transNoise * np.random.randn(num_particles)
    rot2 = u[2] + r2Noise * np.random.randn(num_particles)

    # update pose using motion model
    particles[:, 0] += tras1 * np.cos(particles[:, 2] + rot1)
    particles[:, 1] += tras1 * np.sin(particles[:, 2] + rot1)
    particles[:, 2] += rot1 + rot2

  else:  # use real commands with duration
    # noise in the [v, w] commands when moving the particles
    MOTION_NOISE = [0.05, 0.05]
    vNoise = MOTION_NOISE[0]
    wNoise = MOTION_NOISE[1]

    # use the Gaussian noise to simulate the noise in the motion model
    v = u[0] + vNoise * np.random.randn(num_particles)
    w = u[1] + wNoise * np.random.randn(num_particles)
    gamma = wNoise * np.random.randn(num_particles)

    # update pose using motion models
    particles[:, 0] += - v / w * np.sin(particles[:, 2]) + v / w * np.sin(particles[:, 2] + w * duration)
    particles[:, 1] += v / w * np.cos(particles[:, 2]) - v / w * np.cos(particles[:, 2] + w * duration)
    particles[:, 2] += w * duration + gamma * duration

  return particles


def gen_commands(poses):
  """ Create commands out of the ground truth with noise.
  input:
    ground truth poses

  output:
    commands for each frame.
  """
  # compute noisy-free commands
  # set the default command = [0,0,0]'
  commands = np.zeros((len(poses), 3))
  
  # compute relative poses
  rela_poses = []
  headings = []
  last_pose = poses[0]
  for idx in range(len(poses)):
    rela_poses.append(np.linalg.inv(last_pose).dot(poses[idx]))
    headings.append(euler_angles_from_rotation_matrix(poses[idx][:3, :3])[2])
    last_pose = poses[idx]
  
  rela_poses = np.array(rela_poses)
  dx = (poses[1:, 0, 3] - poses[:-1, 0, 3])
  dy = (poses[1:, 1, 3] - poses[:-1, 1, 3])
  
  direct = np.arctan2(dy, dx)  # atan2(dy, dx), 1X(S-1) direction of the movement
  
  r1 = []
  r2 = []
  distance = []
  
  for idx in range(len(rela_poses) - 1):
    r1.append(direct[idx] - headings[idx])
    r2.append(headings[idx + 1] - direct[idx])
    distance.append(np.sqrt(dx[idx] * dx[idx] + dy[idx] * dy[idx]))
  
  r1 = np.array(r1)
  r2 = np.array(r2)
  distance = np.array(distance)
  
  # add noise to commands
  commands_ = np.c_[r1, distance, r2]
  commands[1:] = commands_ + np.array([0.01 * np.random.randn(len(commands_)),
                                       0.01 * np.random.randn(len(commands_)),
                                       0.01 * np.random.randn(len(commands_))]).T

  return commands


def gen_motion_reckon(commands):
  """ Generate motion reckon only for comparison.
  """

  particle = [0, 0, 0, 1]
  motion_reckon = []
  for cmmand in commands:
    # use the Gaussian noise to simulate the noise in the motion model
    rot1 = cmmand[0]
    tras1 = cmmand[1]
    rot2 = cmmand[2]
  
    # update pose using motion model
    particle[0] = particle[0] + tras1 * np.cos(particle[2] + rot1)
    particle[1] = particle[1] + tras1 * np.sin(particle[2] + rot1)
    particle[2] = particle[2] + rot1 + rot2

    motion_reckon.append([particle[0], particle[1]])
  
  return np.array(motion_reckon)


if __name__ == '__main__':
  pass

