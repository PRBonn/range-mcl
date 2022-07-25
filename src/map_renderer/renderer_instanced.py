#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: rendering views from a given map at arbitrary 3d locations.

from typing import Dict
import os
import math
import time
import numpy as np
from map_renderer import Mesh
import OpenGL.GL as gl

from map_renderer.glow import GlBuffer, GlProgram, GlShader, GlFramebuffer, \
  GlRenderbuffer, GlTexture2D, vec4, vec3, GlTextureBuffer

import map_renderer.glow as glow
glow.WARN_INVALID_UNIFORMS = True


def glPerspective(fov, aspect, z_near, z_far):
  """ generate perspective matrix.
  For more details see https://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml
  """

  M = np.zeros((4, 4), dtype=np.float32)

  # Copied from gluPerspective
  f = 1.0 / math.tan(0.5 * fov)

  M[0, 0] = f / aspect
  M[1, 1] = f
  M[2, 2] = (z_near + z_far) / (z_near - z_far)
  M[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
  M[3, 2] = -1.0

  return M


def normalize(vec: np.array):
  """ normalize. """
  length = math.sqrt(np.dot(vec, vec))
  if abs(length) < 0.0000001: return vec
  return vec / length


def lookAt(x_cam, y_cam, z_cam, x_ref, y_ref, z_ref):
  """ generate view matrix. """
  # determine rotation from current location:
  pos_cam = vec3(x_cam, y_cam, z_cam)
  pos = vec3(x_ref, y_ref, z_ref)
  up = vec3(0.0, 1.0, 0.0)
  f = normalize(pos - pos_cam)
  x_axis = normalize(np.cross(f, up))
  y_axis = normalize(np.cross(x_axis, f))
  z_axis = -f

  view_matrix = np.zeros((4, 4), dtype=np.float32)

  view_matrix[0, :3] = x_axis
  view_matrix[1, :3] = y_axis
  view_matrix[2, :3] = z_axis

  view_matrix[3, 3] = 1.0

  # effectively => R * T
  view_matrix[0, 3] = np.dot(-pos_cam, x_axis)
  view_matrix[1, 3] = np.dot(-pos_cam, y_axis)
  view_matrix[2, 3] = np.dot(-pos_cam, z_axis)

  return view_matrix


class MapRenderer:
  """ rendering views from a given map at arbitrary 3d locations. """

  def __init__(self, params: Dict):
    """
    Args:
      params[Dict]: the parameter that are used to render: width, height, fov_up, fov_down, min_depth, max_depth
    """

    self._width = int(params["width"])
    self._height = int(params["height"])
    self.max_instance = params["max_instance"]

    self._mesh = None

    current_directory = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    self._instance_render_program = GlProgram()
    self._instance_render_program.attach(
        GlShader.from_file(gl.GL_VERTEX_SHADER, os.path.join(current_directory, "shader/render_instanced.vert")))
    self._instance_render_program.attach(
        GlShader.from_file(gl.GL_GEOMETRY_SHADER, os.path.join(current_directory, "shader/render_instanced.geom")))
    self._instance_render_program.attach(
        GlShader.from_file(gl.GL_FRAGMENT_SHADER, os.path.join(current_directory, "shader/render_instanced.frag")))
    self._instance_render_program.link()

    self._instance_render_program.bind()
    self._instance_render_program["fov_up"] = float(params["fov_up"])
    self._instance_render_program["fov_down"] = float(params["fov_down"])
    self._instance_render_program["min_depth"] = float(params["min_range"])
    self._instance_render_program["max_depth"] = float(params["max_range"])
    self._instance_render_program.release()

    self._pose_buffer = GlBuffer(gl.GL_TEXTURE_BUFFER)
    self._pose_texture = GlTextureBuffer(self._pose_buffer, gl.GL_RGBA32F)

    self._draw_program = GlProgram()
    self._draw_program.attach(
        GlShader.from_file(gl.GL_VERTEX_SHADER, os.path.join(current_directory, "shader/draw_mesh.vert")))
    self._draw_program.attach(
        GlShader.from_file(gl.GL_FRAGMENT_SHADER, os.path.join(current_directory, "shader/draw_mesh.frag")))
    self._draw_program.link()

    self._draw_program.bind()

    def set_uniform(program: GlProgram, name: str, value):
      loc = gl.glGetUniformLocation(program.id, name)
      if isinstance(value, float):
        gl.glUniform1f(loc, value)
      elif isinstance(value, np.ndarray):
        if value.shape[0] == 4:
          gl.glUniform4fv(loc, 1, value)
        elif value.shape[0] == 3:
          gl.glUniform3fv(loc, 1, value)
      else:
        raise NotImplementedError("implement.")

    set_uniform(self._draw_program, "lights[0].position", vec4(0, 0, -1, 0))
    set_uniform(self._draw_program, "lights[0].ambient", vec3(.01, .01, .01))
    set_uniform(self._draw_program, "lights[0].diffuse", vec3(.9, .9, .9))
    set_uniform(self._draw_program, "lights[0].specular", vec3(.9, .9, .9))

    # more evenly distributed sun light...
    dirs = [vec4(1, -1, 1, 0), vec4(-1, -1, 1, 0), vec4(1, -1, -1, 0), vec4(-1, -1, -1, 0)]
    indirect_intensity = vec3(.1, .1, .1)

    for i, direction in enumerate(dirs):
      light_name = "lights[{}]".format(i + 1)

      set_uniform(self._draw_program, light_name + ".position", direction)
      set_uniform(self._draw_program, light_name + ".ambient", vec3(.01, .01, .01))
      set_uniform(self._draw_program, light_name + ".diffuse", indirect_intensity)
      set_uniform(self._draw_program, light_name + ".specular", indirect_intensity)

    set_uniform(self._draw_program, "material.ambient", vec3(0.9, 0.9, 0.9))
    set_uniform(self._draw_program, "material.diffuse", vec3(0.0, 0.0, 0.9))
    set_uniform(self._draw_program, "material.specular", vec3(0.0, 0.0, 0.0))
    set_uniform(self._draw_program, "material.emission", vec3(0.0, 0.0, 0.0))
    set_uniform(self._draw_program, "material.shininess", 1.0)
    set_uniform(self._draw_program, "material.alpha", 1.0)

    self._draw_program.release()
    self._draw_program["num_lights"] = 5

    self._draw_program["model_mat"] = np.identity(4, dtype=np.float32)
    self._draw_program["normal_mat"] = np.identity(4, dtype=np.float32)

    self._vao_no_point = gl.glGenVertexArrays(1)
    self._draw_texture = GlProgram()
    self._draw_texture.attach(
        GlShader.from_file(gl.GL_VERTEX_SHADER, os.path.join(current_directory, "shader/empty.vert")))
    self._draw_texture.attach(
        GlShader.from_file(gl.GL_GEOMETRY_SHADER, os.path.join(current_directory, "shader/quad.geom")))
    self._draw_texture.attach(
        GlShader.from_file(gl.GL_FRAGMENT_SHADER, os.path.join(current_directory, "shader/drawtex.frag")))
    self._draw_texture.link()

    self.width_window = 640
    self.height_window = 480

    fov = math.radians(45.0)
    aspect = self.width_window / self.height_window

    self._projection = glPerspective(fov, aspect, 0.1, 10000.0)
    self._conversion = np.array([0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32).reshape(4, 4)

    """ initialize buffers dynamically for poses and textures"""
    self._instance_render_program["max_particles"] = self.max_instance
    self._pose_buffer.assign(np.zeros((self.max_instance * 4 * 4)))

    self.vertex_map = GlTexture2D(self._width, self.max_instance * self._height, gl.GL_RGBA32F)
    self.normal_map = GlTexture2D(self._width, self.max_instance * self._height, gl.GL_RGBA32F)
    self.depth_map = GlTexture2D(self._width, self.max_instance * self._height, gl.GL_R32F)

    # prepare for instance rendering output.
    self.instance_vertex_map = GlTexture2D(self._width, self.max_instance * self._height, gl.GL_RGBA32F)
    self.instance_normal_map = GlTexture2D(self._width, self.max_instance * self._height, gl.GL_RGBA32F)
    self.instance_depth_map = GlTexture2D(self._width, self.max_instance * self._height, gl.GL_R32F)

    self.instance_data_vertex = np.zeros([self.max_instance * self._height, self._width, 4])
    self.instance_data_normal = np.zeros([self.max_instance * self._height, self._width, 4])
    self.instance_data_depth = np.zeros([self.max_instance * self._height, self._width])

    self._instance_rbo = GlRenderbuffer(self._width, self.max_instance * self._height, gl.GL_DEPTH24_STENCIL8)

    self._instance_framebuffer = GlFramebuffer(self._width, self.max_instance * self._height)
    self._instance_framebuffer.attach(gl.GL_COLOR_ATTACHMENT0, self.instance_vertex_map)
    self._instance_framebuffer.attach(gl.GL_COLOR_ATTACHMENT1, self.instance_normal_map)
    self._instance_framebuffer.attach(gl.GL_COLOR_ATTACHMENT2, self.instance_depth_map)
    self._instance_framebuffer.attach(gl.GL_DEPTH_STENCIL_ATTACHMENT, self._instance_rbo)

    self._instance_framebuffer.bind()
    gl.glDrawBuffers([gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2])
    self._instance_framebuffer.release()

  def set_mesh(self, mesh: Mesh):
    self._mesh = mesh

  def render_instanced(self, poses, start, size):
    # prepare buffers for poses and textures
    num_particles = len(poses)
    # self.prepare_buffers(num_particles)

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClearColor(0, 0, 0, 0)
    gl.glDepthFunc(gl.GL_LESS)

    self._instance_framebuffer.bind()

    gl.glViewport(0, 0, self._instance_framebuffer.width, self._instance_framebuffer.height)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    self._instance_render_program.bind()
    self._instance_render_program["model_pose"] = np.identity(4, dtype=np.float32)

    # reshape poses into 1D buffer compatible format.
    # use transpose because glsl declare mat column-wisely
    buffered_poses = np.zeros((num_particles * 16), dtype=np.float32)
    for i, pose in enumerate(poses):
      buffered_poses[i * 16:(i + 1) * 16] = np.linalg.inv(pose.T).reshape(-1)
    self._pose_buffer.assign(buffered_poses)

    # draw renderings
    self._mesh.draw_with_tile_instanced(self._instance_render_program, start, size, num_particles)

    self._instance_render_program.release()
    self._instance_framebuffer.release()

  def get_instance_depth_map(self):
    """ Return the depth map.
        Returns: the depthmap as a numpy array of size (height, width)
    """
    data = self.instance_depth_map.download()
    self.instance_data_depth = data[:, :, 0]
    instance_data_depth = np.flipud(
        self.instance_data_depth.reshape(self.instance_data_depth.shape[1], self.instance_data_depth.shape[0]))
    slice_len = self._height
    sliced_instances = [instance_data_depth[slice_len*i:slice_len*(i+1)]
                        for i in range(len(instance_data_depth) // slice_len)]

    return sliced_instances[::-1]
