#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: A renderable triangle mesh.

import open3d as o3d
import numpy as np

import OpenGL.GL as gl
from map_renderer.glow import GlBuffer, GlProgram


class Mesh:
  """ Representation of a mesh build.
      A mesh stores the vertices and the vertex indices for each triangle.
  """

  def __init__(self):
    self._buf_vertices = GlBuffer()
    self._buf_normals = GlBuffer()

    # setup the vertex array with the buffers.
    self._vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(self._vao)

    SIZEOF_FLOAT = 4

    self._buf_vertices.bind()
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * SIZEOF_FLOAT, None)
    gl.glEnableVertexAttribArray(0)

    self._buf_normals.bind()
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * SIZEOF_FLOAT, None)
    gl.glEnableVertexAttribArray(1)

    gl.glBindVertexArray(0)

  def draw(self, program: GlProgram):
    """ use program to draw triangles. """

    gl.glBindVertexArray(self._vao)
    program.bind()

    # gl.glDrawElements(gl.GL_TRIANGLES, self._buf_triangles.size, gl.GL_UNSIGNED_INT, 0)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._buf_vertices.size//3)

    program.release()
    gl.glBindVertexArray(0)
    
  def draw_with_tile(self, program, start, size):
    """ draw triangles. """

    gl.glBindVertexArray(self._vao)
    program.bind()

    # gl.glDrawElements(gl.GL_TRIANGLES, self._buf_triangles.size, gl.GL_UNSIGNED_INT, 0)
    # gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._buf_vertices.size)
    gl.glDrawArrays(gl.GL_TRIANGLES, start//3, size//3)

    program.release()
    gl.glBindVertexArray(0)

  def draw_with_tile_instanced(self, program, start, size, num_particles):
    """ draw triangles with batch of instances. """

    gl.glBindVertexArray(self._vao)
    program.bind()

    gl.glDrawArraysInstanced(gl.GL_TRIANGLES, start//3, size//3, num_particles)

    program.release()
    gl.glBindVertexArray(0)

  @staticmethod
  def Load(filename: str):
    """ load the mesh. """
    o3d_mesh = o3d.io.read_triangle_mesh(filename)
    if not o3d_mesh.has_vertex_normals(): o3d_mesh.compute_vertex_normals()
    
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float32)
    normals = np.asarray(o3d_mesh.vertex_normals, dtype=np.float32)
    triangles = np.asarray(o3d_mesh.triangles, dtype=np.int32)

    rearranged_vertices = vertices[triangles]
    rearranged_normals = normals[triangles]

    mesh = Mesh()
    mesh._buf_vertices.assign(rearranged_vertices.reshape(-1))
    mesh._buf_normals.assign(rearranged_normals.reshape(-1))

    return mesh, rearranged_vertices, o3d_mesh

  @property
  def num_triangles(self):
    return self._buf_vertices.size // 9
