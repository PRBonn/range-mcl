#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: Representation of an off-screen Window using glfw.

import glfw


class OffscreenWindow:
  """ Representation of an off-screen Window using glfw. """
  
  def __init__(self, init_glfw: bool = True, show: bool = False):
    if init_glfw:
      if not glfw.init(): raise RuntimeError("Unable to initialize glfw.")
    
    # See https://www.glfw.org/docs/latest/context.html#context_offscreen
    if not show: glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    self._window = glfw.create_window(640, 480, "test", None, None)
    glfw.make_context_current(self._window)
    
    if not self._window:
      glfw.terminate()
      raise RuntimeError("Unable to create window.")
  
  @property
  def glfw_window(self):
    return self._window
