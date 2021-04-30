#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: OpenGL Object Wrapper (GLOW)  in python.
# Some convenience classes to simplify resource management.

import re
from typing import Any, Union
from OpenGL.raw.GL.VERSION.GL_3_0 import GL_R32F, GL_RG

import numpy as np
import OpenGL.GL as gl

gl.ERROR_CHECKING = True
gl.ERROR_ON_COPY = True
gl.WARN_ON_FORMAT_UNAVAILABLE = True
WARN_INVALID_UNIFORMS = False


def vec2(x: float, y: float) -> np.array:
  """ returns an vec2-compatible numpy array """
  return np.array([x, y], dtype=np.float32)


def vec3(x: float, y: float, z: float) -> np.array:
  """ returns an vec3-compatible numpy array """
  return np.array([x, y, z], dtype=np.float32)


def vec4(x: float, y: float, z: float, w: float) -> np.array:
  """ returns an vec4-compatible numpy array """
  return np.array([x, y, z, w], dtype=np.float32)


def ivec2(x: int, y: int) -> np.array:
  """ returns an ivec2-compatible numpy array """
  return np.array([x, y], dtype=np.int32)


def ivec3(x: int, y: int, z: int) -> np.array:
  """ returns an ivec3-compatible numpy array """
  return np.array([x, y, z], dtype=np.int32)


def ivec4(x: int, y: int, z: int, w: int) -> np.array:
  """ returns an ivec4-compatible numpy array """
  return np.array([x, y, z, w], dtype=np.int32)


def uivec2(x: int, y: int) -> np.array:
  """ returns an uivec2-compatible numpy array """
  return np.array([x, y], dtype=np.uint32)


def uivec3(x: int, y: int, z: int) -> np.array:
  """ returns an uivec3-compatible numpy array """
  return np.array([x, y, z], dtype=np.uint32)


def uivec4(x: int, y: int, z: int, w: int) -> np.array:
  """ returns an uivec4-compatible numpy array """
  return np.array([x, y, z, w], dtype=np.uint32)


class GlBuffer:
  """ Buffer object representing a vertex array buffer.
  """
  
  def __init__(self, target: gl.Constant = gl.GL_ARRAY_BUFFER, usage: gl.Constant = gl.GL_STATIC_DRAW):
    self.id_ = gl.glGenBuffers(1)
    self.target_ = target
    self.usage_ = usage
    self._size = 0
  
  def __del__(self):
    gl.glDeleteBuffers(1, [self.id_])
  
  def assign(self, array: np.array):
    """ set buffer content to given numpy array. """
    gl.glBindBuffer(self.target_, self.id_)
    gl.glBufferData(self.target_, array, self.usage_)
    gl.glBindBuffer(self.target_, 0)
    self._size = array.shape[0]
  
  def bind(self):
    """ bind buffer """
    gl.glBindBuffer(self.target_, self.id_)
  
  def release(self):
    """ release buffer """
    gl.glBindBuffer(self.target_, 0)
  
  @property
  def id(self) -> int:
    """ get buffer id """
    return self.id_
  
  @property
  def usage(self) -> gl.Constant:
    """ get buffer usage """
    return self.usage_
  
  @property
  def target(self) -> gl.Constant:
    """ get buffer target """
    return self.target_
  
  @property
  def size(self) -> int:
    return self._size


class GlTextureBuffer:
  """
    Texture based on a GlBuffer object. A texture object, where the texture's data is stored in a buffer object.
  """

  def __init__(self, buffer: GlBuffer, tex_format):
    self._buffer = buffer  # keep here a reference to avoid garbage collection.
    self.id_ = gl.glGenTextures(1)
    self._tex_format = tex_format

    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.id_)
    self._buffer.bind()
    gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, self._tex_format, self._buffer.id)
    self._buffer.release()

  def __del__(self):
    gl.glDeleteBuffers(1, np.array([self.id_]))

  def bind(self):
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.id_)

  def release(self):
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)


class GlTextureRectangle:
  """ TextureRectangle
      TODO: make GlTextureRectangle and GlTexture2D subclasses of GlTextureBase
  """
  
  def __init__(self, width, height, internalFormat=gl.GL_RGBA, format=gl.GL_RGBA):
    self.id_ = gl.glGenTextures(1)
    self.internalFormat_ = internalFormat  # gl.GL_RGB_FLOAT, gl.GL_RGB_UNSIGNED, ...
    self.format = format  # GL_RG. GL_RG_INTEGER, ...
    
    self.width_ = width
    self.height_ = height
    
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, self.id_)
    gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
    gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, 0)
  
  def __del__(self):
    gl.glDeleteTextures(1, [self.id_])
  
  def bind(self, texture_unit_id: int):
    """ bind texture to given texture unit.

    Args:
      texture_unit_id (int): id of texture unit to which the texture should be bound.
    """
    gl.glActiveTexture(gl.GL_TEXTURE0 + int(texture_unit_id))
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, self.id_)
  
  def release(self, texture_unit_id: int):
    """ release texture from given texture unit.

    Args:
      texture_unit_id (int): id of texture unit from which the texture should be released.
    """
    gl.glActiveTexture(gl.GL_TEXTURE0 + int(texture_unit_id))
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, 0)
  
  def assign(self, array: np.array):
    """ assign data from given numpy array to the texture.

    Depending on the dtype of np.array different texture uploads are triggered.

    Args:
        array (np.array): data represented as numpy array.

    Raises:
        NotImplementedError: raised when unsupported dtype of the given array is encountered.
    """
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, self.id_)
    
    if array.dtype == np.uint8:
      gl.glTexImage2D(gl.GL_TEXTURE_RECTANGLE, 0, self.internalFormat_, self.width_, self.height_, 0, self.format,
                      gl.GL_UNSIGNED_BYTE, array)
    elif array.dtype == np.float32:
      gl.glTexImage2D(gl.GL_TEXTURE_RECTANGLE, 0, self.internalFormat_, self.width_, self.height_, 0, self.format,
                      gl.GL_FLOAT, array)
    else:
      raise NotImplementedError("pixel type not implemented.")
    
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, 0)
  
  @property
  def id(self):
    """ get the texture's OpenGL id. """
    return self.id_


class GlTexture2D:
  """ Texture2D
  """
  
  def __init__(self, width, height, internal_format=gl.GL_RGBA):
    self.id_ = gl.glGenTextures(1)
    self.internal_format_ = internal_format  # gl.GL_RGB_FLOAT, gl.GL_RGB_UNSIGNED, ...
    
    self.width_ = width
    self.height_ = height
    
    gl.glBindTexture(gl.GL_TEXTURE_2D, self.id_)
    self._allocateMemory()
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
    # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
    
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
  
  def __del__(self):
    gl.glDeleteTextures([self.id_])
  
  def bind(self, texture_unit_id: int):
    """ bind texture to given texture unit.

    Args:
      texture_unit_id (int): id of texture unit to which the texture should be bound.
    """
    gl.glActiveTexture(gl.GL_TEXTURE0 + int(texture_unit_id))
    gl.glBindTexture(gl.GL_TEXTURE_2D, self.id_)
  
  def release(self, texture_unit_id: int):  # pylint: disable=no-self-use
    """ release texture from given texture unit.

    Args:
        texture_unit_id (int): id of texture unit from which the texture should be released.
    """
    gl.glActiveTexture(gl.GL_TEXTURE0 + int(texture_unit_id))
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
  
  def assign(self, array: np.array, format: gl.GLenum):
    """assign data from given numpy array to the texture.

    Depending on the dtype of np.array different texture uploads are triggered.

    Args:
        array (np.array): data represented as numpy array.

    Raises:
        NotImplementedError: raised when unsupported dtype of the given array is encountered.
    """
    gl.glBindTexture(gl.GL_TEXTURE_2D, self.id_)
    
    if array.dtype == np.uint8:
      gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.internal_format_, self.width_, self.height_, 0, format,
                      gl.GL_UNSIGNED_BYTE, array)
    elif array.dtype == np.uint32:
      gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.internal_format_, self.width_, self.height_, 0, format,
                      gl.GL_UNSIGNED_INT, array)
    elif array.dtype == np.float32:
      gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.internal_format_, self.width_, self.height_, 0, format, gl.GL_FLOAT,
                      array)
    else:
      raise NotImplementedError("pixel type not implemented.")
    
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
  
  def set_filter(self, min_filter: gl.Constant, mag_filter: gl.Constant):
    """ set the filter operations performance when up-/down-sampling of texture is required. 

    Args:
        min_filter (gl.Constant): filtering used for down-sampling
        mag_filter (gl.Constant): filtering used for up-sampling
    """
    gl.glBindTexture(gl.GL_TEXTURE_2D, self.id_)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, min_filter)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, mag_filter)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
  
  def resize(self, width: int, height: int):
    """ resize texture to given width and height. """
    self.width_ = width
    self.height_ = height
    # need to copy?
  
  def download(self) -> np.array:
    gl.glBindTexture(gl.GL_TEXTURE_2D, self.id_)
    array = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, gl.GL_FLOAT)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    return array
  
  @property
  def id(self) -> gl.GLuint:
    """ get the texture's id """
    return self.id_
  
  @property
  def width(self) -> int:
    """ get the texture's width """
    return self.width_
  
  @property
  def height(self) -> int:
    """ get the texture's height """
    return self.height_
  
  def _allocateMemory(self):
    pixel_format = gl.GL_RGBA
    pixel_type = gl.GL_UNSIGNED_BYTE
    
    if self.internal_format_ in [gl.GL_R, gl.GL_RG, gl.GL_RGB, gl.GL_RGBA]:
      pixel_type = gl.GL_UNSIGNED_BYTE
    elif self.internal_format_ in [gl.GL_R32I, gl.GL_RG32I, gl.GL_RGB32I, gl.GL_RGBA32I]:
      pixel_type = gl.GL_INT
    elif self.internal_format_ in [gl.GL_R32F, gl.GL_RG32F, gl.GL_RGB32F, gl.GL_RGBA32F]:
      pixel_type = gl.GL_FLOAT
    
    if self.internal_format_ in [gl.GL_R, GL_R32F]:
      pixel_format = gl.GL_RED
    elif self.internal_format_ in [gl.GL_RG, gl.GL_RG32F]:
      pixel_format = gl.GL_RG
    elif self.internal_format_ in [gl.GL_RGB, gl.GL_RGB32F]:
      pixel_format = gl.GL_RGB
    elif self.internal_format_ in [gl.GL_RGBA, gl.GL_RGBA32F]:
      pixel_format = gl.GL_RGBA
    elif self.internal_format_ in [gl.GL_R32I, gl.GL_R32UI]:
      pixel_format = gl.GL_RED_INTEGER
    elif self.internal_format_ in [gl.GL_RG32I, gl.GL_RG32UI]:
      pixel_format = gl.GL_RG_INTEGER
    elif self.internal_format_ in [gl.GL_RGB32I, gl.GL_RGB32UI]:
      pixel_format = gl.GL_RGB_INTEGER
    elif self.internal_format_ in [gl.GL_RGBA32I, gl.GL_RGBA32UI]:
      pixel_format = gl.GL_RGBA_INTEGER
    
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.internal_format_, self.width_, self.height_, 0, pixel_format, pixel_type,
                    None)


class GlTexture1D:
  """ Texture1D
      1-dimensional texture.
  """
  
  def __init__(self, width, internalFormat=gl.GL_RGBA):
    self.id_ = gl.glGenTextures(1)
    self.internalFormat_ = internalFormat  # gl.GL_RGB_FLOAT, gl.GL_RGB_UNSIGNED, ...
    
    self.width_ = width
    
    gl.glBindTexture(gl.GL_TEXTURE_1D, self.id_)
    
    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
    
    # gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.internalFormat_,
    #                 width_, height_, 0, gl.format, gl.GL_UNSIGNED_BYTE, None)
    gl.glBindTexture(gl.GL_TEXTURE_1D, 0)
  
  def __del__(self):
    gl.glDeleteTextures(1, [self.id_])
  
  def bind(self, textureUnitId: int):
    gl.glActiveTexture(gl.GL_TEXTURE0 + int(textureUnitId))
    gl.glBindTexture(gl.GL_TEXTURE_1D, self.id_)
  
  def release(self, textureUnitId: int):
    gl.glActiveTexture(gl.GL_TEXTURE0 + int(textureUnitId))
    gl.glBindTexture(gl.GL_TEXTURE_1D, 0)
  
  def assign(self, array: np.array, format: gl.GLenum):
    gl.glBindTexture(gl.GL_TEXTURE_1D, self.id_)
    
    if array.dtype == np.uint8:
      gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, self.internalFormat_, self.width_, 0, format, gl.GL_UNSIGNED_BYTE, array)
    elif array.dtype == np.uint32:
      gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, self.internalFormat_, self.width_, 0, format, gl.GL_UNSIGNED_INT, array)
    elif array.dtype == np.float32:
      gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, self.internalFormat_, self.width_, 0, format, gl.GL_FLOAT, array)
    else:
      raise NotImplementedError("pixel type not implemented.")
    
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
  
  def resize(self, width: int):
    """ resize texture. """
    self.width_ = width
  
  @property
  def id(self) -> gl.GLuint:
    """ get the id of the program. """
    return self.id_
  
  @property
  def width(self) -> int:
    """ get width of the texture. """
    return self.width_


class GlShader:
  """ OpenGL shader 
  """
  
  def __init__(self, shader_type, source):
    self._code = source
    self._shader_type = shader_type
    
    self.id_ = gl.glCreateShader(self._shader_type)
    gl.glShaderSource(self.id_, source)
    
    gl.glCompileShader(self.id_)
    
    success = gl.glGetShaderiv(self.id_, gl.GL_COMPILE_STATUS)
    if success == gl.GL_FALSE:
      error_string = gl.glGetShaderInfoLog(self.id_).decode("utf-8")
      readable_error = []
      source_lines = source.split("\n")
      for line in error_string.split("\n"):
        match = re.search(r"\(([0-9]+)\) : ([\s\w]+): ([\s\w\S\W]+)", line)
        if match:
          lineno = match.group(1)
          errorno = match.group(2)
          message = match.group(3)
          readable_message = "{}: {} at line {}:\n  {}: {}".format(errorno, message, lineno, lineno,
                                                                   source_lines[int(lineno)].strip())
          readable_error.append(readable_message)
        else:
          readable_error.append(line)
      
      raise RuntimeError("\n".join(readable_error))
  
  def __del__(self):
    gl.glDeleteShader(self.id_)
  
  @property
  def type(self) -> gl.Constant:
    """ return shader type """
    return self._shader_type
  
  @property
  def id(self) -> gl.GLuint:
    """ get id of shader """
    return self.id_
  
  @property
  def code(self):
    """ get shader source code. """
    return self._code
  
  @staticmethod
  def from_file(shader_type: gl.Constant, filename: str):
    """ load and initialize shader from given filename """
    f = open(filename)
    source = "".join(f.readlines())
    # todo: preprocess.
    f.close()
    
    return GlShader(shader_type, source)


class GlProgram:
  """ An OpenGL program handle. """
  
  def __init__(self):
    self.id_ = gl.glCreateProgram()
    self._shaders = {}
    self._uniform_types = {}
    self.is_linked = False
  
  # todo: figure this out.
  def __del__(self):
    gl.glDeleteProgram(self.id_)
  
  def bind(self):
    if not self.is_linked:
      raise RuntimeError("Program must be linked before usage.")
    gl.glUseProgram(self.id_)
  
  def release(self):
    gl.glUseProgram(0)
  
  def attach(self, shader):
    self._shaders[shader.type] = shader
  
  @property
  def id(self):
    return self.id_
  
  def __setitem__(self, name: str, value: Any):
    # quitely ignore
    if name not in self._uniform_types:
      if WARN_INVALID_UNIFORMS: print("No uniform with name '{}' available.".format(name))
      return
    
    current = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
    if current != self.id_: self.bind()
    
    loc = gl.glGetUniformLocation(self.id_, name)
    T = self._uniform_types[name]
    
    if T == "int":
      gl.glUniform1i(loc, np.int32(value))
    elif T == "uint":
      gl.glUniform1ui(loc, np.uint32(value))
    elif T == "float":
      gl.glUniform1f(loc, np.float32(value))
    elif T == "bool":
      gl.glUniform1f(loc, value)
    elif T == "vec2":
      gl.glUniform2fv(loc, 1, value)
    elif T == "vec3":
      gl.glUniform3fv(loc, 1, value)
    elif T == "vec4":
      gl.glUniform4fv(loc, 1, value)
    elif T == "ivec2":
      gl.glUniform2iv(loc, 1, value)
    elif T == "ivec3":
      gl.glUniform3iv(loc, 1, value)
    elif T == "ivec4":
      gl.glUniform4iv(loc, 1, value)
    elif T == "uivec2":
      gl.glUniform2uiv(loc, 1, value)
    elif T == "uivec3":
      gl.glUniform3uiv(loc, 1, value)
    elif T == "uivec4":
      gl.glUniform4uiv(loc, 1, value)
    elif T == "mat4":
      # print("set matrix: ", value)
      gl.glUniformMatrix4fv(loc, 1, False, value.T.astype(np.float32))
    elif T.endswith("sampler1D"):
      gl.glUniform1i(loc, np.int32(value))
    elif T.endswith("sampler2D"):
      gl.glUniform1i(loc, np.int32(value))
    elif T.endswith("sampler2DRect"):
      gl.glUniform1i(loc, np.int32(value))
    else:
      raise NotImplementedError("uniform type {} not implemented. :(".format(T))
    
    if current != self.id_: gl.glUseProgram(current)
  
  def link(self):
    if gl.GL_VERTEX_SHADER not in self._shaders or gl.GL_FRAGMENT_SHADER not in self._shaders:
      raise RuntimeError("program needs at least vertex and fragment shader")
    
    for shader in self._shaders.values():
      gl.glAttachShader(self.id_, shader.id)
      for line in shader.code.split("\n"):
        match = re.search(r"uniform\s+(\S+)\s+(\S+)\s*;", line)
        if match:
          self._uniform_types[match.group(2)] = match.group(1)
    
    gl.glLinkProgram(self.id_)
    is_linked = bool(gl.glGetProgramiv(self.id_, gl.GL_LINK_STATUS))
    if not is_linked:
      msg = gl.glGetProgramInfoLog(self.id_)
      
      raise RuntimeError(str(msg.decode("utf-8")))
    
    # after linking we don't need the source code anymore.
    # for shader in self.shaders_:
    #   del shader
    self._shaders = {}
    self.is_linked = True


class GlRenderbuffer:
  """ Wrapper for an OpenGL's renderbuffer """
  
  def __init__(self, width: int, height: int, renderbuffer_format=gl.GL_RGBA):
    self._format = renderbuffer_format
    self._width = width
    self._height = height
    self._id = gl.glGenRenderbuffers(1)
    
    # allocate storage.
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._id)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, self._format, self._width, self._height)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
  
  def __del__(self):
    gl.glDeleteRenderbuffers(1, [self._id])
  
  def bind(self):
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._id)
  
  def release(self):
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
  
  @property
  def width(self):
    return self._width
  
  @property
  def height(self):
    return self._height
  
  @property
  def id(self):
    return self._id


class GlFramebuffer:
  """ Wrapper for an OpenGL's framebuffer. """
  
  def __init__(self, width: int, height: int):
    self._id = gl.glGenFramebuffers(1)
    self._target = gl.GL_FRAMEBUFFER
    self._attachments = {}
    self._valid = False
    self._width = width
    self._height = height
  
  def __del__(self):
    gl.glDeleteFramebuffers(1, [self._id])
  
  def bind(self):
    if not self._valid: raise RuntimeError("Incomplete framebuffer.")
    gl.glBindFramebuffer(self._target, self._id)
  
  def release(self):
    gl.glBindFramebuffer(self._target, 0)
  
  def attach(self, target: int, attachment: Union[GlTexture2D, GlTextureRectangle, GlRenderbuffer]):
    """ attach Texture or Renderbuffer to given attachment target.

        Args:
          target: attachment target, e.g., GL_COLOR_ATTACHMENT0, GL_DEPTH_STENCIL_ATTACHMENT, ...
          attachment: texture or renderbuffer to attach.
    """
    
    if isinstance(attachment, (GlTexture2D, GlTextureRectangle)):
      texture_target = None
      if isinstance(attachment, GlTexture2D): texture_target = gl.GL_TEXTURE_2D
      if isinstance(attachment, GlTextureRectangle): texture_target = gl.GL_TEXTURE_RECTANGLE
      
      gl.glBindFramebuffer(self._target, self._id)
      gl.glFramebufferTexture2D(self._target, target, texture_target, attachment.id, 0)
      gl.glBindFramebuffer(self._target, 0)
    
    elif isinstance(attachment, GlRenderbuffer):
      gl.glBindFramebuffer(self._target, self._id)
      gl.glFramebufferRenderbuffer(self._target, target, gl.GL_RENDERBUFFER, attachment.id)
      gl.glBindFramebuffer(self._target, 0)
    else:
      raise ValueError("texture should be GlTexture2D, GlTextureRectangle or GlRenderbuffer.")
    
    self._attachments[target] = attachment
    gl.glBindFramebuffer(self._target, self._id)
    self._valid = gl.glCheckFramebufferStatus(self._target) == gl.GL_FRAMEBUFFER_COMPLETE
    
    gl.glBindFramebuffer(self._target, 0)
  
  @property
  def valid(self):
    """ is framebuffer valid? """
    return self._valid

  @property
  def height(self):
    """ framebuffer's height """
    return self._height

  @property
  def width(self):
    """ framebuffer's width """
    return self._width