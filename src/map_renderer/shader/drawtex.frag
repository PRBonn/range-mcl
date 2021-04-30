#version 330 core


uniform sampler2D tex;
uniform sampler2DRect tex_rect;

uniform int tex_mode;

uniform int color_mode; // 0 plain, 1 min/max viridis,  2 colormap, 3 random, 4 nonzero, 5 abs
uniform sampler1D texColormap;
uniform int colorMapSize;

uniform float min_value;
uniform float max_value;
uniform int component;
uniform int num_components;

in vec2 texCoords;
out vec4 color;

// source: http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

const vec4 rviridis = vec4(2.90912735, -2.14404531, 0.04439198,  0.29390206);
const vec4 gviridis = vec4(-0.17293242, -0.16906214,  1.24131122,  0.01871256);
const vec4 bviridis = vec4(0.17848859, -1.72405244,  1.23042564,  0.34479632);

// approximate version of viridis colormap.
vec3 viridis(float t)
{
  vec4 tt = vec4(t*t*t, t*t, t, 1.0);
  return vec3(dot(tt, rviridis), dot(tt, gviridis), dot(tt, bviridis)); 
}

float rand(float x)
{
  return fract(sin(x)*1234823.);
}


void main()
{
  vec4 in_color = vec4(0,0,0,1);
  
  in_color = texture(tex, texCoords);
  if(tex_mode == 1) in_color = texture(tex_rect, texCoords * textureSize(tex_rect));

  
  if (color_mode == 0)
  {
    color = vec4(in_color.xyz, 1.0); 
  }
  else if (color_mode == 1)
  {
    color = vec4(viridis((in_color[component] - min_value) / (max_value - min_value)), 1.0); 
  }
  else if (color_mode == 2)
  {
    float factor = float(colorMapSize) / float(textureSize(texColormap, 0));
    color = texture(texColormap, factor * ((in_color[component] - min_value) / (max_value - min_value)));
    // color = vec4(hsv2rgb(vec3(in_color[0] / 50.0, 1, 1)), 1.0);
  }
  else if (color_mode == 3)
  {
    color = vec4(hsv2rgb(vec3(rand(in_color[component]), 1, 1)), 1.0); 
  }
  else if (color_mode == 4)
  {
    if(abs(in_color[component]) > 0) color = vec4(1, 0, 0, 1);
    else color = vec4(0, 0, 0, 1); 
  }
    else if (color_mode == 5)
  {
    color = abs(in_color); 
  }
}