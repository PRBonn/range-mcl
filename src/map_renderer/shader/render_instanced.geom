#version 330
layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;

in VS_OUT {
  vec4 position;
  vec4 normal;
  int instance;
}
gs_in[];

uniform float fov_up;
uniform float fov_down;
uniform float max_depth;
uniform float min_depth;
uniform int max_particles;

out vec4 normal;
out vec4 vertex;

const float pi = 3.14159265358979323846f;
const float inv_pi = 0.31830988618379067154f;
const float pi_2 = 1.57079632679;

vec3 project2model(vec4 position) {
  float fov = abs(fov_up) + abs(fov_down);
  float depth = length(position.xyz);
  float yaw = atan(position.y, position.x);
  float pitch = -asin(position.z / depth);

  float x = 0.5 * ((-yaw * inv_pi) + 1.0);                  // in [0, 1]
  float y = (1.0 - (degrees(pitch) + fov_up) / fov);        // in [0, 1]
  if(y <= 0.0) y = 0.0;
  if(y >= 1.0) y = 1.0;
  // if(y < 0 || y > 1) EndPrimitive();
  float z = (depth - min_depth) / (max_depth - min_depth);  // in [0, 1]

  return vec3(x, y, z);
}


/** 
 Compute intersection of straight line between a and b with x-axis.
**/
vec4 xaxis_intersection(vec4 a, vec4 b)
{
  float alpha = -b.y / (a.y - b.y);

  return alpha * a + (1 - alpha) * b;
}

/** 
 Compute intersection of straight line between a and b with y-axis.
**/
vec4 yaxis_intersection(vec4 a, vec4 b)
{
  float alpha = -b.x / (a.x - b.x);

  return alpha * a + (1 - alpha) * b;
}

// split given triangle at x axis if needed.
void split_at_xaxis(vec4[3] positions, vec4[3] normals)
{
  float subdiv_height =  1.0 / max_particles;

  // just check all cases...not prettey but should help to copy with boundary abovementioned issue.
  if((positions[0].y < 0 && !(positions[1].y < 0 || positions[2].y < 0)) || 
     (positions[1].y < 0 && !(positions[0].y < 0 || positions[2].y < 0)) || 
     (positions[2].y < 0 && !(positions[0].y < 0 || positions[1].y < 0)))
  {
    // first case:
    
    //   x-axis
    //     ^
    //     |
    // 1 --3-- 0 
    // |   |  /
    // |   | /
    // |   |/
    // |   4
    // |  /|
    // | / |
    // |/  |
    // 2   |

    int first_index = 0;
    if(positions[1].y < 0) first_index = 1;
    if(positions[2].y < 0) first_index = 2;
    
    vec4 v0 = positions[first_index];
    vec4 v1 = positions[(first_index + 1)%3];
    vec4 v2 = positions[(first_index + 2)%3];
    vec4 n0 = normals[first_index];
    vec4 n1 = normals[(first_index + 1)%3];
    vec4 n2 = normals[(first_index + 2)%3];

    vec4 v3 = xaxis_intersection(v0, v1);
    vec4 v4 = xaxis_intersection(v0, v2);

    vec3 pp;
    vec4 p;
    pp = project2model(v0);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
    gl_Position = p;
    vertex = vec4(v0.xyz, 1.0);  // w indicates the validity.
    normal = vec4(n0.xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(v3);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    if(p.x < -0.5 || p.x > 0.5) p.x = 1; // ensure that vertex is on the right side.
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v3.xyz, 1.0);
    normal = vec4(n1.xyz, 1.0);
    EmitVertex();

    pp = project2model(v4);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    if(p.x < -0.5 || p.x > 0.5) p.x = 1; // ensure that vertex is on the right side.
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v4.xyz, 1.0);
    normal = vec4(n2.xyz, 1.0);
    EmitVertex();
    EndPrimitive();

    // ====================================================

    // split other side into two triangles.
    pp = project2model(v3);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
    if(p.x < -0.5 || p.x > 0.5) p.x = -1;
    gl_Position = p;   
    vertex = vec4(v3.xyz, 1.0);  // w indicates the validity.
    normal = vec4(n0.xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(v1);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v1.xyz, 1.0);
    normal = vec4(n1.xyz, 1.0);
    EmitVertex();

    pp = project2model(v2);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v2.xyz, 1.0);
    normal = vec4(n2.xyz, 1.0);
    EmitVertex();
    EndPrimitive();

    // ====================================================

    pp = project2model(v3);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
    if(p.x < -0.5 || p.x > 0.5) p.x = -1;
    gl_Position = p;   
    vertex = vec4(v3.xyz, 1.0);  // w indicates the validity.
    normal = vec4(n0.xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(v2);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v2.xyz, 1.0);
    normal = vec4(n2.xyz, 1.0);
    EmitVertex();

    pp = project2model(v4);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    if(p.x < -0.5 || p.x > 0.5) p.x = -1;
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v4.xyz, 1.0);
    normal = vec4(n2.xyz, 1.0);
    EmitVertex();
    EndPrimitive();

  }
  else if((positions[0].y > 0 && !(positions[1].y > 0 || positions[2].y > 0)) || 
     (positions[1].y > 0 && !(positions[0].y > 0 || positions[2].y > 0)) || 
     (positions[2].y > 0 && !(positions[0].y > 0 || positions[1].y > 0)))
  {

    // second case:
    
    //   x-axis
    //     ^
    //     |
    // 0 --4-- 2 
    //  \  |   |
    //   \ |   |
    //    \|   |
    //     3   |
    //     |\  |
    //     | \ |
    //     |  1
    
    int first_index = 0;
    if(positions[1].y > 0) first_index = 1;
    if(positions[2].y > 0) first_index = 2;
    
    vec4 v0 = positions[first_index];
    vec4 v1 = positions[(first_index + 1)%3];
    vec4 v2 = positions[(first_index + 2)%3];
    vec4 n0 = normals[first_index];
    vec4 n1 = normals[(first_index + 1)%3];
    vec4 n2 = normals[(first_index + 2)%3];

    vec4 v3 = xaxis_intersection(v0, v1);
    vec4 v4 = xaxis_intersection(v0, v2);

    vec3 pp;
    vec4 p;
    pp = project2model(v0);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
    gl_Position = p;   
    vertex = vec4(v0.xyz, 1.0);  // w indicates the validity.
    normal = vec4(n0.xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(v3);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    if(p.x < -0.5 || p.x > 0.5) p.x = -1; // ensure that vertex is on the right side.
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v3.xyz, 1.0);
    normal = vec4(n1.xyz, 1.0);
    EmitVertex();

    pp = project2model(v4);  // [0,1] x [0,1]
    p = vec4(2.0 * pp - 1.0, 1.0f);
    if(p.x < -0.5 || p.x > 0.5) p.x = -1; // ensure that vertex is on the right side.
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v4.xyz, 1.0);
    normal = vec4(n2.xyz, 1.0);
    EmitVertex();
    EndPrimitive();

    // ====================================================

    // split other side into two triangles.
    pp = project2model(v3);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
    if(p.x < -0.5 || p.x > 0.5) p.x = 1;
    gl_Position = p;   
    vertex = vec4(v3.xyz, 1.0);  // w indicates the validity.
    normal = vec4(n0.xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(v1);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v1.xyz, 1.0);
    normal = vec4(n1.xyz, 1.0);
    EmitVertex();

    pp = project2model(v2);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v2.xyz, 1.0);
    normal = vec4(n2.xyz, 1.0);
    EmitVertex();
    EndPrimitive();

    // ====================================================

    pp = project2model(v3);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
    if(p.x < -0.5 || p.x > 0.5) p.x = 1;
    gl_Position = p;   
    vertex = vec4(v3.xyz, 1.0);  // w indicates the validity.
    normal = vec4(n0.xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(v2);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v2.xyz, 1.0);
    normal = vec4(n2.xyz, 1.0);
    EmitVertex();

    pp = project2model(v4);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    p = vec4(2.0 * pp - 1.0, 1.0f);
    if(p.x < -0.5 || p.x > 0.5) p.x = 1;
    gl_Position = p;   // [-1, 1] x [-1, 1]
    vertex = vec4(v4.xyz, 1.0);
    normal = vec4(n0.xyz, 1.0);
    EmitVertex();
    EndPrimitive();
  }
  else
  {
    // normal triangle.
    vec3 pp = project2model(positions[0]);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    gl_Position = vec4(2.0 * pp - 1.0, 1.0f);   // [-1, 1] x [-1, 1]
    vertex = vec4(positions[0].xyz, 1.0);  // w indicates the validity.
    normal = vec4(normals[0].xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(positions[1]);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    gl_Position = vec4(2.0 * pp - 1.0, 1.0f);             // [-1, 1] x [-1, 1]
    vertex = vec4(positions[1].xyz, 1.0);  // w indicates the validity.
    normal = vec4(normals[1].xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(positions[2]);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    gl_Position = vec4(2.0 * pp - 1.0, 1.0f);             // [-1, 1] x [-1, 1]
    vertex = vec4(positions[2].xyz, 1.0);  // w indicates the validity.
    normal = vec4(normals[2].xyz, 1.0);    // w indicates the validity.
    EmitVertex();
    EndPrimitive();
  }
  
}

void split_at_yaxis(vec4[3] positions, vec4[3] normals)
{
  float subdiv_height = 1.0 / max_particles;

    if((positions[0].x < 0 && !(positions[1].x < 0 || positions[2].x < 0)) || 
     (positions[1].x < 0 && !(positions[0].x < 0 || positions[2].x < 0)) || 
     (positions[2].x < 0 && !(positions[0].x < 0 || positions[1].x < 0)))
    {
 
      //  2 ----- 1 
      //  |      /
      //  |     /
      //  |    /
      //--4---3------> y axis
      //  |  / 
      //  | / 
      //  |/  
      //  0   

      int first_index = 0;
      if(positions[1].x < 0) first_index = 1;
      if(positions[2].x < 0) first_index = 2;
      
      vec4 v0 = positions[first_index];
      vec4 v1 = positions[(first_index + 1)%3];
      vec4 v2 = positions[(first_index + 2)%3];
      vec4 n0 = normals[first_index];
      vec4 n1 = normals[(first_index + 1)%3];
      vec4 n2 = normals[(first_index + 2)%3];

      vec4 v3 = yaxis_intersection(v0, v1);
      vec4 v4 = yaxis_intersection(v0, v2);


      vec3 pp;
      vec4 p;

      pp = project2model(v1);  // [0,1] x [0,1]
      pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;
      p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
      gl_Position = p;
      vertex = vec4(v1.xyz, 1.0);  // w indicates the validity.
      normal = vec4(n1.xyz, 1.0);    // w indicates the validity.
      EmitVertex();

      pp = project2model(v2);  // [0,1] x [0,1]
      pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;
      p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
      gl_Position = p;
      vertex = vec4(v2.xyz, 1.0);  // w indicates the validity.
      normal = vec4(n2.xyz, 1.0);    // w indicates the validity.
      EmitVertex();

      pp = project2model(v3);  // [0,1] x [0,1]
      pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;
      p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
      gl_Position = p;
      vertex = vec4(v3.xyz, 1.0);  // w indicates the validity.
      normal = vec4(n1.xyz, 1.0);    // w indicates the validity.
      EmitVertex();
      EndPrimitive();

      pp = project2model(v2);  // [0,1] x [0,1]
      pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;
      p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
      gl_Position = p;
      vertex = vec4(v2.xyz, 1.0);  // w indicates the validity.
      normal = vec4(n2.xyz, 1.0);    // w indicates the validity.
      EmitVertex();

      pp = project2model(v4);  // [0,1] x [0,1]
      pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;
      p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
      gl_Position = p;
      vertex = vec4(v4.xyz, 1.0);  // w indicates the validity.
      normal = vec4(n2.xyz, 1.0);    // w indicates the validity.
      EmitVertex();

      pp = project2model(v3);  // [0,1] x [0,1]
      pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;
      p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
      gl_Position = p;
      vertex = vec4(v3.xyz, 1.0);  // w indicates the validity.
      normal = vec4(n1.xyz, 1.0);    // w indicates the validity.
      EmitVertex();
      EndPrimitive();


      // take care of remaining triangles.
      split_at_xaxis(vec4[3](v0, v3, v4), vec4[3](n0, n1, n2));
    }
    else if((positions[0].x > 0 && !(positions[1].x > 0 || positions[2].x > 0)) || 
     (positions[1].x > 0 && !(positions[0].x > 0 || positions[2].x > 0)) || 
     (positions[2].x > 0 && !(positions[0].x > 0 || positions[1].x > 0)))
    {

      //  0
      //  |\
      //  | \
      //--3--4------> y axis
      //  |   \
      //  |    \
      //  1 --- 2    

      int first_index = 0;
      if(positions[1].x > 0) first_index = 1;
      if(positions[2].x > 0) first_index = 2;
      
      vec4 v0 = positions[first_index];
      vec4 v1 = positions[(first_index + 1)%3];
      vec4 v2 = positions[(first_index + 2)%3];
      vec4 n0 = normals[first_index];
      vec4 n1 = normals[(first_index + 1)%3];
      vec4 n2 = normals[(first_index + 2)%3];

      vec4 v3 = yaxis_intersection(v0, v1);
      vec4 v4 = yaxis_intersection(v0, v2);

      vec3 pp;
      vec4 p;

      pp = project2model(v0);  // [0,1] x [0,1]
      pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;
      p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
      gl_Position = p;
      vertex = vec4(v0.xyz, 1.0);  // w indicates the validity.
      normal = vec4(n0.xyz, 1.0);    // w indicates the validity.
      EmitVertex();

      pp = project2model(v3);  // [0,1] x [0,1]
      pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;
      p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
      gl_Position = p;
      vertex = vec4(v3.xyz, 1.0);  // w indicates the validity.
      normal = vec4(n1.xyz, 1.0);    // w indicates the validity.
      EmitVertex();

      pp = project2model(v4);  // [0,1] x [0,1]
      pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;
      p = vec4(2.0 * pp - 1.0, 1.0f); // [-1, 1] x [-1, 1]
      gl_Position = p;
      vertex = vec4(v4.xyz, 1.0);  // w indicates the validity.
      normal = vec4(n2.xyz, 1.0);    // w indicates the validity.
      EmitVertex();
      EndPrimitive();

      // now we have to check for splits on the x axis:
      split_at_xaxis(vec4[3](v1, v4, v3), vec4[3](n1, n0, n0));
      split_at_xaxis(vec4[3](v1, v2, v4), vec4[3](n1, n2, n0));
    }
    else{
      // check if split at x axis is needed.
      split_at_xaxis(positions, normals);
    }
}


// split given triangle at x axis if needed.
void faster_rendering(vec4[3] positions, vec4[3] normals)
{
  float subdiv_height =  1.0 / max_particles;

  // just check all cases...not prettey but should help to copy with boundary abovementioned issue.
  if((positions[0].y < 0 && !(positions[1].y < 0 || positions[2].y < 0)) ||
     (positions[1].y < 0 && !(positions[0].y < 0 || positions[2].y < 0)) ||
     (positions[2].y < 0 && !(positions[0].y < 0 || positions[1].y < 0)))
  {
    EndPrimitive();
  }
  else if((positions[0].y > 0 && !(positions[1].y > 0 || positions[2].y > 0)) ||
     (positions[1].y > 0 && !(positions[0].y > 0 || positions[2].y > 0)) ||
     (positions[2].y > 0 && !(positions[0].y > 0 || positions[1].y > 0)))
  {
    EndPrimitive();
  }
  else
  {
    // normal triangle.
    vec3 pp = project2model(positions[0]);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    gl_Position = vec4(2.0 * pp - 1.0, 1.0f);   // [-1, 1] x [-1, 1]
    vertex = vec4(positions[0].xyz, 1.0);  // w indicates the validity.
    normal = vec4(normals[0].xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(positions[1]);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    gl_Position = vec4(2.0 * pp - 1.0, 1.0f);             // [-1, 1] x [-1, 1]
    vertex = vec4(positions[1].xyz, 1.0);  // w indicates the validity.
    normal = vec4(normals[1].xyz, 1.0);    // w indicates the validity.
    EmitVertex();

    pp = project2model(positions[2]);  // [0,1] x [0,1]
    pp.y = subdiv_height * pp.y + gs_in[0].instance * subdiv_height;

    gl_Position = vec4(2.0 * pp - 1.0, 1.0f);             // [-1, 1] x [-1, 1]
    vertex = vec4(positions[2].xyz, 1.0);  // w indicates the validity.
    normal = vec4(normals[2].xyz, 1.0);    // w indicates the validity.
    EmitVertex();
    EndPrimitive();
  }
}


void main() {
  // Note: We have to deal with triangles that have vertices that are around y = 0 and therefore it could happend that
  // one vertex has yaw = 290, but the other vertex has yaw = 10...thus it could be that the renderer draws an triangle 
  // from 290 -> 0.

  //nicer rendering
  //split_at_yaxis(vec4[3](gs_in[0].position, gs_in[1].position, gs_in[2].position), vec4[3](gs_in[0].normal, gs_in[1].normal, gs_in[2].normal));
  //split_at_xaxis(vec4[3](gs_in[0].position, gs_in[1].position, gs_in[2].position), vec4[3](gs_in[0].normal, gs_in[1].normal, gs_in[2].normal));

  //faster rendering
  faster_rendering(vec4[3](gs_in[0].position, gs_in[1].position, gs_in[2].position), vec4[3](gs_in[0].normal, gs_in[1].normal, gs_in[2].normal));

}