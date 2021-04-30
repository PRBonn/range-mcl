#version 330

layout(location = 0) in vec4 position_in;
layout(location = 1) in vec4 normal_in;

out VS_OUT {
  vec4 position;
  vec4 normal;
  int instance;
}
vs_out;

// uniform mat4 inv_pose;  // the inverse of the desired pose in global coordiantes.
uniform mat4 model_pose;  // pose of the mesh in global coordinate system
uniform samplerBuffer poseBuffer;

mat4 get_pose(int instance)
{
  int offset = 4 * instance;
  // declaration column-wise
  return mat4(texelFetch(poseBuffer, offset), texelFetch(poseBuffer, offset + 1),
              texelFetch(poseBuffer, offset + 2), texelFetch(poseBuffer, offset+3));
}


void main() {
  mat4 inv_pose =  get_pose(gl_InstanceID);
  vs_out.position = inv_pose * model_pose * vec4(position_in.xyz, 1.0);
  vs_out.normal = inv_pose * model_pose *
           vec4(normal_in.xyz, 0.0);  // this is fine, since we have no scaling.
  vs_out.instance = gl_InstanceID;


}