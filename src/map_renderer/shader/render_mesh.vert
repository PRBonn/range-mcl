#version 330

layout(location = 0) in vec4 position_in;
layout(location = 1) in vec4 normal_in;

out VS_OUT {
  vec4 position;
  vec4 normal;
}
vs_out;

uniform mat4 inv_pose;  // the inverse of the desired pose in global coordiantes.
uniform mat4 model_pose;  // pose of the mesh in global coordinate system


void main() {
  vs_out.position = inv_pose * model_pose * vec4(position_in.xyz, 1.0);
  vs_out.normal = inv_pose * model_pose *
           vec4(normal_in.xyz, 0.0);  // this is fine, since we have no scaling.


}