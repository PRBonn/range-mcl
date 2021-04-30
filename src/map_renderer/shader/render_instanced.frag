#version 330 core

in vec4 vertex;
in vec4 normal;

layout (location = 0) out vec4 vertexmap;
layout (location = 1) out vec4 normalmap;
layout (location = 2) out float depthmap;

void main()
{
 if(vertex.w == 0) discard;
 depthmap = length(vertex.xyz); // euclidean distance.
 if(depthmap < 2 || depthmap > 50) discard;
 vertexmap = vertex;
 normalmap = normal;
}