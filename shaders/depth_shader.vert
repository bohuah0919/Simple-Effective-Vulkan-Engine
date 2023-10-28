#version 460

layout(set = 0, binding = 0) uniform  CameraBuffer{   
    vec3 pos;
	mat4 viewproj; 
    mat4 view; 
} cameraData;

struct ObjectData{
	mat4 model;
    vec4 sphereBound;
}; 

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer{   
	ObjectData objects[];
} objectBuffer;


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec4 fragPosition;

void main() {
    mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
    gl_Position = cameraData.viewproj * modelMatrix * vec4(inPosition, 1.0);

}