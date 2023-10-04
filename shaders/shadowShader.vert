#version 460

layout(set = 0, binding = 0) uniform  CameraBuffer{   
    vec3 pos;
	mat4 viewproj; 
} cameraData;

struct ObjectData{
	mat4 model;
}; 

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer{   
	ObjectData objects[];
} objectBuffer;


layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec4 fragPosition;
void main() {
    mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
    vec4 position = modelMatrix * vec4(inPosition, 1.0);
    position = position / position.w;
    position = cameraData.viewproj * position;
    gl_Position = position;
    fragPosition = position;

}