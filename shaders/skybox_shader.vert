#version 460

layout(set = 0, binding = 0) uniform  CameraBuffer{   
    vec3 pos;
	mat4 viewproj; 
} cameraData;


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    vec4 position = vec4(inPosition, 1.0);
    position = position / position.w;
    position = cameraData.viewproj * position;
    gl_Position = position.xyww;

    fragTexCoord = inTexCoord;

}