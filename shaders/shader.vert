#version 460

layout(set = 0, binding = 0) uniform  CameraBuffer{   
    vec3 pos;
	mat4 viewproj; 
} cameraData;

layout(set = 1, binding = 0) uniform  lightBuffer{   
    vec3 pos;
	mat4 viewproj; 
} lightData;

layout(set = 1, binding = 1) uniform  SceneData{   
	vec3 lightEmit;
    float zNear;
    float zFar;
} sceneData;

struct ObjectData{
	mat4 model;
}; 

layout(std140, set = 2, binding = 0) readonly buffer ObjectBuffer{   
	ObjectData objects[];
} objectBuffer;


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec4 fragPosition;
layout(location = 1) out vec3 fragColor;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec2 fragTexCoord;
layout(location = 4) out vec4 lightPosition;
layout(location = 5) out vec4 lightTexCoord;

void main() {
    mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
    vec4 position = modelMatrix * vec4(inPosition, 1.0);
    position = position / position.w;
    vec4 normal = modelMatrix * vec4(inNormal, 1.0);
    normal = normal / normal.w;
    vec4 light_pos = vec4(lightData.pos, 1.0);
    gl_Position = cameraData.viewproj * position;
    //gl_Position = lightData.viewproj * position;

    fragPosition = position;
    fragColor = inColor;
    fragNormal = normal.xyz;
    fragTexCoord = inTexCoord;
    lightPosition = light_pos;
    lightTexCoord = lightData.viewproj * position;
}