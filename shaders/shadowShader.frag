#version 450

layout(location = 0) in vec4 fragPosition;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform  SceneData{   
	vec3 lightEmit;
    float zNear;
    float zFar;
} sceneData;

vec4 pack (float depth) {
    vec4 rgbaDepth = fract(depth * vec4(1.0, 255.0, 255.0 * 255.0, 255.0 * 255.0 * 255.0));
    rgbaDepth -= rgbaDepth.yzww * vec4(1.0/255.0, 1.0/255.0, 1.0/255.0, 0.0);
    return rgbaDepth;
}

void main() {
    outColor = pack((fragPosition.z - sceneData.zNear) / (sceneData.zFar - sceneData.zNear));
}