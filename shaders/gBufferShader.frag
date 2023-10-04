#version 450

layout(location = 0) in vec4 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in vec4 lightPosition;
layout(location = 5) in vec4 lightTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outPosition;
layout(location = 2) out vec4 outNormal;
layout(location = 3) out vec4 outVisibility;
layout(location = 4) out vec4 outDepth;

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

layout(set = 3, binding = 0) uniform sampler2D shadowMapSampler;

layout(set = 4, binding = 0) uniform sampler2D tex1;

#define PI 3.141592653589793
#define NUM_SAMPLES 150
#define LIGHT_SIZE 0.004

vec4 pack (float depth) {
    vec4 rgbaDepth = fract(depth * vec4(1.0, 255.0, 255.0 * 255.0, 255.0 * 255.0 * 255.0));
    rgbaDepth -= rgbaDepth.yzww * vec4(1.0/255.0, 1.0/255.0, 1.0/255.0, 0.0);
    return rgbaDepth;
}

float unpack(vec4 rgbaDepth) {
    return dot(rgbaDepth, vec4(1.0, 1.0/255.0, 1.0/(255.0*255.0), 1.0/(256.0*255.0*255.0)));
}

float sampleShadowMap(sampler2D shadowMapSampler, vec4 shadow_map_coord, float bias){
    float visibility = 1.0;

    if (unpack(texture(shadowMapSampler, shadow_map_coord.xy)) < (shadow_map_coord.z - bias))
        visibility = 0.3;

    return visibility;
}


void main() {
    outColor = texture(tex1, fragTexCoord);

    outPosition = fragPosition;
    outNormal = vec4(normalize(fragNormal), 1.0);   

    vec4 coord = lightTexCoord / lightTexCoord.w;

    vec3 light_vector = normalize(lightPosition.xyz - fragPosition.xyz);
    vec3 normal_vector = normalize(fragNormal); 
    float cos = max(0.0, dot(normal_vector, light_vector));
    float bias = max(0.005 * (1.0 - cos), 0.004); 

    vec4 shadow_map_coord = vec4(coord.xy * 0.5 + 0.5, (coord.z * lightTexCoord.w - sceneData.zNear) / (sceneData.zFar- sceneData.zNear), coord.w);
    float visibility = sampleShadowMap(shadowMapSampler, shadow_map_coord, bias);

    outVisibility = vec4(vec3(visibility), 1.0); 

    vec4 screenPosition = cameraData.viewproj * fragPosition;
    outDepth = pack((screenPosition.z - 10.0) / (100.0- 10.0));

}