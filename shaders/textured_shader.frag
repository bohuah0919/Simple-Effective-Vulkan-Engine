#version 450

layout(location = 0) in vec4 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in vec4 lightPosition;
layout(location = 5) in vec4 lightTexCoord;


layout(location = 0) out vec4 outColor;

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

float rand_1to1(float x) { 
  return fract(sin(x)*10000.0);
}

float rand_2to1(vec2 uv) { 
	return fract(sin(dot(uv.xy, vec2(12.9898,78.233)))* 43758.5453123);
}

float unpack(vec4 rgbaDepth) {
    return dot(rgbaDepth, vec4(1.0, 1.0/255.0, 1.0/(255.0*255.0), 1.0/(256.0*255.0*255.0)));
}

vec2 sampleDisk[NUM_SAMPLES];

void uniformDiskSamples(vec2 randomSeed) {
  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1(randNum) ;
  float sampleY = rand_1to1(sampleX) ;

  float angle = sampleX * 2.0 * PI;
  float radius = sqrt(sampleY);

  for(int i = 0; i < NUM_SAMPLES; i++) {
    sampleDisk[i] = vec2(radius * cos(angle) , radius * sin(angle));

    sampleX = rand_1to1(sampleY) ;
    sampleY = rand_1to1(sampleX) ;

    angle = sampleX * 2.0 * PI;
    radius = sqrt(sampleY);
  }
}

float findAvgBlockerDepth(sampler2D shadowMapSampler, vec2 uv, float dReceiver) {
	uniformDiskSamples(uv);

    int blockerNum = 0;
    float depth = 0.0;

    float radius = LIGHT_SIZE * (lightTexCoord.z - sceneData.zNear) / lightTexCoord.z;

    for(int i = 0; i < NUM_SAMPLES; i++) {
        vec2 offset = radius * sampleDisk[i];   
        float shadowMapDepth = unpack(texture(shadowMapSampler, uv + offset));   

        if(shadowMapDepth < dReceiver){
            blockerNum++;
            depth += shadowMapDepth;
        }
    }

    if (blockerNum == 0)
        return -1.0;
    else 
        return depth / blockerNum;
}

float sampleShadowMap(sampler2D shadowMapSampler, vec4 shadow_map_coord, float bias){
    float visibility = 1.0;

    if (unpack(texture(shadowMapSampler, shadow_map_coord.xy)) < (shadow_map_coord.z - bias))
        visibility = 0.3;

    return visibility;
}

float PCF(sampler2D shadowMapSampler, vec4 shadow_map_coord, float radius, float bias) {
    uniformDiskSamples(shadow_map_coord.xy);
    float visibility = 0.0;

    for(int i = 0; i < NUM_SAMPLES; i++) {
        vec4 offset = vec4(radius * sampleDisk[i], 0.0, 0.0);
        visibility += sampleShadowMap(shadowMapSampler, shadow_map_coord + offset, bias);
    }

    return visibility / NUM_SAMPLES;
}

float PCSS(sampler2D shadowMapSampler, vec4 shadow_map_coord, float bias){
  float dReceiver = (lightTexCoord.z - sceneData.zNear) / (sceneData.zFar- sceneData.zNear);
  float dBlock = findAvgBlockerDepth(shadowMapSampler, shadow_map_coord.xy, dReceiver);
  if (dBlock == -1.0)
    return 1.0;

  float wPenumbra = (dReceiver - dBlock) * LIGHT_SIZE * sceneData.zNear / lightTexCoord.z / dBlock;
  float visibility =  PCF(shadowMapSampler, shadow_map_coord, wPenumbra, bias);

  return visibility;
}

vec3 blinnPhong(){
    vec3 color = texture(tex1, fragTexCoord).xyz;
    vec3 ambient = 0.05 * color;

    vec3 light_coff = sceneData.lightEmit;
    vec3 light_vector = normalize(lightPosition.xyz - fragPosition.xyz);
    vec3 normal_vector = normalize(fragNormal); 
    float cos = max(0.0, dot(normal_vector, light_vector));
    vec3 diffuse = color * light_coff  * cos;

    vec3 view_vector = normalize(cameraData.pos - fragPosition.xyz);
    vec3 half_vector = normalize(light_vector + view_vector);
    float spec_cos = pow(max(dot(half_vector, normal_vector), 0.0), 32.0);
    vec3 specular = 0.5 * light_coff * spec_cos;
    return (diffuse + ambient + specular);
}

void main() {
    vec3 color = texture(tex1, fragTexCoord).xyz;
    vec3 BSDF = color / PI;
    vec3 light_vector = normalize(lightPosition.xyz - fragPosition.xyz);
    vec3 normal_vector = normalize(fragNormal); 
    vec3 view_vector = normalize(cameraData.pos - fragPosition.xyz);
    float cos = max(0.0, dot(normal_vector, light_vector));

    float bias = max(0.005 * (1.0 - cos), 0.004); 

    vec4 coord = lightTexCoord / lightTexCoord.w;
    vec4 shadow_map_coord = vec4(coord.xy * 0.5 + 0.5, (coord.z * lightTexCoord.w - sceneData.zNear) / (sceneData.zFar- sceneData.zNear), coord.w);

    float visibility = PCSS(shadowMapSampler, shadow_map_coord ,bias);
    vec3 half_vector = normalize(light_vector + view_vector);
    vec3 diffuse = sceneData.lightEmit * BSDF * cos;
    vec3 Le =  diffuse;

    outColor = vec4(Le * visibility + color * 0.2, 1.0);
    //float shadowMapDepth = unpack(texture(shadowMapSampler, shadow_map_coord.xy));
    //outColor = vec4(shadowMapDepth);
}