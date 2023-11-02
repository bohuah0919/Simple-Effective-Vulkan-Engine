#version 450

#define SHADOW_MAP_CASCADE_COUNT 4
const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 
);

layout(location = 0) in vec4 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in vec4 inViewPos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform  CameraBuffer{   
    vec3 pos;
	mat4 viewproj; 
    mat4 view; 
} cameraData;

layout (set = 1, binding = 0) uniform csmBuffer{
	mat4 cascadeSplits;
	mat4 cascadeViewProjMat[SHADOW_MAP_CASCADE_COUNT];
    mat4 frustumSizes;
} csmData;

layout(set = 1, binding = 1) uniform  SceneData{   
	vec3 lightEmit;
    vec3 lightDir;
    float zNear;
    float zFar;
} sceneData;

layout(set = 3, binding = 0) uniform sampler2DArray shadowMapSampler;

layout(set = 4, binding = 0) uniform sampler2D tex1;

#define PI 3.141592653589793
#define NUM_SAMPLES 20
#define LIGHT_SIZE 0.004

float rand_1to1(float x) { 
  return fract(sin(x)*10000.0);
}

float rand_2to1(vec2 uv) { 
	return fract(sin(dot(uv.xy, vec2(12.9898,78.233)))* 43758.5453123);
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

float findAvgBlockerDepth(sampler2DArray shadowMapSampler, vec2 uv, float dReceiver, uint cascadeIndex) {
	uniformDiskSamples(uv);

    int blockerNum = 0;
    float depth = 0.0;

    float radius = LIGHT_SIZE * (dReceiver - sceneData.zNear) / dReceiver;

    for(int i = 0; i < NUM_SAMPLES; i++) {
        vec2 offset = radius * sampleDisk[i];   
        float shadowMapDepth = texture(shadowMapSampler, vec3(uv + offset,cascadeIndex)).r;   

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

float sampleShadowMap(sampler2DArray shadowMapSampler, vec4 shadow_map_coord, uint cascadeIndex,float bias){
    float visibility = 1.0;

    if (texture(shadowMapSampler, vec3(shadow_map_coord.xy, cascadeIndex)).r < (shadow_map_coord.z - bias))
        visibility = 0.0;

    return visibility;
}
float filterPCF(vec4 shadow_map_coord, uint cascadeIndex, float bias)
{
	ivec2 texDim = textureSize(shadowMapSampler, 0).xy;
	float scale = 1.5;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;
	
	for (int x = -range; x <= range; x++) {
		for (int y = -range; y <= range; y++) {
            vec4 offset = vec4(vec2(dx*x, dy*y), 0.0, 0.0);
			shadowFactor += sampleShadowMap(shadowMapSampler, shadow_map_coord + offset, cascadeIndex, bias);
			count++;
		}
	}
	return shadowFactor / count;
}

float PCF(sampler2DArray shadowMapSampler, vec4 shadow_map_coord, float radius, uint cascadeIndex, float bias) {
    uniformDiskSamples(shadow_map_coord.xy);
    float visibility = 0.0;

    for(int i = 0; i < NUM_SAMPLES; i++) {
        vec4 offset = vec4(radius * sampleDisk[i], 0.0, 0.0);
        visibility += sampleShadowMap(shadowMapSampler, shadow_map_coord + offset, cascadeIndex, bias);

    }

    return visibility / NUM_SAMPLES;
}

float PCSS(sampler2DArray shadowMapSampler, vec4 shadow_map_coord, float dReceiver, uint cascadeIndex, float bias){
  float dBlock = findAvgBlockerDepth(shadowMapSampler, shadow_map_coord.xy, dReceiver, cascadeIndex);
  if (dBlock == -1.0)
    return 1.0;

  float wPenumbra = (dReceiver - dBlock) * LIGHT_SIZE / (dBlock);
  float visibility =  PCF(shadowMapSampler, shadow_map_coord, wPenumbra, cascadeIndex, bias);

  return visibility;
}



void main() {
    vec3 color = texture(tex1, fragTexCoord).xyz;
    vec3 BSDF = color / PI;
    vec3 light_vector = sceneData.lightDir;
    vec3 normal_vector = normalize(fragNormal); 
    vec3 view_vector = normalize(cameraData.pos - fragPosition.xyz);
    float cos = max(0.0, dot(normal_vector, light_vector));

    float bias = max(0.005 * (1.0 - cos), 0.004);
    bias = max(0.001* (1.0 - cos), 0.001);

    uint cascadeIndex = 0;

	for(uint i = 0; i < SHADOW_MAP_CASCADE_COUNT - 1; ++i) {
        uint rowIndex = i / 4;
		uint colIndex = i % 4;
		if(inViewPos.z < csmData.cascadeSplits[rowIndex][colIndex]) {	
			cascadeIndex = i + 1; 
		}
	}
    

    //cascadeIndex = 6;

    //bias *= -10.0/(inViewPos.z);

    vec4 coord = biasMat* csmData.cascadeViewProjMat[cascadeIndex] * fragPosition;
    vec4 shadow_map_coord = coord / coord.w;

    float visibility;
    
    visibility = PCSS(shadowMapSampler, shadow_map_coord, coord.z, cascadeIndex ,bias);
    //visibility = filterPCF(shadow_map_coord, cascadeIndex, bias);

    vec3 half_vector = normalize(light_vector + view_vector);
    vec3 diffuse = sceneData.lightEmit * BSDF * cos;
    vec3 Le =  diffuse;

    outColor = vec4(Le * visibility + color * 0.5, 1.0);

    //vec4 testColor = vec4(0.0);
    //testColor[cascadeIndex%4] += 0.25;
    //outColor = testColor;
}