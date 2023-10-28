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
    vec3 lightDir;
    float zNear;
    float zFar;
} sceneData;

layout(set = 3, binding = 0) uniform sampler2D shadowMapSampler;

layout(set = 4, binding = 0) uniform sampler2D tex1;

layout(set = 5, binding = 0) uniform sampler2D gColorSampler;
layout(set = 5, binding = 1) uniform sampler2D gPositionSampler;
layout(set = 5, binding = 2) uniform sampler2D gNormalSampler;
layout(set = 5, binding = 3) uniform sampler2D gVisibilitySampler;
layout(set = 5, binding = 4) uniform sampler2D gDepthSampler;

#define PI 3.141592653589793
#define NUM_SAMPLES 20
#define LIGHT_SIZE 0.002

float rand_1to1(float x) { 
  return fract(sin(x)*10000.0);
}

float rand_2to1(vec2 uv) { 
	return fract(sin(dot(uv.xy, vec2(12.9898,78.233)))* 43758.5453123);
}

float randFloat(inout float seed) { 
    seed = fract(sin(seed)*10000.0);
	return fract(sin(seed)*10000.0);

}

vec2 randVec2(inout float seed) { 
    float x = randFloat(seed);
    float y = randFloat(seed);

	return vec2(x, y);
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

vec3 SampleHemisphereCos(inout float seed, out float pdf) {
  vec2 uv = randVec2(seed);
  float z = sqrt(1.0 - uv.x);
  float r = sqrt(1.0 - z * z);
  float phi = uv.y * PI * 2.0;
  vec3 dir = vec3(r * cos(phi), z, r * sin(phi));
  pdf = z / PI;
  return dir;
}

vec3 SampleHemisphereLobe(inout float seed, out float pdf) {
  vec2 uv = randVec2(seed);
  float z = pow(uv.x, 1 / 12.0);
  float r = sqrt(1.0 - z * z);
  float phi = uv.y * PI * 2;
  vec3 dir = vec3(r * cos(phi), z, r * sin(phi));
  pdf = 12.0 *  pow(dot(normalize(dir),vec3(0.0 ,1.0, 0.0)), 11.0)/ (2.0 * PI);
  return dir;
}

vec3 reflectDir(vec3 wi, vec3 N) {
    return 2.0 * dot(wi, N) * N - wi;
}

float fresnelSchlick( vec3 wo, vec3 N) {
		float cos = dot(wo, N);
		if (cos < 0.1) return 0.0;
		float n1 = 1.0;
		float n2 = 7.5;
		float R0 = pow((n1 - n2) / (n1 + n2), 2.0);
		return R0 + (1 - R0) * pow((1 - cos), 5.0);
	}

vec3 localToWorld(vec3 localDir, vec3 N) {
    vec3 T, B;

    if ((N.x * N.x + N.z * N.z) > 0.0){
        float inv = 1.0 / sqrt(N.x * N.x + N.z * N.z);
        T = inv * vec3(N.z, 0.0f, -N.x);
	    B = cross(N, T);
    }
    else{
        float inv = 1.0 / sqrt(N.y * N.y + N.z * N.z);
	    T = inv * vec3(0.0, N.z, -N.y);
	    B = cross(N, T);
    }

	return localDir.x * T + localDir.y * B + localDir.z * N;
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


bool RayMarch(vec3 ori, vec3 dir, out vec2 hitPosUV, out float visibility) {

      float step_size = 0.2;
      int total_steps = 50;
      for(int i = 0; i < total_steps; i++) {
        vec3 pos = ori + i * step_size * dir;
        vec4 current_coord = cameraData.viewproj * vec4(pos, 1.0);
        vec2 texCoord = current_coord.xy /current_coord.w * 0.5 + 0.5;
        if(texCoord.x < 0.0 || texCoord.x > 1.0 || texCoord.y < 0.0 || texCoord.y > 1.0){
            return false;
        }

        float current_depth = (current_coord.z  - 10.0) / (100.0- 10.0);
        
        float record_depth = unpack(texture(gDepthSampler, texCoord));

        if ((current_depth - record_depth > 0.004) && (current_depth - record_depth < 0.008) && (dot(dir,normalize(texture(gNormalSampler, texCoord).xyz)) < 0.0)){
            hitPosUV = current_coord.xy /current_coord.w * 0.5 + 0.5;
            visibility =  (1.0 - pow(float(i) / float(total_steps - 1), 5.0)) *(1.0 - pow(abs(texCoord.x - 0.5) * 2.0, 2.0)) * (1.0 - pow(abs(texCoord.y - 0.5) * 2.0, 2.0));
            return true;
        }
        else if (current_depth - record_depth > 0.05){
            return false;
        }
        
      }

      return false;
}


vec3 EvalDirectionalLight(vec2 uv) {
  vec4 pos = texture(gPositionSampler, uv);
  vec3 light_vector = normalize(lightPosition.xyz - pos.xyz);
  vec3 normal_vector = normalize(texture(gNormalSampler, uv).xyz); 
  float cos = max(0.0, dot(normal_vector, light_vector));

  vec3 BSDF = texture(gColorSampler, uv).xyz / PI;

  float visibility = texture(gVisibilitySampler, uv).x;

  return sceneData.lightEmit * BSDF * cos * visibility + texture(gColorSampler, uv).xyz * 0.2;

}


void main() {
    vec3 view_vector = normalize(cameraData.pos - fragPosition.xyz);
    vec3 light_vector = normalize(lightPosition.xyz - fragPosition.xyz);
    vec3 normal_vector = normalize(fragNormal); 
    float cos = max(0.0, dot(normal_vector, light_vector));

    float bias = max(0.005 * (1.0 - cos), 0.004); 

    vec3 color, BSDF;
    color = texture(tex1, fragTexCoord).xyz;

    BSDF = color / PI;

    float seed = rand_2to1(gl_FragCoord.xy);

    vec4 coord = lightTexCoord / lightTexCoord.w;
    vec4 shadow_map_coord = vec4(coord.xy * 0.5 + 0.5, (coord.z * lightTexCoord.w - sceneData.zNear) / (sceneData.zFar- sceneData.zNear), coord.w);
    float visibility = PCSS(shadowMapSampler, shadow_map_coord, bias);
    float specular_coff = fresnelSchlick(view_vector, normal_vector);
    //vec3 specular_coff = uKs;

    vec3 half_vector = normalize(light_vector + view_vector);
    float spec_cos = pow(max(dot(half_vector, normal_vector), 0.0), 32.0);
    vec3 diffuse = sceneData.lightEmit * BSDF * cos;
    vec3 specular = sceneData.lightEmit * spec_cos;
    vec3 Le =  (1.0 - specular_coff) *diffuse + specular_coff * specular;
    vec3 L_indir = vec3(0.0);

    float specular_visibility;
    vec2 specular_hitPosUV;
    vec3 specular_dir = reflectDir(view_vector, normal_vector);
    if(RayMarch(fragPosition.xyz, specular_dir, specular_hitPosUV, specular_visibility)){
        L_indir += specular_coff * clamp(EvalDirectionalLight(specular_hitPosUV), vec3(0.0), vec3(1.0)) * specular_visibility;
    }

    Le += L_indir;
    outColor = vec4(Le * visibility + color * 0.2, 1.0);
    //float shadowMapDepth = unpack(texture(shadowMapSampler, shadow_map_coord.xy));
    //outColor = vec4(shadowMapDepth);
}