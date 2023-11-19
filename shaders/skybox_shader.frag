#version 450

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(set = 4, binding = 0) uniform sampler2D tex1;

void main() {
    vec3 color = texture(tex1, fragTexCoord).xyz;

    outColor = vec4(color, 1.0);
    //outColor = vec4(1.0);
}