#pragma once

#include "vk_types.h"
#include <vector>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <cstring>
#include <iostream>

struct VertexInputDescription {
	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPipelineVertexInputStateCreateFlags flags = 0;
};


struct Vertex {

	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 uv;

	static VertexInputDescription get_vertex_description();
};

struct Mesh {
	std::vector<Vertex> _vertices;
	std::string name;
	AllocatedBuffer _vertexBuffer;
	glm::vec4 sphereBound;
	bool load_from_obj(const char* filename);
};

struct Meshes {
	std::vector<Mesh> _meshes;

	bool load_from_obj(const char* filename);
};
