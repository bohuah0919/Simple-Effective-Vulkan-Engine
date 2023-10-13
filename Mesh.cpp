#include "Mesh.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <iostream>
#include <unordered_map>
const std::unordered_map<std::string, glm::vec3> KdMap = {
	{"Glass_border_Cube.005", {1.0f, 1.0f, 1.0f} },
		{"Glass_Cube.004", {0.0f, 0.0f, 0.0f}},
		{"Table_Leg.001_Cube.009", {1.0f, 1.0f, 1.0f}},
		{"Table_Leg.002_Cube.010", {1.0f, 1.0f, 1.0f}},
		{"Table_Leg.003_Cube.006",  {1.0f, 1.0f, 1.0f}},
		{"Table_Leg.004_Cube.011", {1.0f, 1.0f, 1.0f}},
		{"Chair_Cube.007", {1.0f, 1.0f, 1.0f}},
		{"Seat_Cube.001", {0.8, 0.0f, 0.0f}},
		{"BackRest_Cube.008", {0.8f, 0.0f, 0.0f}},
		{"Chair.001_Cube.015", {1.0f, 1.0f, 1.0f}},
		{"Seat.001_Cube.016", {0.8f, 0.0f, 0.0f}},
		{"BackRest.001_Cube.017", {0.8f, 0.0f, 0.0f}},
		{"Chair.002_Cube.018", {1.0f, 1.0f, 1.0f}},
		{"Seat.002_Cube.019", {0.8f, 0.0f, 0.0f}},
		{"BackRest.002_Cube.020", {0.8f, 0.0f, 0.0f}},
		{"Chair.003_Cube.022", {1.0f, 1.0f, 1.0f}},
		{"BackRest.003_Cube.023", {0.8f, 0.0f, 0.0f}},
		{"Seat.003_Cube.021", {0.8f, 0.0f, 0.0f}},
		{"Plane_Plane.002", {1.0f, 1.0f, 1.0f}},
		{"Plane.001", {0.7f, 0.7f, 0.7f}},
		{"Plane.002_Plane.003", {0.7f, 0.7f, 0.7f}},
};

VertexInputDescription Vertex::get_vertex_description()
{
	VertexInputDescription description;
	VkVertexInputBindingDescription mainBinding = {};
	mainBinding.binding = 0;
	mainBinding.stride = sizeof(Vertex);
	mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	description.bindings.push_back(mainBinding);

	VkVertexInputAttributeDescription positionAttribute = {};
	positionAttribute.binding = 0;
	positionAttribute.location = 0;
	positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttribute.offset = offsetof(Vertex, position);

	VkVertexInputAttributeDescription normalAttribute = {};
	normalAttribute.binding = 0;
	normalAttribute.location = 1;
	normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	normalAttribute.offset = offsetof(Vertex, normal);

	VkVertexInputAttributeDescription colorAttribute = {};
	colorAttribute.binding = 0;
	colorAttribute.location = 2;
	colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	colorAttribute.offset = offsetof(Vertex, color);

	VkVertexInputAttributeDescription uvAttribute = {};
	uvAttribute.binding = 0;
	uvAttribute.location = 3;
	uvAttribute.format = VK_FORMAT_R32G32_SFLOAT;
	uvAttribute.offset = offsetof(Vertex, uv);

	description.attributes.push_back(positionAttribute);
	description.attributes.push_back(normalAttribute);
	description.attributes.push_back(colorAttribute);
	description.attributes.push_back(uvAttribute);
	return description;
}

bool Mesh::load_from_obj(const char* filename)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;

	tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename,
		nullptr);

	if (!err.empty()) {
		std::cerr << err << std::endl;
		return false;
	}

	for (size_t s = 0; s < shapes.size(); s++) {
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

			int fv = 3;

			for (size_t v = 0; v < fv; v++) {

				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

				tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
				tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
				tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

				tinyobj::real_t ux = attrib.texcoords[2 * idx.texcoord_index + 0];
				tinyobj::real_t uy = attrib.texcoords[2 * idx.texcoord_index + 1];

				Vertex new_vert;
				new_vert.position.x = vx;
				new_vert.position.y = vy;
				new_vert.position.z = vz;

				new_vert.normal.x = nx;
				new_vert.normal.y = ny;
				new_vert.normal.z = nz;

				new_vert.uv.x = ux;
				new_vert.uv.y = 1 - uy;

				new_vert.color = new_vert.normal;


				_vertices.push_back(new_vert);
			}
			index_offset += fv;
		}
	}

	return true;
}

bool Meshes::load_from_obj(const char* filename)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;

	tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename,
		nullptr);

	if (!err.empty()) {
		std::cerr << err << std::endl;
		return false;
	}

	for (size_t s = 0; s < shapes.size(); s++) {
		Mesh _mesh;
		glm::vec3 maxP = { -std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity() };
		glm::vec3 minP = {std::numeric_limits<float>::infinity(),
			std::numeric_limits<float>::infinity(),
			std::numeric_limits<float>::infinity() };

		_mesh.name = shapes[s].name;
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

			int fv = 3;

			for (size_t v = 0; v < fv; v++) {

				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

				tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
				tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
				tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

				tinyobj::real_t ux = attrib.texcoords[2 * idx.texcoord_index + 0];
				tinyobj::real_t uy = attrib.texcoords[2 * idx.texcoord_index + 1];

				Vertex new_vert;
				new_vert.position.x = vx;
				new_vert.position.y = vy;
				new_vert.position.z = vz;

				new_vert.normal.x = nx;
				new_vert.normal.y = ny;
				new_vert.normal.z = nz;

				new_vert.uv.x = ux;
				new_vert.uv.y = 1 - uy;

				new_vert.color = KdMap.at(shapes[s].name);

				maxP[0] = std::max(maxP[0], new_vert.position.x);
				maxP[1] = std::max(maxP[1], new_vert.position.y);
				maxP[2] = std::max(maxP[2], new_vert.position.z);

				minP[0] = std::min(minP[0], new_vert.position.x);
				minP[1] = std::min(minP[1], new_vert.position.y);
				minP[2] = std::min(minP[2], new_vert.position.z);

				glm::vec3 center = (minP + maxP) / 2.0f;
				float radius = glm::length(maxP - minP) / 2.0f;
				_mesh.sphereBound = { center,radius };

				_mesh._vertices.push_back(new_vert);
			}
			index_offset += fv;
		}
		_meshes.push_back(_mesh);
	}

	return true;
}