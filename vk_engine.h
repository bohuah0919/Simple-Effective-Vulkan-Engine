#pragma once
#include "vk_descriptors.h"
#include "vk_types.h"
#include <vector>
#include <deque>
#include <functional>
#include "Mesh.h"
#include <unordered_map>
#include <string>

class PipelineBuilder {
public:

	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	std::vector<VkPipelineColorBlendAttachmentState> _colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;
	VkPipelineDepthStencilStateCreateInfo _depthStencil;
	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass, int size);
	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};



struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); 
		}

		deletors.clear();
	}
};

struct MeshPushConstants {
	glm::vec4 data;
	glm::mat4 render_matrix;
};


struct Material {
	VkDescriptorSet textureSet{ VK_NULL_HANDLE };
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct Texture {
	AllocatedImage image;
	VkImageView imageView;
};

struct RenderObject {
	Mesh* mesh;

	Material* material;

	glm::mat4 transformMatrix;


};

struct RenderObjects {
	std::vector<RenderObject*> RenderObjects;

};


struct FrameData {
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	DeletionQueue _frameDeletionQueue;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkCommandPool _shadowCommandPool;
	VkCommandBuffer _shadowCommandBuffer;

	VkCommandPool _gBufferCommandPool;
	VkCommandBuffer _gBufferCommandBuffer;

	AllocatedBuffer cameraBuffer;
	VkDescriptorSet globalDescriptor;

	AllocatedBuffer objectBuffer;
	VkDescriptorSet objectDescriptor;

	AllocatedBuffer lightBuffer;
	VkDescriptorSet lightDescriptor;

};

struct UploadContext {
	VkFence _uploadFence;
	VkCommandPool _commandPool;
	VkCommandBuffer _commandBuffer;
};
struct GPUCameraData {
	alignas(16) glm::vec3 pos;
	alignas(16) glm::mat4 viewproj;
};


struct GPUSceneData {
	alignas(16) glm::vec3 lightColor;
	alignas(4) float zNear;
	alignas(4) float zFar;
};

struct GPUObjectData {
	alignas(16) glm::mat4 modelMatrix;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber{ 0 };
	int _selectedShader{ 0 };

	VkExtent2D _windowExtent{ 1700, 900 };
	VkExtent2D _shadowExtent{ 4096, 4096 };
	VkExtent2D _gBufferExtent{ 512, 512};

	GLFWwindow* _window{ nullptr };

	glm::vec3 _camPos = { 0.0f, 10.0f, 30.0f };
	glm::vec3 _oriFoc = { 0.0f, 10.0f, 0.0f };
	glm::vec3 _foc = { 0.0f, 10.0f, 0.0f };
	glm::vec2 _posPress;
	float _verAngle = 0.0f;
	float _horAngle = 0.0f;
	float _verAngleOffset = 0.0f;
	float _horAngleOffset = 0.0f;

	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;

	VkPhysicalDeviceProperties _gpuProperties;

	FrameData _frames[FRAME_OVERLAP];

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	VkRenderPass _renderPass;
	VkRenderPass _gBufferPass;
	VkRenderPass _shadowPass;
	VkDescriptorSet _shadowMap;
	VkDescriptorSet _gBuffer;

	VkSurfaceKHR _surface;
	VkSwapchainKHR _swapchain;
	VkFormat _swachainImageFormat;

	std::vector<VkFramebuffer> _framebuffers;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;

	VkFormat _shadowMapFormat;
	VkFramebuffer _shadowFramebuffer;
	VkSampler _shadowSampler;
	AllocatedImage _shadowColorImage;
	VkImageView _shadowColorImageView;
	AllocatedImage _shadowDepthImage;
	VkImageView _shadowDepthImageView;

	VkFramebuffer _gBufferFramebuffer;
	AllocatedImage _gBufferPosImage;
	VkImageView _gBufferPosImageView;
	AllocatedImage _gBufferColorImage;
	VkImageView _gBufferColorImageView;
	AllocatedImage _gBufferNormalImage;
	VkImageView _gBufferNormalImageView;
	AllocatedImage _gBufferVisibilityImage;
	VkImageView _gBufferVisibilityImageView;
	AllocatedImage _gBufferDepthRGBAImage;
	VkImageView _gBufferDepthRGBAImageView;
	AllocatedImage _gBufferDepthImage;
	VkImageView _gBufferDepthImageView;

	VkPipelineLayout _trianglePipelineLayout;

	VkPipeline _trianglePipeline;
	VkPipeline _redTrianglePipeline;

	DeletionQueue _mainDeletionQueue;

	VkPipeline _meshPipeline;
	Mesh _triangleMesh;
	Mesh _monkeyMesh;

	VkPipelineLayout _meshPipelineLayout;

	VmaAllocator _allocator;

	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	VkFormat _depthFormat;

	vkutil::DescriptorLayoutCache* _descriptorLayoutCache;
	vkutil::DescriptorAllocator* _descriptorAllocator;


	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;
	VkDescriptorSetLayout _singleTextureSetLayout;
	VkDescriptorSetLayout _lightSetLayout;
	VkDescriptorSetLayout _gBufferSetLayout;
	GPUSceneData _sceneParameters;
	AllocatedBuffer _sceneParameterBuffer;

	UploadContext _uploadContext;

	bool framebufferResized = false;

	void init();

	void cleanup();

	void draw();

	void run();

	FrameData& get_current_frame();
	FrameData& get_last_frame();

	std::vector<RenderObject> _renderables;

	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;
	std::unordered_map<std::string, Texture> _loadedTextures;

	Material* create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);

	Material* get_material(const std::string& name);

	Mesh* get_mesh(const std::string& name);

	void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);

	void draw_gbuffer(VkCommandBuffer cmd, RenderObject* first, int count);

	void draw_shadow(VkCommandBuffer cmd, RenderObject* first, int count);

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

	size_t pad_uniform_buffer_size(size_t originalSize);

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

private:

	void init_vulkan();

	void init_swapchain();

	void init_shadow_renderpass();

	void init_gbuffer_renderpass();

	void init_default_renderpass();

	void init_framebuffers();

	void init_commands();

	void init_sync_structures();

	void init_pipelines();

	void init_scene();

	void init_descriptors();

	bool load_shader_module(const char* filePath, VkShaderModule* outShaderModule);

	void load_meshes();

	void load_images();

	void upload_mesh(Mesh& mesh);

	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);

	static void cursor_enter_callback(GLFWwindow* window, int entered);

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
}; 

