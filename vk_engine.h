#pragma once
#include "vk_descriptors.h"
#include "vk_types.h"
#include <vector>
#include <deque>
#include <functional>
#include "Mesh.h"
#include <unordered_map>
#include <string>

constexpr unsigned int FRAME_OVERLAP = 2;
constexpr unsigned int SHADOW_MAP_CASCADE_COUNT = 4;

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

class ComputePipelineBuilder {
public:

	VkPipelineShaderStageCreateInfo  _shaderStage;
	VkPipelineLayout _pipelineLayout;
	VkPipeline build_pipeline(VkDevice device);
};


struct Cascade {
	VkFramebuffer frameBuffer;
	VkDescriptorSet descriptorSet;
	VkImageView view;
	
	float radius;
	float splitDepth;
	glm::mat4 viewProjMatrix;

	void destroy(VkDevice device) {
		vkDestroyImageView(device, view, nullptr);
		vkDestroyFramebuffer(device, frameBuffer, nullptr);
	}
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

struct Material {
	VkDescriptorSet textureSet{ VK_NULL_HANDLE };
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct IndirectBatch {
	Mesh* mesh;
	Material* material;
	uint32_t first;
	uint32_t count;
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

	VkCommandPool _cullCommandPool;
	VkCommandBuffer _cullCommandBuffer;

	VkCommandPool _cullShadowCommandPool;
	VkCommandBuffer _cullShadowCommandBuffer;

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

	std::array<Cascade, SHADOW_MAP_CASCADE_COUNT> cascades;
	std::array<AllocatedBuffer, SHADOW_MAP_CASCADE_COUNT> cascadesBuffers;

	AllocatedBuffer cascadesSetBuffer;
	VkDescriptorSet cascadesSetDescriptor;

	AllocatedBuffer instanceBuffer;
	VkDescriptorSet cullDescriptor;
	std::array<VkDescriptorSet, SHADOW_MAP_CASCADE_COUNT> cullCascadeDescriptors;

	AllocatedBuffer indirectBuffer;
	std::array<AllocatedBuffer, SHADOW_MAP_CASCADE_COUNT> indirectShadowBuffers;
};

struct Camera {
	glm::vec3 _camPos = { 0.0f, 2.0f, 30.0f };
	glm::vec3 _oriFoc = { 0.0f, 2.0f, 0.0f };
	glm::vec3 _foc = { 0.0f, 2.0f, 0.0f };
	glm::vec2 _posPress;
	float _verAngle = 0.0f;
	float _horAngle = 0.0f;
	float _verAngleOffset = 0.0f;
	float _horAngleOffset = 0.0f;
	glm::mat4 viewproj;
	float zNear;
	float zFar;
};

struct CascadesSet {
	alignas(4) glm::mat4 cascadeSplits;
	alignas(16) glm::mat4 cascadeViewProjMat[SHADOW_MAP_CASCADE_COUNT];
	alignas(16) glm::mat4 cascadeSizes;
};

struct UploadContext {
	VkFence _uploadFence;
	VkCommandPool _commandPool;
	VkCommandBuffer _commandBuffer;
};
struct GPUCameraData {
	alignas(16) glm::vec3 pos;
	alignas(16) glm::mat4 viewproj;
	alignas(16) glm::mat4 view;
};


struct GPUSceneData {
	alignas(16) glm::vec3 lightColor;
	alignas(16) glm::vec3 lightDir;
	alignas(4) float zNear;
	alignas(4) float zFar;
};

struct GPUObjectData {
	alignas(16) glm::mat4 modelMatrix;
	alignas(16) glm::vec4 sphereBound;
};

struct GPUInstance {
	alignas(4) uint32_t objectID;
};

struct CullConstants {
	alignas(16) glm::mat4 view;
	alignas(16) glm::vec4 frustum;
	alignas(4) float distance;
	alignas(4) float znear;
	alignas(4) float zfar;
	alignas(4) uint32_t count;
};

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber{ 0 };
	int _selectedShader{ 0 };

	VkExtent2D _windowExtent{ 1700, 900 };
	VkExtent2D _shadowExtent{ 2048, 2048 };
	VkExtent2D _gBufferExtent{ 512, 512};

	GLFWwindow* _window{ nullptr };

	glm::vec3 _lightPos = { -120.0f,140.0f,80.0f };
	glm::vec3 _lightFoc = { 0.0f, 0.0f, 0.0f };
	Camera _camera;

	VkSampleCountFlagBits _msaaSamples = VK_SAMPLE_COUNT_8_BIT;

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
	VkRenderPass _depthPass;
	VkDescriptorSet _shadowMap;
	VkDescriptorSet _csmSet;
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
	AllocatedImage _depth;
	VkImageView _depthView;

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

	VkImageView _colorImageView;
	AllocatedImage _colorImage;

	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	VkFormat _depthFormat;

	vkutil::DescriptorLayoutCache* _descriptorLayoutCache;
	vkutil::DescriptorAllocator* _descriptorAllocator;


	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;
	VkDescriptorSetLayout _singleTextureSetLayout;
	VkDescriptorSetLayout _csmSetLayout;
	VkDescriptorSetLayout _lightSetLayout;
	VkDescriptorSetLayout _cascadesSetLayout;
	VkDescriptorSetLayout _gBufferSetLayout;
	GPUSceneData _sceneParameters;
	AllocatedBuffer _sceneParameterBuffer;

	VkDescriptorSetLayout _cullSetLayout;

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

	std::vector<IndirectBatch> compact_draws(RenderObject* objects, int count);

	void execute_shadow_culling(VkCommandBuffer cmd, RenderObject* first, int count, int cascadesIndex);

	void execute_culling(VkCommandBuffer cmd, RenderObject* first, int count);

	void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);

	void draw_gbuffer(VkCommandBuffer cmd, RenderObject* first, int count);

	void prepare_depthpass();

	void update_csm_descriptors(RenderObject* first, int count);

	void update_csm(VkCommandBuffer cmd, RenderObject* first, int count, int cascadesIndex);

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

	bool load_compute_shader(const char* shaderPath);

	void upload_mesh(Mesh& mesh);

	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);

	static void cursor_enter_callback(GLFWwindow* window, int entered);

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
}; 

