#include "vk_types.h"
#include "vk_initializers.h"
#include "vk_descriptors.h"
#include "VkBootstrap.h"
#include "vk_engine.h"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <fstream>

#include "Texture.h"

#define VMA_STATIC_VULKAN_FUNCTIONS 0 
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>


#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = x;                                           \
		if (err)                                                    \
		{                                                           \
			std::cout <<"Detected Vulkan error: " << err << std::endl; \
			abort();                                                \
		}                                                           \
	} while (0)

#ifdef NDEBUG
constexpr bool bUseValidationLayers = false;
#else
constexpr bool bUseValidationLayers = true;
#endif

const std::vector<std::string> TEXTURE_PATHS =
{
	"assets/marble.jpg",
	"assets/WoodFineDark004_COL_3K.jpg",
	 "assets/green-marble.jpg"
};

const std::vector<std::string> TEXTURE_NAMES =
{
	"marble",
	"wood",
	 "green-marble"
};


void VulkanEngine::init()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	_window = glfwCreateWindow(_windowExtent.width,
		_windowExtent.height, "Vulkan", nullptr, nullptr);

	glfwSetWindowUserPointer(_window, this);
	glfwSetFramebufferSizeCallback(_window, framebufferResizeCallback);

	glfwSetWindowUserPointer(_window, this);

	glfwSetKeyCallback(_window, key_callback);
	glfwSetMouseButtonCallback(_window, mouse_button_callback);
	glfwSetCursorPosCallback(_window, cursor_position_callback);
	glfwSetCursorEnterCallback(_window, cursor_enter_callback);

	init_vulkan();

	init_swapchain();

	init_shadow_renderpass();

	init_gbuffer_renderpass();

	init_default_renderpass();

	init_framebuffers();

	init_commands();

	init_sync_structures();

	init_descriptors();

	init_pipelines();

	load_images();

	load_meshes();

	init_scene();

	_isInitialized = true;
}
void VulkanEngine::cleanup()
{
	if (_isInitialized) {

		vkDeviceWaitIdle(_device);

		_mainDeletionQueue.flush();


		vkDestroySurfaceKHR(_instance, _surface, nullptr);

		_descriptorAllocator->cleanup();
		_descriptorLayoutCache->cleanup();

		vkDestroyDevice(_device, nullptr);
		vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
		vkDestroyInstance(_instance, nullptr);

		glfwDestroyWindow(_window);

		glfwTerminate();
	}
}

void VulkanEngine::draw()
{
	VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));
	VK_CHECK(vkResetCommandBuffer(get_current_frame()._shadowCommandBuffer, 0));
	VK_CHECK(vkResetCommandBuffer(get_current_frame()._gBufferCommandBuffer, 0));
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._presentSemaphore, nullptr, &swapchainImageIndex));

	VkCommandBuffer cmd = get_current_frame()._shadowCommandBuffer;

	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	
	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	VkClearValue clearValue;
	clearValue.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
	
	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.f;

	VkClearValue clearValues[] = { clearValue, depthClear };
	
	VkRenderPassBeginInfo sdrpInfo = vkinit::renderpass_begin_info(_shadowPass, _shadowExtent, _shadowFramebuffer);

	sdrpInfo.clearValueCount = 2;

	sdrpInfo.pClearValues = &clearValues[0];

	vkCmdBeginRenderPass(cmd, &sdrpInfo, VK_SUBPASS_CONTENTS_INLINE);

	draw_shadow(cmd, _renderables.data(), _renderables.size());

	vkCmdEndRenderPass(cmd);
	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo shadowSubmit = vkinit::submit_info(&cmd);

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &shadowSubmit, nullptr));
	
	cmd = get_current_frame()._gBufferCommandBuffer;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	VkClearValue gBufferClearValues[] = { clearValue, clearValue, clearValue, clearValue, clearValue, depthClear };

	VkRenderPassBeginInfo grpInfo = vkinit::renderpass_begin_info(_gBufferPass, _gBufferExtent, _gBufferFramebuffer);

	grpInfo.clearValueCount = 6;

	grpInfo.pClearValues = &gBufferClearValues[0];

	vkCmdBeginRenderPass(cmd, &grpInfo, VK_SUBPASS_CONTENTS_INLINE);

	draw_gbuffer(cmd, _renderables.data(), _renderables.size());

	vkCmdEndRenderPass(cmd);
	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo gbufferSubmit = vkinit::submit_info(&cmd);

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &gbufferSubmit, nullptr));

	cmd = get_current_frame()._mainCommandBuffer;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_renderPass, _windowExtent, _framebuffers[swapchainImageIndex]);

	rpInfo.clearValueCount = 2;

	rpInfo.pClearValues = &clearValues[0];

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	draw_objects(cmd, _renderables.data(), _renderables.size());

	vkCmdEndRenderPass(cmd);
	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &get_current_frame()._presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &get_current_frame()._renderSemaphore;

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

	VkPresentInfoKHR presentInfo = vkinit::present_info();

	presentInfo.pSwapchains = &_swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

	_frameNumber++;
}

void VulkanEngine::run()
{
	while (!glfwWindowShouldClose(_window)) {
		glfwPollEvents();
		draw();
	}

	vkDeviceWaitIdle(_device);
}

FrameData& VulkanEngine::get_current_frame()
{
	return _frames[_frameNumber % FRAME_OVERLAP];
}


FrameData& VulkanEngine::get_last_frame()
{
	return _frames[(_frameNumber - 1) % 2];
}

void VulkanEngine::init_vulkan()
{
	vkb::InstanceBuilder builder;

	auto inst_ret = builder.set_app_name("Example Vulkan Application")
		.request_validation_layers(bUseValidationLayers)
		.use_default_debug_messenger()
		.require_api_version(1, 1, 0)
		.build();

	vkb::Instance vkb_inst = inst_ret.value();

	_instance = vkb_inst.instance;
	_debug_messenger = vkb_inst.debug_messenger;

	glfwCreateWindowSurface(_instance, _window, NULL, &_surface);

	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 1)
		.set_surface(_surface)
		.select()
		.value();

	vkb::DeviceBuilder deviceBuilder{ physicalDevice };
	VkPhysicalDeviceShaderDrawParametersFeatures shader_draw_parameters_features = {};
	shader_draw_parameters_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
	shader_draw_parameters_features.pNext = nullptr;
	shader_draw_parameters_features.shaderDrawParameters = true;
	vkb::Device vkbDevice = deviceBuilder.add_pNext(&shader_draw_parameters_features).build().value();

	_device = vkbDevice.device;
	_chosenGPU = physicalDevice.physical_device;

	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();

	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	VmaVulkanFunctions vulkanFunctions = {};
	vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
	vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = _chosenGPU;
	allocatorInfo.device = _device;
	allocatorInfo.instance = _instance;
	allocatorInfo.pVulkanFunctions = &vulkanFunctions;
	vmaCreateAllocator(&allocatorInfo, &_allocator);

	_mainDeletionQueue.push_function([&]() {
		vmaDestroyAllocator(_allocator);
		});

	vkGetPhysicalDeviceProperties(_chosenGPU, &_gpuProperties);

}

void VulkanEngine::init_swapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(_windowExtent.width, _windowExtent.height)
		.build()
		.value();

	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews = vkbSwapchain.get_image_views().value();

	_swachainImageFormat = vkbSwapchain.image_format;

	_mainDeletionQueue.push_function([=]() {
		vkDestroySwapchainKHR(_device, _swapchain, nullptr);
		});

	VkExtent3D depthImageExtent = {
		_windowExtent.width,
		_windowExtent.height,
		1
	};

	VkExtent3D shadowExtent = {
		_shadowExtent.width,
		_shadowExtent.height,
		1
	};

	VkExtent3D gBufferExtent = {
		_gBufferExtent.width,
		_gBufferExtent.height,
		1
	};

	_depthFormat = VK_FORMAT_D32_SFLOAT;
	_shadowMapFormat = VK_FORMAT_R8G8B8A8_SRGB;

	VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image, &_depthImage._allocation, nullptr);

	VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

	VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

	VkImageCreateInfo dimg_shadow_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, shadowExtent);

	vmaCreateImage(_allocator, &dimg_shadow_info, &dimg_allocinfo, &_shadowDepthImage._image, &_shadowDepthImage._allocation, nullptr);

	VkImageViewCreateInfo dview_shadow_info = vkinit::imageview_create_info(_depthFormat, _shadowDepthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

	VK_CHECK(vkCreateImageView(_device, &dview_shadow_info, nullptr, &_shadowDepthImageView));

	VmaAllocationCreateInfo cimg_allocinfo = {};
	cimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	cimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkImageCreateInfo cimg_shadow_info = vkinit::image_create_info(_shadowMapFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, shadowExtent);

	vmaCreateImage(_allocator, &cimg_shadow_info, &cimg_allocinfo, &_shadowColorImage._image, &_shadowColorImage._allocation, nullptr);

	VkImageViewCreateInfo cview_shadow_info = vkinit::imageview_create_info(_shadowMapFormat, _shadowColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);

	VK_CHECK(vkCreateImageView(_device, &cview_shadow_info, nullptr, &_shadowColorImageView));

	VkImageCreateInfo cimg_gbuffer_info = vkinit::image_create_info(_shadowMapFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, gBufferExtent);

	vmaCreateImage(_allocator, &cimg_gbuffer_info, &cimg_allocinfo, &_gBufferColorImage._image, &_gBufferColorImage._allocation, nullptr);
	vmaCreateImage(_allocator, &cimg_gbuffer_info, &cimg_allocinfo, &_gBufferPosImage._image, &_gBufferPosImage._allocation, nullptr);
	vmaCreateImage(_allocator, &cimg_gbuffer_info, &cimg_allocinfo, &_gBufferNormalImage._image, &_gBufferNormalImage._allocation, nullptr);
	vmaCreateImage(_allocator, &cimg_gbuffer_info, &cimg_allocinfo, &_gBufferVisibilityImage._image, &_gBufferVisibilityImage._allocation, nullptr);
	vmaCreateImage(_allocator, &cimg_gbuffer_info, &cimg_allocinfo, &_gBufferDepthRGBAImage._image, &_gBufferDepthRGBAImage._allocation, nullptr);

	VkImageViewCreateInfo cview_gbuffer_info1 = vkinit::imageview_create_info(_shadowMapFormat, _gBufferColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
	VkImageViewCreateInfo cview_gbuffer_info2 = vkinit::imageview_create_info(_shadowMapFormat, _gBufferPosImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
	VkImageViewCreateInfo cview_gbuffer_info3 = vkinit::imageview_create_info(_shadowMapFormat, _gBufferNormalImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
	VkImageViewCreateInfo cview_gbuffer_info4 = vkinit::imageview_create_info(_shadowMapFormat, _gBufferVisibilityImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
	VkImageViewCreateInfo cview_gbuffer_info5 = vkinit::imageview_create_info(_shadowMapFormat, _gBufferDepthRGBAImage._image, VK_IMAGE_ASPECT_COLOR_BIT);

	VK_CHECK(vkCreateImageView(_device, &cview_gbuffer_info1, nullptr, &_gBufferColorImageView));
	VK_CHECK(vkCreateImageView(_device, &cview_gbuffer_info2, nullptr, &_gBufferPosImageView));
	VK_CHECK(vkCreateImageView(_device, &cview_gbuffer_info3, nullptr, &_gBufferNormalImageView));
	VK_CHECK(vkCreateImageView(_device, &cview_gbuffer_info4, nullptr, &_gBufferVisibilityImageView));
	VK_CHECK(vkCreateImageView(_device, &cview_gbuffer_info5, nullptr, &_gBufferDepthRGBAImageView));

	VkImageCreateInfo dimg_gbuffer_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, gBufferExtent);
	vmaCreateImage(_allocator, &dimg_gbuffer_info, &dimg_allocinfo, &_gBufferDepthImage._image, &_gBufferDepthImage._allocation, nullptr);
	VkImageViewCreateInfo dview_gbuffer_info = vkinit::imageview_create_info(_depthFormat, _gBufferDepthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);
	VK_CHECK(vkCreateImageView(_device, &dview_gbuffer_info, nullptr, &_gBufferDepthImageView));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, _depthImageView, nullptr);
		vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);

		vkDestroyImageView(_device, _shadowColorImageView, nullptr);
		vmaDestroyImage(_allocator, _shadowColorImage._image, _shadowColorImage._allocation);
		vkDestroyImageView(_device, _shadowDepthImageView, nullptr);
		vmaDestroyImage(_allocator, _shadowDepthImage._image, _shadowDepthImage._allocation);

		vkDestroyImageView(_device, _gBufferColorImageView, nullptr);
		vmaDestroyImage(_allocator, _gBufferColorImage._image, _gBufferColorImage._allocation);
		vkDestroyImageView(_device, _gBufferPosImageView, nullptr);
		vmaDestroyImage(_allocator, _gBufferPosImage._image, _gBufferPosImage._allocation);
		vkDestroyImageView(_device, _gBufferNormalImageView, nullptr);
		vmaDestroyImage(_allocator, _gBufferNormalImage._image, _gBufferNormalImage._allocation);
		vkDestroyImageView(_device, _gBufferVisibilityImageView, nullptr);
		vmaDestroyImage(_allocator, _gBufferVisibilityImage._image, _gBufferVisibilityImage._allocation);
		vkDestroyImageView(_device, _gBufferDepthRGBAImageView, nullptr);
		vmaDestroyImage(_allocator, _gBufferDepthRGBAImage._image, _gBufferDepthRGBAImage._allocation);
		vkDestroyImageView(_device, _gBufferDepthImageView, nullptr);
		vmaDestroyImage(_allocator, _gBufferDepthImage._image, _gBufferDepthImage._allocation);
		});
}

void VulkanEngine::init_shadow_renderpass()
{

	VkAttachmentDescription color_attachment = {};
	color_attachment.format = VK_FORMAT_R8G8B8A8_SRGB;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	depth_attachment.flags = 0;
	depth_attachment.format = _depthFormat;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 1;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkSubpassDependency depth_dependency = {};
	depth_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	depth_dependency.dstSubpass = 0;
	depth_dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depth_dependency.srcAccessMask = 0;
	depth_dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depth_dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	VkSubpassDependency dependencies[2] = { dependency, depth_dependency };

	VkAttachmentDescription attachments[2] = { color_attachment, depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 2;
	render_pass_info.pAttachments = &attachments[0];
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;
	render_pass_info.dependencyCount = 2;
	render_pass_info.pDependencies = &dependencies[0];

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_shadowPass));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _shadowPass, nullptr);
		});

}

void VulkanEngine::init_gbuffer_renderpass()
{

	VkAttachmentDescription color_attachment = {};
	color_attachment.format = VK_FORMAT_R8G8B8A8_SRGB;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference color_attachment_ref[5]{};
	color_attachment_ref[0].attachment = 0;
	color_attachment_ref[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	color_attachment_ref[1].attachment = 1;
	color_attachment_ref[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	color_attachment_ref[2].attachment = 2;
	color_attachment_ref[2].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	color_attachment_ref[3].attachment = 3;
	color_attachment_ref[3].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	color_attachment_ref[4].attachment = 4;
	color_attachment_ref[4].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	depth_attachment.flags = 0;
	depth_attachment.format = _depthFormat;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 5;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 5;
	subpass.pColorAttachments = &color_attachment_ref[0];
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkSubpassDependency depth_dependency = {};
	depth_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	depth_dependency.dstSubpass = 0;
	depth_dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depth_dependency.srcAccessMask = 0;
	depth_dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depth_dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	VkSubpassDependency dependencies[2] = { dependency, depth_dependency };

	VkAttachmentDescription attachments[6] = { color_attachment, color_attachment ,color_attachment, color_attachment,color_attachment, depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 6;
	render_pass_info.pAttachments = &attachments[0];
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;
	render_pass_info.dependencyCount = 2;
	render_pass_info.pDependencies = &dependencies[0];

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_gBufferPass));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _gBufferPass, nullptr);
		});

}

void VulkanEngine::init_default_renderpass()
{

	VkAttachmentDescription color_attachment = {};
	color_attachment.format = _swachainImageFormat;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	depth_attachment.flags = 0;
	depth_attachment.format = _depthFormat;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 1;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription color_attachment_res = {};
	color_attachment_res.format = _swachainImageFormat;
	color_attachment_res.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment_res.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment_res.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment_res.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment_res.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment_res.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment_res.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_res_ref = {};
	color_attachment_res_ref.attachment = 2;
	color_attachment_res_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;
	subpass.pDepthStencilAttachment = &depth_attachment_ref;
	//subpass.pResolveAttachments = &color_attachment_res_ref;

	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkSubpassDependency depth_dependency = {};
	depth_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	depth_dependency.dstSubpass = 0;
	depth_dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depth_dependency.srcAccessMask = 0;
	depth_dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depth_dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	VkSubpassDependency dependencies[2] = { dependency, depth_dependency };

	VkAttachmentDescription attachments[3] = { color_attachment, depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 2;
	render_pass_info.pAttachments = &attachments[0];
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;
	render_pass_info.dependencyCount = 2;
	render_pass_info.pDependencies = &dependencies[0];

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _renderPass, nullptr);
		});

}

void VulkanEngine::init_framebuffers()
{
	VkImageView shadowAttachments[2];
	shadowAttachments[0] = _shadowColorImageView;
	shadowAttachments[1] = _shadowDepthImageView;

	VkFramebufferCreateInfo sh_info = vkinit::framebuffer_create_info(_shadowPass, _shadowExtent);
	sh_info.pAttachments = shadowAttachments;
	sh_info.attachmentCount = 2;
	VK_CHECK(vkCreateFramebuffer(_device, &sh_info, nullptr, &_shadowFramebuffer));

	VkImageView gBufferAttachments[] = { _gBufferPosImageView, _gBufferColorImageView, _gBufferNormalImageView, _gBufferVisibilityImageView, _gBufferDepthRGBAImageView,  _gBufferDepthImageView };

	VkFramebufferCreateInfo g_info = vkinit::framebuffer_create_info(_gBufferPass, _gBufferExtent);
	g_info.pAttachments = gBufferAttachments;
	g_info.attachmentCount = 6;
	VK_CHECK(vkCreateFramebuffer(_device, &g_info, nullptr, &_gBufferFramebuffer));

	VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_renderPass, _windowExtent);

	const uint32_t swapchain_imagecount = _swapchainImages.size();
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);


	for (int i = 0; i < swapchain_imagecount; i++) {
		VkImageView attachments[2];
		attachments[0] = _swapchainImageViews[i];
		attachments[1] = _depthImageView;

		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 2;

		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
			vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
			});
	}
	_mainDeletionQueue.push_function([=]() {
		vkDestroyFramebuffer(_device, _shadowFramebuffer, nullptr);
		vkDestroyFramebuffer(_device, _gBufferFramebuffer, nullptr);
		});
}

void VulkanEngine::init_commands()
{
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);


	for (int i = 0; i < FRAME_OVERLAP; i++) {


		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));
		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._shadowCommandPool));
		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._gBufferCommandPool));

		VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

		VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));

		cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._shadowCommandPool, 1);
		VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._shadowCommandBuffer));

		cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._gBufferCommandPool, 1);
		VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._gBufferCommandBuffer));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
			vkDestroyCommandPool(_device, _frames[i]._shadowCommandPool, nullptr);
			vkDestroyCommandPool(_device, _frames[i]._gBufferCommandPool, nullptr);
			});
	}


	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
	VK_CHECK(vkCreateCommandPool(_device, &uploadCommandPoolInfo, nullptr, &_uploadContext._commandPool));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(_device, _uploadContext._commandPool, nullptr);
		});

	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_uploadContext._commandPool, 1);

	VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_uploadContext._commandBuffer));
}

void VulkanEngine::init_sync_structures()
{
	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);

	VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

	for (int i = 0; i < FRAME_OVERLAP; i++) {

		VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
			});


		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._presentSemaphore));
		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));

		_mainDeletionQueue.push_function([=]() {
			vkDestroySemaphore(_device, _frames[i]._presentSemaphore, nullptr);
			vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
			});
	}


	VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info();

	VK_CHECK(vkCreateFence(_device, &uploadFenceCreateInfo, nullptr, &_uploadContext._uploadFence));
	_mainDeletionQueue.push_function([=]() {
		vkDestroyFence(_device, _uploadContext._uploadFence, nullptr);
		});
}

void VulkanEngine::init_pipelines()
{
	VkShaderModule colorMeshShader;
	if (!load_shader_module("shaders/color_shader.frag.spv", &colorMeshShader))
	{
		std::cout << "Error when building the colored mesh shader" << std::endl;
	}

	VkShaderModule texturedMeshShader;
	if (!load_shader_module("shaders/textured_shader.frag.spv", &texturedMeshShader))
	{
		std::cout << "Error when building the textured mesh shader" << std::endl;
	}

	VkShaderModule texturedSpecularMeshShader;
	if (!load_shader_module("shaders/ssr_shader.frag.spv", &texturedSpecularMeshShader))
	{
		std::cout << "Error when building the textured specular mesh shader" << std::endl;
	}

	VkShaderModule meshVertShader;
	if (!load_shader_module("shaders/shader.vert.spv", &meshVertShader))
	{
		std::cout << "Error when building the mesh vertex shader module" << std::endl;
	}

	VkShaderModule gBufferShader;
	if (!load_shader_module("shaders/gbuffer_shader.frag.spv", &gBufferShader))
	{
		std::cout << "Error when building the gbuffer shader module" << std::endl;
	}

	VkShaderModule gBufferTexturedShader;
	if (!load_shader_module("shaders/gbuffer_textured_shader.frag.spv", &gBufferTexturedShader))
	{
		std::cout << "Error when building the gbuffer shader module" << std::endl;
	}

	VkShaderModule shadowVertShader;
	if (!load_shader_module("shaders/shadow_shader.vert.spv", &shadowVertShader))
	{
		std::cout << "Error when building the shadow vertex shader module" << std::endl;
	}

	VkShaderModule depthShader;
	if (!load_shader_module("shaders/shadow_shader.frag.spv", &depthShader))
	{
		std::cout << "Error when building the depth shader module" << std::endl;
	}

	PipelineBuilder pipelineBuilder;

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, colorMeshShader));

	VkPipelineLayoutCreateInfo mesh_pipeline_layout_info = vkinit::pipeline_layout_create_info();

	VkDescriptorSetLayout setLayouts[] = { _globalSetLayout, _lightSetLayout, _objectSetLayout, _singleTextureSetLayout };

	mesh_pipeline_layout_info.setLayoutCount = 4;
	mesh_pipeline_layout_info.pSetLayouts = setLayouts;

	VkPipelineLayout meshPipeLayout;
	VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_info, nullptr, &meshPipeLayout));

	VkPipelineLayoutCreateInfo textured_pipeline_layout_info = mesh_pipeline_layout_info;

	VkDescriptorSetLayout texturedSetLayouts[] = { _globalSetLayout, _lightSetLayout, _objectSetLayout, _singleTextureSetLayout, _singleTextureSetLayout };

	textured_pipeline_layout_info.setLayoutCount = 5;
	textured_pipeline_layout_info.pSetLayouts = texturedSetLayouts;

	VkPipelineLayout texturedPipeLayout;
	VK_CHECK(vkCreatePipelineLayout(_device, &textured_pipeline_layout_info, nullptr, &texturedPipeLayout));

	VkPipelineLayoutCreateInfo textured_specular_pipeline_layout_info = mesh_pipeline_layout_info;

	VkDescriptorSetLayout texturedSpecularSetLayouts[] = 
	{ 
		_globalSetLayout,
		_lightSetLayout,
		_objectSetLayout,
		_singleTextureSetLayout,
		_singleTextureSetLayout,
		_gBufferSetLayout 
	};

	textured_specular_pipeline_layout_info.setLayoutCount = 6;
	textured_specular_pipeline_layout_info.pSetLayouts = texturedSpecularSetLayouts;

	VkPipelineLayout texturedSpecularPipeLayout;
	VK_CHECK(vkCreatePipelineLayout(_device, &textured_specular_pipeline_layout_info, nullptr, &texturedSpecularPipeLayout));

	VkPipelineLayoutCreateInfo shadow_pipeline_layout_info = mesh_pipeline_layout_info;
	VkDescriptorSetLayout shadowSetLayouts[] = { _lightSetLayout, _objectSetLayout };

	shadow_pipeline_layout_info.setLayoutCount = 2;
	shadow_pipeline_layout_info.pSetLayouts = shadowSetLayouts;

	VkPipelineLayout shadowPipeLayout;
	VK_CHECK(vkCreatePipelineLayout(_device, &shadow_pipeline_layout_info, nullptr, &shadowPipeLayout));

	pipelineBuilder._pipelineLayout = meshPipeLayout;

	pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();


	pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	pipelineBuilder._viewport.x = 0.0f;
	pipelineBuilder._viewport.y = 0.0f;
	pipelineBuilder._viewport.width = (float)_windowExtent.width;
	pipelineBuilder._viewport.height = (float)_windowExtent.height;
	pipelineBuilder._viewport.minDepth = 0.0f;
	pipelineBuilder._viewport.maxDepth = 1.0f;

	pipelineBuilder._scissor.offset = { 0, 0 };
	pipelineBuilder._scissor.extent = _windowExtent;

	pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

	pipelineBuilder._colorBlendAttachment.push_back(vkinit::color_blend_attachment_state());

	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);


	VertexInputDescription vertexDescription = Vertex::get_vertex_description();

	pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

	pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	VkPipeline meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

	create_material(meshPipeline, meshPipeLayout, "defaultmesh");

	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, texturedMeshShader));

	pipelineBuilder._pipelineLayout = texturedPipeLayout;
	VkPipeline texPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);
	create_material(texPipeline, texturedPipeLayout, "texturedmesh");
	create_material(texPipeline, texturedPipeLayout, TEXTURE_NAMES[0]);
	create_material(texPipeline, texturedPipeLayout, TEXTURE_NAMES[1]);

	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, texturedSpecularMeshShader));
	pipelineBuilder._pipelineLayout = texturedSpecularPipeLayout;
	VkPipeline ssrPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);
	create_material(ssrPipeline, texturedSpecularPipeLayout, TEXTURE_NAMES[2]);

	pipelineBuilder._viewport.width = (float)_shadowExtent.width;
	pipelineBuilder._viewport.height = (float)_shadowExtent.height;
	pipelineBuilder._scissor.extent = _shadowExtent;
	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, shadowVertShader));

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, depthShader));

	pipelineBuilder._pipelineLayout = shadowPipeLayout;
	VkPipeline shadowPipeline = pipelineBuilder.build_pipeline(_device, _shadowPass);
	create_material(shadowPipeline, shadowPipeLayout, "depth");

	pipelineBuilder._viewport.width = (float)_gBufferExtent.width;
	pipelineBuilder._viewport.height = (float)_gBufferExtent.height;
	pipelineBuilder._scissor.extent = _gBufferExtent;
	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, gBufferShader));

	pipelineBuilder._pipelineLayout = texturedPipeLayout;
	pipelineBuilder._colorBlendAttachment.push_back(vkinit::color_blend_attachment_state());
	pipelineBuilder._colorBlendAttachment.push_back(vkinit::color_blend_attachment_state());
	pipelineBuilder._colorBlendAttachment.push_back(vkinit::color_blend_attachment_state());
	pipelineBuilder._colorBlendAttachment.push_back(vkinit::color_blend_attachment_state());
	VkPipeline gBufferPipeline = pipelineBuilder.build_pipeline(_device, _gBufferPass, 5);
	create_material(gBufferPipeline, texturedPipeLayout, "gbuffer");

	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, gBufferTexturedShader));

	pipelineBuilder._pipelineLayout = meshPipeLayout;
	VkPipeline gBufferTexturedPipeline = pipelineBuilder.build_pipeline(_device, _gBufferPass, 5);
	create_material(gBufferTexturedPipeline, meshPipeLayout, "gbufferTextured");

	vkDestroyShaderModule(_device, meshVertShader, nullptr);
	vkDestroyShaderModule(_device, colorMeshShader, nullptr);
	vkDestroyShaderModule(_device, texturedMeshShader, nullptr);
	vkDestroyShaderModule(_device, shadowVertShader, nullptr);
	vkDestroyShaderModule(_device, depthShader, nullptr);
	vkDestroyShaderModule(_device, gBufferShader, nullptr);
	vkDestroyShaderModule(_device, gBufferTexturedShader, nullptr);
	vkDestroyShaderModule(_device, texturedSpecularMeshShader, nullptr);

	_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(_device, meshPipeline, nullptr);
		vkDestroyPipeline(_device, texPipeline, nullptr);
		vkDestroyPipeline(_device, shadowPipeline, nullptr);
		vkDestroyPipeline(_device, gBufferPipeline, nullptr);
		vkDestroyPipeline(_device, gBufferTexturedPipeline, nullptr);
		vkDestroyPipeline(_device, ssrPipeline, nullptr);

		vkDestroyPipelineLayout(_device, meshPipeLayout, nullptr);
		vkDestroyPipelineLayout(_device, texturedPipeLayout, nullptr);
		vkDestroyPipelineLayout(_device, shadowPipeLayout, nullptr);
		vkDestroyPipelineLayout(_device, texturedSpecularPipeLayout, nullptr);
		});

}

bool VulkanEngine::load_shader_module(const char* filePath, VkShaderModule* outShaderModule)
{
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		return false;
	}

	size_t fileSize = (size_t)file.tellg();

	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

	file.seekg(0);

	file.read((char*)buffer.data(), fileSize);

	file.close();

	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.pNext = nullptr;

	createInfo.codeSize = buffer.size() * sizeof(uint32_t);
	createInfo.pCode = buffer.data();

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		return false;
	}
	*outShaderModule = shaderModule;
	return true;
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass)
{

	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.pNext = nullptr;

	viewportState.viewportCount = 1;
	viewportState.pViewports = &_viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &_scissor;

	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.pNext = nullptr;

	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &_colorBlendAttachment[0];

	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = nullptr;

	pipelineInfo.stageCount = _shaderStages.size();
	pipelineInfo.pStages = _shaderStages.data();
	pipelineInfo.pVertexInputState = &_vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &_inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &_rasterizer;
	pipelineInfo.pMultisampleState = &_multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDepthStencilState = &_depthStencil;
	pipelineInfo.layout = _pipelineLayout;
	pipelineInfo.renderPass = pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(
		device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		std::cout << "failed to create pipline\n";
		return VK_NULL_HANDLE; 
	}
	else
	{
		return newPipeline;
	}
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass, int size)
{

	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.pNext = nullptr;

	viewportState.viewportCount = 1;
	viewportState.pViewports = &_viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &_scissor;

	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.pNext = nullptr;

	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = size;
	colorBlending.pAttachments = &_colorBlendAttachment[0];

	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = nullptr;

	pipelineInfo.stageCount = _shaderStages.size();
	pipelineInfo.pStages = _shaderStages.data();
	pipelineInfo.pVertexInputState = &_vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &_inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &_rasterizer;
	pipelineInfo.pMultisampleState = &_multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDepthStencilState = &_depthStencil;
	pipelineInfo.layout = _pipelineLayout;
	pipelineInfo.renderPass = pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(
		device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		std::cout << "failed to create pipline\n";
		return VK_NULL_HANDLE;
	}
	else
	{
		return newPipeline;
	}
}

void VulkanEngine::load_meshes()
{
	Mesh triMesh{};
	triMesh._vertices.resize(3);

	triMesh._vertices[0].position = { 1.f,1.f, 0.0f };
	triMesh._vertices[1].position = { -1.f,1.f, 0.0f };
	triMesh._vertices[2].position = { 0.f,-1.f, 0.0f };

	triMesh._vertices[0].color = { 0.f,1.f, 0.0f };
	triMesh._vertices[1].color = { 0.f,1.f, 0.0f }; 
	triMesh._vertices[2].color = { 0.f,1.f, 0.0f }; 

	Mesh monkeyMesh{};
	monkeyMesh.load_from_obj("./assets/monkey_smooth.obj");

	Mesh lostEmpire{};
	lostEmpire.load_from_obj("./assets/lost_empire.obj");

	Meshes chairsSet{};
	chairsSet.load_from_obj("./assets/Table&Chair Set .obj");

	upload_mesh(triMesh);
	upload_mesh(monkeyMesh);
	upload_mesh(lostEmpire);

	for (auto m : chairsSet._meshes) {
		upload_mesh(m);
		_meshes[m.name] = m;
	}

	_meshes["monkey"] = monkeyMesh;
	_meshes["triangle"] = triMesh;
	_meshes["empire"] = lostEmpire;
}

void VulkanEngine::load_images()
{
	Texture lostEmpire;

	vkutil::load_image_from_file(*this, "assets/lost_empire-RGBA.png", lostEmpire.image);

	VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_SRGB, lostEmpire.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(_device, &imageinfo, nullptr, &lostEmpire.imageView);

	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, lostEmpire.imageView, nullptr);
		});

	_loadedTextures["empire_diffuse"] = lostEmpire;

	std::vector<Texture> texList;
	

	for (int i = 0; i < TEXTURE_PATHS.size(); i++) {

		Texture tex;

		vkutil::load_image_from_file(*this, TEXTURE_PATHS[i], tex.image);

		VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_SRGB, tex.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
		vkCreateImageView(_device, &imageinfo, nullptr, &tex.imageView);

		_mainDeletionQueue.push_function([=]() {
			vkDestroyImageView(_device, tex.imageView, nullptr);
			});

		_loadedTextures[TEXTURE_NAMES[i]] = tex;
	}
}

void VulkanEngine::upload_mesh(Mesh& mesh)
{
	const size_t bufferSize = mesh._vertices.size() * sizeof(Vertex);
	VkBufferCreateInfo stagingBufferInfo = {};
	stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingBufferInfo.pNext = nullptr;
	stagingBufferInfo.size = bufferSize;

	stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	AllocatedBuffer stagingBuffer;

	VK_CHECK(vmaCreateBuffer(_allocator, &stagingBufferInfo, &vmaallocInfo,
		&stagingBuffer._buffer,
		&stagingBuffer._allocation,
		nullptr));

	void* data;
	vmaMapMemory(_allocator, stagingBuffer._allocation, &data);

	memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));

	vmaUnmapMemory(_allocator, stagingBuffer._allocation);

	VkBufferCreateInfo vertexBufferInfo = {};
	vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vertexBufferInfo.pNext = nullptr;
	vertexBufferInfo.size = bufferSize;
	vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	vmaallocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VK_CHECK(vmaCreateBuffer(_allocator, &vertexBufferInfo, &vmaallocInfo,
		&mesh._vertexBuffer._buffer,
		&mesh._vertexBuffer._allocation,
		nullptr));
	_mainDeletionQueue.push_function([=]() {

		vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
		});

	immediate_submit([=](VkCommandBuffer cmd) {
		VkBufferCopy copy;
		copy.dstOffset = 0;
		copy.srcOffset = 0;
		copy.size = bufferSize;
		vkCmdCopyBuffer(cmd, stagingBuffer._buffer, mesh._vertexBuffer._buffer, 1, &copy);
		});

	vmaDestroyBuffer(_allocator, stagingBuffer._buffer, stagingBuffer._allocation);
}

Material* VulkanEngine::create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name)
{
	Material mat;
	mat.pipeline = pipeline;
	mat.pipelineLayout = layout;
	_materials[name] = mat;
	return &_materials[name];
}

Material* VulkanEngine::get_material(const std::string& name)
{
	auto it = _materials.find(name);
	if (it == _materials.end()) {
		return nullptr;
	}
	else {
		return &(*it).second;
	}
}


Mesh* VulkanEngine::get_mesh(const std::string& name)
{
	auto it = _meshes.find(name);
	if (it == _meshes.end()) {
		return nullptr;
	}
	else {
		return &(*it).second;
	}
}

void VulkanEngine::draw_shadow(VkCommandBuffer cmd, RenderObject* first, int count)
{
	_sceneParameters.lightColor = { 1.0,1.0,1.0 };
	_sceneParameters.zNear = 140.0f;
	_sceneParameters.zFar = 240.0f;
	char* sceneData;
	vmaMapMemory(_allocator, _sceneParameterBuffer._allocation, (void**)&sceneData);

	int frameIndex = _frameNumber % FRAME_OVERLAP;

	sceneData += pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;

	memcpy(sceneData, &_sceneParameters, sizeof(GPUSceneData));

	vmaUnmapMemory(_allocator, _sceneParameterBuffer._allocation);

	glm::mat4 clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, -1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.5f, 0.0f,
		0.0f, 0.0f, 0.5f, 1.0f);

	glm::vec3 lightPos = { -120.0f,140.0f,80.0f };

	glm::mat4 view = glm::lookAt(lightPos, { 20.0f, -20.0f, 0.0f }, glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 projection = clip * glm::perspective(glm::radians(45.0f), _shadowExtent.width / (float)_shadowExtent.height, _sceneParameters.zNear, _sceneParameters.zFar);

	GPUCameraData lightData;
	lightData.pos = lightPos;
	lightData.viewproj = projection * view;

	void* data;
	vmaMapMemory(_allocator, get_current_frame().lightBuffer._allocation, &data);

	memcpy(data, &lightData, sizeof(GPUCameraData));

	vmaUnmapMemory(_allocator, get_current_frame().lightBuffer._allocation);

	view = glm::lookAt(_camPos, _foc, glm::vec3(0.0f, 1.0f, 0.0f));
	projection = clip * glm::perspective(glm::radians(45.0f), _windowExtent.width / (float)_windowExtent.height, 10.0f, 100.0f);

	GPUCameraData camData;
	camData.pos = _camPos;
	camData.viewproj = projection * view;

	vmaMapMemory(_allocator, get_current_frame().cameraBuffer._allocation, &data);

	memcpy(data, &camData, sizeof(GPUCameraData));

	vmaUnmapMemory(_allocator, get_current_frame().cameraBuffer._allocation);


	void* objectData;
	vmaMapMemory(_allocator, get_current_frame().objectBuffer._allocation, &objectData);

	GPUObjectData* objectSSBO = (GPUObjectData*)objectData;


	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];
		objectSSBO[i].modelMatrix = object.transformMatrix;
	}

	vmaUnmapMemory(_allocator, get_current_frame().objectBuffer._allocation);

	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];	

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, get_material("depth")->pipeline);

		uint32_t uniform_offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, get_material("depth")->pipelineLayout, 0, 1, &get_current_frame().lightDescriptor, 1, &uniform_offset);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, get_material("depth")->pipelineLayout, 1, 1, &get_current_frame().objectDescriptor, 0, nullptr);

		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer, &offset);


		vkCmdDraw(cmd, object.mesh->_vertices.size(), 1, 0, i);
	}
}

void VulkanEngine::draw_gbuffer(VkCommandBuffer cmd, RenderObject* first, int count)
{
	int frameIndex = _frameNumber % FRAME_OVERLAP;



	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];
		VkPipelineLayout layout;
		VkPipeline pipeline;
		if (object.material->textureSet != VK_NULL_HANDLE) {
			layout = get_material("gbuffer")->pipelineLayout;
			pipeline = get_material("gbuffer")->pipeline;
		}
		else {
			layout = get_material("gbufferTextured")->pipelineLayout;
			pipeline = get_material("gbufferTextured")->pipeline;
		}

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		uint32_t uniform_offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &get_current_frame().globalDescriptor, 0, nullptr);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &get_current_frame().lightDescriptor, 1, &uniform_offset);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 2, 1, &get_current_frame().objectDescriptor, 0, nullptr);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 3, 1, &_shadowMap, 0, nullptr);

		if (object.material->textureSet != VK_NULL_HANDLE) {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 4, 1, &object.material->textureSet, 0, nullptr);
		}

		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer, &offset);

		vkCmdDraw(cmd, object.mesh->_vertices.size(), 1, 0, i);
	}




}

void VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject* first, int count)
{
	int frameIndex = _frameNumber % FRAME_OVERLAP;

	Mesh* lastMesh = nullptr;
	Material* lastMaterial = nullptr;

	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];

		if (object.material != lastMaterial) {

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
			lastMaterial = object.material;

			uint32_t uniform_offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 0, 1, &get_current_frame().globalDescriptor, 0, nullptr);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 1, 1, &get_current_frame().lightDescriptor, 1, &uniform_offset);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 2, 1, &get_current_frame().objectDescriptor, 0, nullptr);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 3, 1, &_shadowMap, 0, nullptr);

			if (object.material->textureSet != VK_NULL_HANDLE) {
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 4, 1, &object.material->textureSet, 0, nullptr);
				if (object.material == get_material(TEXTURE_NAMES[2])) {
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 5, 1, &_gBuffer, 0, nullptr);
				}
			}

			
		}


		if (object.mesh != lastMesh) {
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer, &offset);
			lastMesh = object.mesh;
		}

		vkCmdDraw(cmd, object.mesh->_vertices.size(), 1, 0, i);
	}
}

void VulkanEngine::init_scene()
{
	std::vector<std::string> nameVec = {
	"Glass_border_Cube.005",
		"Glass_Cube.004",
		"Table_Leg.001_Cube.009",
		"Table_Leg.002_Cube.010", 
		"Table_Leg.003_Cube.006", 
		"Table_Leg.004_Cube.011", 
		"Chair_Cube.007", 
		"Seat_Cube.001", 
		"BackRest_Cube.008", 
		"Chair.001_Cube.015", 
		"Seat.001_Cube.016", 
		"BackRest.001_Cube.017", 
		"Chair.002_Cube.018", 
		"Seat.002_Cube.019", 
		"BackRest.002_Cube.020", 
		"Chair.003_Cube.022", 
		"BackRest.003_Cube.023", 
		"Seat.003_Cube.021", 
		"Plane_Plane.002", 
		"Plane.001",
		"Plane.002_Plane.003"

	};

	const std::unordered_map<std::string, size_t> texMap = {
	{"Glass_border_Cube.005", 1 },
		{"Glass_Cube.004", 2},
		{"Table_Leg.001_Cube.009", 1},
		{"Table_Leg.002_Cube.010", 1},
		{"Table_Leg.003_Cube.006", 1},
		{"Table_Leg.004_Cube.011", 1},
		{"Chair_Cube.007", 1},
		{"Seat_Cube.001", -1},
		{"BackRest_Cube.008", -1},
		{"Chair.001_Cube.015", 1},
		{"Seat.001_Cube.016", -1},
		{"BackRest.001_Cube.017", -1},
		{"Chair.002_Cube.018", 1},
		{"Seat.002_Cube.019", -1},
		{"BackRest.002_Cube.020", -1},
		{"Chair.003_Cube.022", 1},
		{"BackRest.003_Cube.023", -1},
		{"Seat.003_Cube.021", -1},
		{"Plane_Plane.002", 0},
		{"Plane.001", -1},
		{"Plane.002_Plane.003", -1},

	};
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
	for (auto n : nameVec) {
		RenderObject obj;
		obj.mesh = get_mesh(n);
		if (texMap.at(n)!=-1) obj.material = get_material(TEXTURE_NAMES[texMap.at(n)]);
		else obj.material = get_material("defaultmesh");
		obj.transformMatrix = glm::mat4(1.0f);
		_renderables.push_back(obj);
	}


	Material* texturedMat = get_material("texturedmesh");

	_descriptorAllocator->allocate(&texturedMat->textureSet, _singleTextureSetLayout);

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);
	VkSamplerCreateInfo shadowSamplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);
	VkSampler blockySampler;
	vkCreateSampler(_device, &samplerInfo, nullptr, &blockySampler);

	vkCreateSampler(_device, &shadowSamplerInfo, nullptr, &_shadowSampler);

	_mainDeletionQueue.push_function([=]() {
		vkDestroySampler(_device, blockySampler, nullptr);
		vkDestroySampler(_device, _shadowSampler, nullptr);
		});
	
	VkDescriptorImageInfo imageBufferInfo;
	imageBufferInfo.sampler = blockySampler;
	imageBufferInfo.imageView = _loadedTextures["empire_diffuse"].imageView;
	imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;	
	
	vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
	.bind_image(0, &imageBufferInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
	.build(texturedMat->textureSet);

	for (auto name : TEXTURE_NAMES) {
		Material* texMat = get_material(name);
		_descriptorAllocator->allocate(&texMat->textureSet, _singleTextureSetLayout);

		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = blockySampler;
		imageBufferInfo.imageView = _loadedTextures[name].imageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_image(0, &imageBufferInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(texMat->textureSet);
	}

	_descriptorAllocator->allocate(&_shadowMap, _singleTextureSetLayout);

	VkDescriptorImageInfo shadowBufferInfo;
	shadowBufferInfo.sampler = _shadowSampler;
	shadowBufferInfo.imageView = _shadowColorImageView;
	shadowBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
		.bind_image(0, &shadowBufferInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.build(_shadowMap);

	_descriptorAllocator->allocate(&_gBuffer, _gBufferSetLayout);

	VkDescriptorImageInfo gBufferInfo1 = vkinit::descriptor_image_info(blockySampler, _gBufferPosImageView);
	VkDescriptorImageInfo gBufferInfo2 = vkinit::descriptor_image_info(blockySampler, _gBufferColorImageView);
	VkDescriptorImageInfo gBufferInfo3 = vkinit::descriptor_image_info(blockySampler, _gBufferNormalImageView);
	VkDescriptorImageInfo gBufferInfo4 = vkinit::descriptor_image_info(blockySampler, _gBufferVisibilityImageView);
	VkDescriptorImageInfo gBufferInfo5 = vkinit::descriptor_image_info(blockySampler, _gBufferDepthRGBAImageView);

	vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
		.bind_image(0, &gBufferInfo1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.bind_image(1, &gBufferInfo2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.bind_image(2, &gBufferInfo3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.bind_image(3, &gBufferInfo4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.bind_image(4, &gBufferInfo5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.build(_gBuffer);

}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;
	bufferInfo.size = allocSize;

	bufferInfo.usage = usage;

	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = memoryUsage;

	AllocatedBuffer newBuffer;

	VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo,
		&newBuffer._buffer,
		&newBuffer._allocation,
		nullptr));

	return newBuffer;
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t originalSize)
{
	size_t minUboAlignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
	size_t alignedSize = originalSize;
	if (minUboAlignment > 0) {
		alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
	}
	return alignedSize;
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
	VkCommandBuffer cmd = _uploadContext._commandBuffer;
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));


	function(cmd);


	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = vkinit::submit_info(&cmd);


	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _uploadContext._uploadFence));

	vkWaitForFences(_device, 1, &_uploadContext._uploadFence, true, 9999999999);
	vkResetFences(_device, 1, &_uploadContext._uploadFence);

	vkResetCommandPool(_device, _uploadContext._commandPool, 0);
}

void VulkanEngine::init_descriptors()
{

	_descriptorAllocator = new vkutil::DescriptorAllocator{};
	_descriptorAllocator->init(_device);

	_descriptorLayoutCache = new vkutil::DescriptorLayoutCache{};
	_descriptorLayoutCache->init(_device);


	VkDescriptorSetLayoutBinding cameraBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	VkDescriptorSetLayoutBinding sceneBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1);
	VkDescriptorSetLayoutBinding bindings[] = { cameraBind,sceneBind };
	

	VkDescriptorSetLayoutCreateInfo setinfo = {};
	setinfo.bindingCount = 1;
	setinfo.flags = 0;
	setinfo.pNext = nullptr;
	setinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setinfo.pBindings = &cameraBind;

	_globalSetLayout = _descriptorLayoutCache->create_descriptor_layout(&setinfo);


	VkDescriptorSetLayoutBinding objectBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);

	VkDescriptorSetLayoutCreateInfo set2info = {};
	set2info.bindingCount = 1;
	set2info.flags = 0;
	set2info.pNext = nullptr;
	set2info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set2info.pBindings = &objectBind;

	_objectSetLayout = _descriptorLayoutCache->create_descriptor_layout(&set2info);

	
	VkDescriptorSetLayoutBinding textureBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);

	VkDescriptorSetLayoutBinding gBufferBind1 = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	VkDescriptorSetLayoutBinding gBufferBind2 = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1);
	VkDescriptorSetLayoutBinding gBufferBind3 = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2);
	VkDescriptorSetLayoutBinding gBufferBind4 = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3);
	VkDescriptorSetLayoutBinding gBufferBind5 = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4);
	VkDescriptorSetLayoutBinding gBufferBindings[] = { gBufferBind1,gBufferBind2,gBufferBind3,gBufferBind4,gBufferBind5 };

	VkDescriptorSetLayoutCreateInfo set3info = {};
	set3info.bindingCount = 1;
	set3info.flags = 0;
	set3info.pNext = nullptr;
	set3info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set3info.pBindings = &textureBind;

	_singleTextureSetLayout = _descriptorLayoutCache->create_descriptor_layout(&set3info);

	VkDescriptorSetLayoutCreateInfo set4info = {};
	set4info.bindingCount = 2;
	set4info.flags = 0;
	set4info.pNext = nullptr;
	set4info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set4info.pBindings = bindings;

	_lightSetLayout = _descriptorLayoutCache->create_descriptor_layout(&set4info);

	VkDescriptorSetLayoutCreateInfo set5info = {};
	set5info.bindingCount = 5;
	set5info.flags = 0;
	set5info.pNext = nullptr;
	set5info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set5info.pBindings = gBufferBindings;

	_gBufferSetLayout = _descriptorLayoutCache->create_descriptor_layout(&set5info);
	
	const size_t sceneParamBufferSize = FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData));

	_sceneParameterBuffer = create_buffer(sceneParamBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);


	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		_frames[i].cameraBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].lightBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		const int MAX_OBJECTS = 10000;
		_frames[i].objectBuffer = create_buffer(sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		_descriptorAllocator->allocate(&_frames[i].lightDescriptor, _lightSetLayout);
		_descriptorAllocator->allocate(&_frames[i].globalDescriptor, _globalSetLayout);
		_descriptorAllocator->allocate(&_frames[i].objectDescriptor, _objectSetLayout);

		VkDescriptorBufferInfo cameraInfo;
		cameraInfo.buffer = _frames[i].cameraBuffer._buffer;
		cameraInfo.offset = 0;
		cameraInfo.range = sizeof(GPUCameraData);

		VkDescriptorBufferInfo sceneInfo;
		sceneInfo.buffer = _sceneParameterBuffer._buffer;
		sceneInfo.offset = 0;
		sceneInfo.range = sizeof(GPUSceneData);

		VkDescriptorBufferInfo objectBufferInfo;
		objectBufferInfo.buffer = _frames[i].objectBuffer._buffer;
		objectBufferInfo.offset = 0;
		objectBufferInfo.range = sizeof(GPUObjectData) * MAX_OBJECTS;

		VkDescriptorBufferInfo lightBufferInfo;
		lightBufferInfo.buffer = _frames[i].lightBuffer._buffer;
		lightBufferInfo.offset = 0;
		lightBufferInfo.range = sizeof(GPUCameraData);

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_buffer(0, &lightBufferInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.bind_buffer(1, &sceneInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(_frames[i].lightDescriptor);

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_buffer(0, &cameraInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(_frames[i].globalDescriptor);

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_buffer(0, &objectBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
			.build(_frames[i].objectDescriptor);
	}

	_mainDeletionQueue.push_function([&]() {
	
		vmaDestroyBuffer(_allocator, _sceneParameterBuffer._buffer, _sceneParameterBuffer._allocation);


		for (int i = 0; i < FRAME_OVERLAP; i++)
		{
			vmaDestroyBuffer(_allocator, _frames[i].cameraBuffer._buffer, _frames[i].cameraBuffer._allocation);
			vmaDestroyBuffer(_allocator, _frames[i].lightBuffer._buffer, _frames[i].lightBuffer._allocation);
			vmaDestroyBuffer(_allocator, _frames[i].objectBuffer._buffer, _frames[i].objectBuffer._allocation);
		}
		});
		
}

void VulkanEngine::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    VulkanEngine* myEngine = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    glm::vec3 dir = myEngine->_foc - myEngine->_camPos;
    dir[1] = 0.0f;
    glm::vec3 forward = glm::normalize(dir);
    glm::vec3 right = glm::cross(forward, glm::vec3{0.0f, 1.0f, 0.0f});
    if (action == GLFW_REPEAT || action == GLFW_PRESS) {
        switch (key)
        {
        case GLFW_KEY_A:
            if (myEngine->_camPos.x - right.x > -40.0f &&
                myEngine->_camPos.x - right.x < 20.0f &&
                myEngine->_camPos.z - right.z > 10.0f &&
                myEngine->_camPos.z - right.z < 45.0f) {
                myEngine->_camPos -= right;
                myEngine->_foc -= right;
                myEngine->_oriFoc -= right;
            }
            break;
        case GLFW_KEY_D:
            if (myEngine->_camPos.x + right.x > -40.0f &&
                myEngine->_camPos.x + right.x < 20.0f &&
                myEngine->_camPos.z + right.z > 10.0f &&
                myEngine->_camPos.z + right.z < 45.0f) {
                myEngine->_camPos += right;
                myEngine->_foc += right;
                myEngine->_oriFoc += right;
            }
            break;
        case GLFW_KEY_W:
            if (myEngine->_camPos.x + forward.x > -40.0f &&
                myEngine->_camPos.x + forward.x < 20.0f &&
                myEngine->_camPos.z + forward.z > 10.0f &&
                myEngine->_camPos.z + forward.z < 45.0f) {
                myEngine->_camPos += forward;
                myEngine->_foc += forward;
                myEngine->_oriFoc += forward;
            }
            break;
        case GLFW_KEY_S:
            if (myEngine->_camPos.x - forward.x > -40.0f &&
                myEngine->_camPos.x - forward.x < 20.0f &&
                myEngine->_camPos.z - forward.z > 10.0f &&
                myEngine->_camPos.z - forward.z < 45.0f) {
                myEngine->_camPos -= forward;
                myEngine->_foc -= forward;
                myEngine->_oriFoc -= forward;
            }
            break;
        default:
            break;
        };
    }
}

void VulkanEngine::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    VulkanEngine* myEngine = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        myEngine->_posPress = glm::vec2(xpos, ypos);
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        myEngine->_horAngle = myEngine->_horAngle + myEngine->_horAngleOffset;
        myEngine->_verAngle = myEngine->_verAngle + myEngine->_verAngleOffset;
        myEngine->_horAngleOffset = 0.0f;
        myEngine->_verAngleOffset = 0.0f;
    }
}

void VulkanEngine::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    VulkanEngine* myEngine = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    if (state == GLFW_PRESS) {

        myEngine->_horAngleOffset = (myEngine->_posPress.x - xpos) * 45.0f / (float)myEngine->_windowExtent.width;
        myEngine->_verAngleOffset = (myEngine->_posPress.y - ypos) * 45.0f / (float)myEngine->_windowExtent.height;

        if (myEngine->_horAngle + myEngine->_horAngleOffset > 90.0f) {
            myEngine->_horAngleOffset = 90.0f - myEngine->_horAngle;
        }
        else if (myEngine->_horAngle + myEngine->_horAngleOffset < -90.0f) {
            myEngine->_horAngleOffset = -90.0f - myEngine->_horAngle;
        }


        if (myEngine->_verAngle + myEngine->_verAngleOffset > 30.0f) {
            myEngine->_verAngleOffset = 30.0f - myEngine->_verAngle;
        }
        else if (myEngine->_verAngle + myEngine->_verAngleOffset < -30.0f) {
            myEngine->_verAngleOffset = -30.0f - myEngine->_verAngle;
        }

        glm::vec3 trans = -myEngine->_camPos;
        glm::mat4 horizatal_rotate = glm::rotate(glm::mat4(1.0f), glm::radians(myEngine->_horAngle + myEngine->_horAngleOffset), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 vertical_rotate = glm::rotate(glm::mat4(1.0f), glm::radians(myEngine->_verAngle + myEngine->_verAngleOffset), glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec4 focal = (glm::translate(glm::mat4(1.0), -trans) * horizatal_rotate * vertical_rotate * glm::translate(glm::mat4(1.0), trans) * glm::vec4(myEngine->_oriFoc, 1.0f));
        focal /= focal.w;
        myEngine->_foc = glm::vec3(focal.x, focal.y, focal.z);
    }

}

void VulkanEngine::cursor_enter_callback(GLFWwindow* window, int entered)
{
    VulkanEngine* myEngine = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    if (!entered) {
        myEngine->_horAngle = myEngine->_horAngle + myEngine->_horAngleOffset;
        myEngine->_verAngle = myEngine->_verAngle + myEngine->_verAngleOffset;
        myEngine->_horAngleOffset = 0.0f;
        myEngine->_verAngleOffset = 0.0f;
    }

}

void VulkanEngine::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}