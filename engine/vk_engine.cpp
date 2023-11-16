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
	"assets/NY_City/City Block OBJ/Building textures/Building 1/Brownstone red_Color.png",
	"assets/NY_City/City Block OBJ/Building textures/Building 2/Building brick cornor_Color.png",
	 "assets/NY_City/City Block OBJ/Building textures/Building 3/Townhouse 001_color.png",
	 "assets/NY_City/City Block OBJ/Building textures/Building 4/Townhouse6_Color.png",
	"assets/NY_City/City Block OBJ/Building textures/Building 5/Brownstone tall_Color.png",
	 "assets/NY_City/City Block OBJ/Building textures/Building 6/Brick Building_Color.png",
	 "assets/NY_City/City Block OBJ/Building textures/Building 7/Building center_Color.png",
	 "assets/NY_City/City Block OBJ/Building textures/Building 9/Townhousecornor_color.png",
	"assets/NY_City/City Block OBJ/Building textures/Building 10/Townhouse Brownstone_Color.png",
	 "assets/NY_City/City Block OBJ/Maple tree/Bark.png",
	 "assets/NY_City/City Block OBJ/Maple tree/leaf2_Color.png",
	 "assets/NY_City/City Block OBJ/Road textures/Center road.png",
	 "assets/NY_City/City Block OBJ/Road textures/Road.png",
	 "assets/NY_City/City Block OBJ/Road textures/Side walk.png",
	"assets/front.png",
	"assets/left.png",
	"assets/right.png",
	"assets/back.png",
	"assets/top.png",
};

const std::vector<std::string> TEXTURE_NAMES =
{
	"Building 1",
	"Building 2",
	 "Building 3",
	 "Building 4",
	"Building 5",
	 "Building 6",
	 "Building 7",
	 "Building 9",
	 "Building 10",
	"Bark",
	"Leaf 2",
	"Center road",
	"Road",
	"Side walk",
	"Front",
	"Left",
	"Right",
	"Back",
	"Top"
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


	init_gbuffer_renderpass();

	init_default_renderpass();

	prepare_depthpass();

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

		for (int i = 0; i < FRAME_OVERLAP; i++)
		{
			_frames[i]._frameDeletionQueue.flush();
		}
		
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
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX, get_current_frame()._presentSemaphore, nullptr, &swapchainImageIndex));

	wait_for_drawing();

	update_descriptors(_renderables.data(), _renderables.size());

	prepare_light_culling();

	prepare_culling();


	VkCommandBuffer shadow_cmd = get_current_frame()._shadowCommandBuffer;
	VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

	prepare_shadowmap();

	VkSubmitInfo shadowSubmit = vkinit::submit_info(&shadow_cmd);

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &shadowSubmit, nullptr));

	render_scene(swapchainImageIndex);

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

void VulkanEngine::multithreading_draw()
{
	
	VkCommandBuffer shadow_cmd = get_current_frame()._shadowCommandBuffer;
	VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
	uint32_t swapchainImageIndex;
	
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX, get_current_frame()._presentSemaphore, nullptr, &swapchainImageIndex));

	wait_for_drawing();

	prepare_culling();
	prepare_light_culling();

	update_descriptors(_renderables.data(), _renderables.size());


	_threadpool.threads[0]->addJob([=] {

		prepare_shadowmap();

		});

	render_scene(swapchainImageIndex);

	_threadpool.wait();

	VkSubmitInfo shadowSubmit = vkinit::submit_info(&shadow_cmd);

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &shadowSubmit, nullptr));

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
		//draw();
		multithreading_draw();
	}

}

void VulkanEngine::wait_for_drawing() {
	VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, UINT64_MAX));
	VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

	VK_CHECK(vkResetCommandBuffer(get_current_frame()._cullShadowCommandBuffer, 0));
	VK_CHECK(vkResetCommandBuffer(get_current_frame()._cullCommandBuffer, 0));
	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));
	VK_CHECK(vkResetCommandBuffer(get_current_frame()._shadowCommandBuffer, 0));
}

void VulkanEngine::prepare_culling() {
	VkCommandBuffer cmd = get_current_frame()._cullShadowCommandBuffer;

	cmd = get_current_frame()._cullCommandBuffer;
	execute_culling(cmd, _renderables.data(), _renderables.size());

	VkSubmitInfo cullSubmit;
	cullSubmit = vkinit::submit_info(&cmd);
	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &cullSubmit, nullptr));
}

void VulkanEngine::prepare_light_culling() {
	VkCommandBuffer cmd = get_current_frame()._cullShadowCommandBuffer;
	for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
		execute_shadow_culling(cmd, _renderables.data(), _renderables.size(), i);
	}
	VkSubmitInfo cullSubmit;
	cullSubmit = vkinit::submit_info(&cmd);
	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &cullSubmit, nullptr));

}

void  VulkanEngine::prepare_shadowmap() {

	VkCommandBuffer cmd = get_current_frame()._shadowCommandBuffer;

	VkClearValue clearValue;
	clearValue.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.f;

	VkClearValue clearValues[] = { clearValue, depthClear };

	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));


	for (size_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
		VkRenderPassBeginInfo sdrpInfo = vkinit::renderpass_begin_info(_depthPass, _shadowExtent, get_current_frame().cascades[i].frameBuffer);

		VkClearValue clearValues[1];
		clearValues[0].depthStencil = { 1.0f, 0 };

		sdrpInfo.clearValueCount = 1;
		sdrpInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(cmd, &sdrpInfo, VK_SUBPASS_CONTENTS_INLINE);

		update_csm(cmd, _renderables.data(), _renderables.size(), i);

		vkCmdEndRenderPass(cmd);

	}
	VK_CHECK(vkEndCommandBuffer(cmd));
}



void  VulkanEngine::render_scene(uint32_t swapchainImageIndex) {

	VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

	VkClearValue clearValue;
	clearValue.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.f;

	VkClearValue clearValues[] = { clearValue, depthClear };

	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_renderPass, _windowExtent, _framebuffers[swapchainImageIndex]);

	rpInfo.clearValueCount = 2;

	rpInfo.pClearValues = &clearValues[0];

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	draw_objects(cmd, _renderables.data(), _renderables.size());
	vkCmdEndRenderPass(cmd);
	VK_CHECK(vkEndCommandBuffer(cmd));

}

FrameData& VulkanEngine::get_current_frame()
{
	return _frames[_frameNumber % FRAME_OVERLAP];
}


FrameData& VulkanEngine::get_last_frame()
{
	return _frames[(_frameNumber - 1) % FRAME_OVERLAP];
}

FrameData& VulkanEngine::get_next_frame()
{
	return _frames[(_frameNumber + 1) % FRAME_OVERLAP];
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
	VkPhysicalDeviceFeatures feats{};

	feats.pipelineStatisticsQuery = true;
	feats.multiDrawIndirect = true;
	feats.drawIndirectFirstInstance = true;
	feats.samplerAnisotropy = true;
	selector.set_required_features(feats);

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
		.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
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

	VkExtent3D windowImageExtent = {
		_windowExtent.width,
		_windowExtent.height,
		1
	};

	_depthFormat = VK_FORMAT_D32_SFLOAT;
	_shadowMapFormat = VK_FORMAT_R8G8B8A8_SRGB;

	VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat,1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, windowImageExtent, _msaaSamples,1);

	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image, &_depthImage._allocation, nullptr);

	VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

	VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

	

	VmaAllocationCreateInfo cimg_allocinfo = {};
	cimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	cimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkImageCreateInfo cimg_info = vkinit::image_create_info(_swachainImageFormat,1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, windowImageExtent, _msaaSamples,1);

	vmaCreateImage(_allocator, &cimg_info, &cimg_allocinfo, &_colorImage._image, &_colorImage._allocation, nullptr);

	VkImageViewCreateInfo cview_info = vkinit::imageview_create_info(_swachainImageFormat, _colorImage._image, VK_IMAGE_ASPECT_COLOR_BIT,1);

	VK_CHECK(vkCreateImageView(_device, &cview_info, nullptr, &_colorImageView));


	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, _depthImageView, nullptr);
		vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);

		vkDestroyImageView(_device, _colorImageView, nullptr);
		vmaDestroyImage(_allocator, _colorImage._image, _colorImage._allocation);

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
	color_attachment.samples = _msaaSamples;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	depth_attachment.flags = 0;
	depth_attachment.format = _depthFormat;
	depth_attachment.samples = _msaaSamples;
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
	subpass.pResolveAttachments = &color_attachment_res_ref;

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

	VkAttachmentDescription attachments[3] = { color_attachment, depth_attachment, color_attachment_res };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 3;
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

	VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_renderPass, _windowExtent);

	const uint32_t swapchain_imagecount = _swapchainImages.size();
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);


	for (int i = 0; i < swapchain_imagecount; i++) {
		VkImageView attachments[3];
		attachments[0] = _colorImageView;
		attachments[1] = _depthImageView;
		attachments[2] = _swapchainImageViews[i];

		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 3;

		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
			vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
			});
	}

}

void VulkanEngine::init_commands()
{
	_threadpool.setThreadCount(1);
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);


	for (int i = 0; i < FRAME_OVERLAP; i++) {

		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._cullCommandPool));
		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._cullShadowCommandPool));
		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));
		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._shadowCommandPool));


		VkCommandBufferAllocateInfo cmdAllocInfo; 
		cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._cullCommandPool, 1);
		VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._cullCommandBuffer));

		cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._cullShadowCommandPool, 1);
		VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._cullShadowCommandBuffer));

		cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);
		VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));

		cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._shadowCommandPool, 1);
		VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._shadowCommandBuffer));


	}


	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
	VK_CHECK(vkCreateCommandPool(_device, &uploadCommandPoolInfo, nullptr, &_uploadContext._commandPool));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(_device, _uploadContext._commandPool, nullptr);
		for (int i = 0; i < FRAME_OVERLAP; i++) {
		vkDestroyCommandPool(_device, _frames[i]._cullCommandPool, nullptr);
		vkDestroyCommandPool(_device, _frames[i]._cullShadowCommandPool, nullptr);
		vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
		vkDestroyCommandPool(_device, _frames[i]._shadowCommandPool, nullptr);
		}
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


	PipelineBuilder pipelineBuilder;

	VkPipelineLayoutCreateInfo mesh_pipeline_layout_info = vkinit::pipeline_layout_create_info();

	VkPipelineLayoutCreateInfo scene_pipeline_layout_info = mesh_pipeline_layout_info;
	VkDescriptorSetLayout sceneLayouts[] = { _globalSetLayout, _cascadesSetLayout, _objectSetLayout, _csmSetLayout, _singleTextureSetLayout };

	scene_pipeline_layout_info.setLayoutCount = 5;
	scene_pipeline_layout_info.pSetLayouts = sceneLayouts;

	VkPipelineLayout scenePipeLayout;
	VK_CHECK(vkCreatePipelineLayout(_device, &scene_pipeline_layout_info, nullptr, &scenePipeLayout));

	VkPipelineLayoutCreateInfo textured_specular_pipeline_layout_info = mesh_pipeline_layout_info;

	VkPipelineLayoutCreateInfo shadow_pipeline_layout_info = mesh_pipeline_layout_info;
	VkDescriptorSetLayout shadowSetLayouts[] = { _lightSetLayout, _objectSetLayout };

	shadow_pipeline_layout_info.setLayoutCount = 2;
	shadow_pipeline_layout_info.pSetLayouts = shadowSetLayouts;

	VkPipelineLayout shadowPipeLayout;
	VK_CHECK(vkCreatePipelineLayout(_device, &shadow_pipeline_layout_info, nullptr, &shadowPipeLayout));

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
	pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_NONE;

	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info(_msaaSamples);

	pipelineBuilder._colorBlendAttachment.push_back(vkinit::color_blend_attachment_state());

	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);


	VertexInputDescription vertexDescription = Vertex::get_vertex_description();

	pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

	pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	VkShaderModule meshVertShader;
	if (!load_shader_module("shaders/shader.vert.spv", &meshVertShader))
	{
		std::cout << "Error when building the mesh vertex shader module" << std::endl;
	}

	VkShaderModule sceneShader;
	if (!load_shader_module("shaders/scene_shader.frag.spv", &sceneShader))
	{
		std::cout << "Error when building the scene shader module" << std::endl;
	}
	
	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, sceneShader));
	pipelineBuilder._pipelineLayout = scenePipeLayout;
	VkPipeline scenePipeline = pipelineBuilder.build_pipeline(_device, _renderPass);
	for (auto name : TEXTURE_NAMES) {
		create_material(scenePipeline, scenePipeLayout, name);
	}
	
	VkShaderModule skyboxVertShader;
	if (!load_shader_module("shaders/skybox_shader.vert.spv", &skyboxVertShader))
	{
		std::cout << "Error when building the skybox vertex shader module" << std::endl;
	}

	VkShaderModule skyboxShader;
	if (!load_shader_module("shaders/skybox_shader.frag.spv", &skyboxShader))
	{
		std::cout << "Error when building the skyboxshader module" << std::endl;
	}

	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, skyboxVertShader));

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, skyboxShader));
	pipelineBuilder._pipelineLayout = scenePipeLayout;
	VkPipeline skyboxPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);
	create_material(skyboxPipeline, scenePipeLayout, "Front");
	create_material(skyboxPipeline, scenePipeLayout, "Left");
	create_material(skyboxPipeline, scenePipeLayout, "Right");
	create_material(skyboxPipeline, scenePipeLayout, "Back");
	create_material(skyboxPipeline, scenePipeLayout, "Top");


	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info(VK_SAMPLE_COUNT_1_BIT);

	pipelineBuilder._viewport.width = (float)_shadowExtent.width;
	pipelineBuilder._viewport.height = (float)_shadowExtent.height;
	pipelineBuilder._scissor.extent = _shadowExtent;

	VkShaderModule directDepthShader;
	if (!load_shader_module("shaders/depth_shader.vert.spv", &directDepthShader))
	{
		std::cout << "Error when building the depthshader module" << std::endl;
	}
	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, directDepthShader));


	pipelineBuilder._pipelineLayout = shadowPipeLayout;
	pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_NONE;
	VkPipeline depthPipeline = pipelineBuilder.build_pipeline(_device, _depthPass);
	create_material(depthPipeline, shadowPipeLayout, "directDepth");


	vkDestroyShaderModule(_device, meshVertShader, nullptr);
	vkDestroyShaderModule(_device, skyboxVertShader, nullptr);
	vkDestroyShaderModule(_device, skyboxShader, nullptr);
	vkDestroyShaderModule(_device, sceneShader, nullptr);
	vkDestroyShaderModule(_device, directDepthShader, nullptr);


	_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(_device, skyboxPipeline, nullptr);
		vkDestroyPipeline(_device, depthPipeline, nullptr);
		vkDestroyPipeline(_device, scenePipeline, nullptr);


		vkDestroyPipelineLayout(_device, shadowPipeLayout, nullptr);
		vkDestroyPipelineLayout(_device, scenePipeLayout, nullptr);
		});

	if (!load_compute_shader("shaders/compute_culling.comp.spv"))
	{
		std::cout << "Error when building the culling shader module" << std::endl;
	}
}

bool VulkanEngine::load_compute_shader(const char* shaderPath)
{
	VkShaderModule computeModule;
	if (!load_shader_module(shaderPath, &computeModule))

	{
		std::cout << "Error when building compute shader shader module" << std::endl;
		return false;
	}

	VkPipelineLayoutCreateInfo compute_pipeline_layout_info = vkinit::pipeline_layout_create_info();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(CullConstants);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayout setLayouts[] = { _cullSetLayout };
	compute_pipeline_layout_info.setLayoutCount = 1;
	compute_pipeline_layout_info.pSetLayouts = setLayouts;
	compute_pipeline_layout_info.pPushConstantRanges = &push_constant;
	compute_pipeline_layout_info.pushConstantRangeCount = 1;

	VkPipelineLayout computePipeLayout;
	VK_CHECK(vkCreatePipelineLayout(_device, &compute_pipeline_layout_info, nullptr, &computePipeLayout));

	ComputePipelineBuilder computeBuilder;

	computeBuilder._pipelineLayout = computePipeLayout;
	computeBuilder._shaderStage = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeModule);

	VkPipeline computePipeline = computeBuilder.build_pipeline(_device);

	create_material(computePipeline, computePipeLayout, "culling");

	vkDestroyShaderModule(_device, computeModule, nullptr);

	_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(_device, computePipeline, nullptr);

		vkDestroyPipelineLayout(_device, computePipeLayout, nullptr);
		});

	return true;
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

VkPipeline ComputePipelineBuilder::build_pipeline(VkDevice device)
{

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.layout = _pipelineLayout;
	pipelineInfo.stage = _shaderStage;

	VkPipeline newPipeline;
	if (vkCreateComputePipelines(
		device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		std::cout << "failed to create pipline\n";
		return VK_NULL_HANDLE;
	}
	else
	{
		return newPipeline;
	}
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
	Mesh skyboxFront{};

	skyboxFront._vertices.resize(6);
	skyboxFront._vertices[0].position = { -200.f, -200.f, -200.0f };
	skyboxFront._vertices[1].position = { 200.f,-200.f, -200.0f };
	skyboxFront._vertices[2].position = { -200.f, 200.f, -200.0f };

	skyboxFront._vertices[0].uv = { 1.0f, 1.0f };
	skyboxFront._vertices[1].uv = { 0.0f, 1.0f };
	skyboxFront._vertices[2].uv = { 1.0f, 0.0f };

	skyboxFront._vertices[0].color = { 1.f,1.f, 1.0f };
	skyboxFront._vertices[1].color = { 1.f,1.f, 1.0f };
	skyboxFront._vertices[2].color = { 1.f,1.f, 1.0f };

	skyboxFront._vertices[3].position = { -200.f, 200.f, -200.0f };
	skyboxFront._vertices[4].position = { 200.f,-200.f, -200.0f };
	skyboxFront._vertices[5].position = { 200.f, 200.f, -200.0f };

	skyboxFront._vertices[3].uv = { 1.0f, 0.0f };
	skyboxFront._vertices[4].uv = { 0.0f, 1.0f };
	skyboxFront._vertices[5].uv = { 0.0f, 0.0f };

	skyboxFront._vertices[3].color = { 1.0f,1.0f, 1.0f };
	skyboxFront._vertices[4].color = { 1.0f,1.0f, 1.0f };
	skyboxFront._vertices[5].color = { 1.0f,1.0f, 1.0f };

	skyboxFront.sphereBound = { glm::vec3{0.0,0.0,200.0},glm::sqrt(200.0f * 200.0f * 2.0f) };
	for (uint32_t i = 0; i < 6; i++) {
		glm::mat4 rotationMat = glm::rotate(glm::mat4(1.0f), glm::radians(210.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		skyboxFront._vertices[i].position = glm::vec3(rotationMat * glm::vec4(skyboxFront._vertices[i].position, 1.0f));
		skyboxFront.sphereBound = { glm::vec3(rotationMat * glm::vec4(0.0,0.0,200.0,0.0f)),glm::sqrt(200.0f * 200.0f * 2.0f) };

		skyboxFront._vertices[i].color = { 1.0f,1.0f, 1.0f };
		skyboxFront._vertices[i].uv = skyboxFront._vertices[i].uv;
	}

	//left
	Mesh skyboxLeft{};
	skyboxLeft._vertices.resize(6);
	for (uint32_t i = 0; i < 6; i++) {
		glm::mat4 rotationMat = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		skyboxLeft._vertices[i].position = glm::vec3(rotationMat * glm::vec4(skyboxFront._vertices[i].position,1.0f));
		skyboxLeft.sphereBound = { glm::vec3(rotationMat * glm::vec4(glm::vec3(skyboxFront.sphereBound),1.0f)),glm::sqrt(200.0f * 200.0f * 2.0f) };

		skyboxLeft._vertices[i].color = { 1.0f,1.0f, 1.0f };
		skyboxLeft._vertices[i].uv = skyboxFront._vertices[i].uv;
	}
	//right
	Mesh skyboxRight{};
	skyboxRight._vertices.resize(6);
	for (uint32_t i = 0; i < 6; i++) {
		glm::mat4 rotationMat = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		skyboxRight._vertices[i].position = glm::vec3(rotationMat * glm::vec4(skyboxFront._vertices[i].position, 1.0f));
		skyboxRight.sphereBound = { glm::vec3(rotationMat * glm::vec4(glm::vec3(skyboxFront.sphereBound),1.0f)),glm::sqrt(200.0f * 200.0f * 2.0f) };

		skyboxRight._vertices[i].color = { 1.0f,1.0f, 1.0f };
		skyboxRight._vertices[i].uv = skyboxFront._vertices[i].uv;
	}
	//back
	Mesh skyboxBack{};
	skyboxBack._vertices.resize(6);
	for (uint32_t i = 0; i < 6; i++) {
		glm::mat4 rotationMat = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		skyboxBack._vertices[i].position = glm::vec3(rotationMat * glm::vec4(skyboxFront._vertices[i].position, 1.0f));
		skyboxBack.sphereBound = { glm::vec3(rotationMat * glm::vec4(glm::vec3(skyboxFront.sphereBound),1.0f)),glm::sqrt(200.0f * 200.0f * 2.0f) };

		skyboxBack._vertices[i].color = { 1.0f,1.0f, 1.0f };
		skyboxBack._vertices[i].uv = skyboxFront._vertices[i].uv;
	}
	//top
	Mesh skyboxTop{};
	skyboxTop._vertices.resize(6);
	for (uint32_t i = 0; i < 6; i++) {
		glm::vec4 axis = glm::rotate(glm::mat4(1.0f), glm::radians(210.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
		glm::mat4 rotationMat = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(axis));
		skyboxTop._vertices[i].position = glm::vec3(rotationMat * glm::vec4(skyboxFront._vertices[i].position, 1.0f));
		skyboxTop.sphereBound = { glm::vec3(rotationMat * glm::vec4(glm::vec3(skyboxFront.sphereBound),1.0f)),glm::sqrt(200.0f * 200.0f * 2.0f) };

		skyboxTop._vertices[i].color = { 1.0f,1.0f, 1.0f };
		skyboxTop._vertices[i].uv = skyboxFront._vertices[i].uv;
	}

	
	upload_mesh(skyboxFront);
	_meshes["front"] = skyboxFront;
	upload_mesh(skyboxLeft);
	_meshes["left"] = skyboxLeft;
	upload_mesh(skyboxRight);
	_meshes["right"] = skyboxRight;
	upload_mesh(skyboxBack);
	_meshes["back"] = skyboxBack;
	upload_mesh(skyboxTop);
	_meshes["top"] = skyboxTop;

	Mesh floor{};
	floor._vertices.resize(6);

	floor._vertices[0].position = { 50.f, -0.1f, -50.0f };
	floor._vertices[1].position = { -50.f, -0.1f, -50.0f };
	floor._vertices[2].position = { 50.f, -0.1f, 50.0f };

	floor._vertices[0].uv = { 0.0f, 0.05f };
	floor._vertices[1].uv = { 0.05f, 0.05f };
	floor._vertices[2].uv = { 0.0f, 0.0f };

	floor._vertices[0].normal = { 0.f,1.f, 0.0f };
	floor._vertices[1].normal = { 0.f,1.f, 0.0f };
	floor._vertices[2].normal = { 0.f,1.f, 0.0f };

	floor._vertices[0].color = { 1.f,1.f, 1.0f };
	floor._vertices[1].color = { 1.f,1.f, 1.0f };
	floor._vertices[2].color = { 1.f,1.f, 1.0f };

	floor._vertices[3].position = { 50.f, -0.1f, 50.0f };
	floor._vertices[4].position = { -50.f,-0.1f, -50.0f };
	floor._vertices[5].position = { -50.f, -0.1f, 50.0f };

	floor._vertices[3].uv = { 0.0f, 0.0f };
	floor._vertices[4].uv = { 0.05f, 0.05f };
	floor._vertices[5].uv = { 0.05f, 0.0f };

	floor._vertices[3].normal = { 0.f,1.f, 0.0f };
	floor._vertices[4].normal = { 0.f,1.f, 0.0f };
	floor._vertices[5].normal = { 0.f,1.f, 0.0f };

	floor._vertices[3].color = { 1.0f,1.0f, 1.0f };
	floor._vertices[4].color = { 1.0f,1.0f, 1.0f };
	floor._vertices[5].color = { 1.0f,1.0f, 1.0f };

	floor.sphereBound = { glm::vec3{0.0f,-0.1f,0.0f},glm::sqrt(100.0f * 100.0f * 2.0f) };

	upload_mesh(floor);
	_meshes["floor"] = floor;
	
	Meshes nyCity{};
	nyCity.load_from_obj("./assets/NY_City/City Block OBJ/City block.obj");

	for (auto m : nyCity._meshes) {
		upload_mesh(m);
		_meshes[m.name] = m;
	}
}

void VulkanEngine::load_images()
{

	
	for (int i = 0; i < TEXTURE_PATHS.size(); i++) {

		Texture tex;

		vkutil::load_image_from_file(*this, TEXTURE_PATHS[i], tex.image);

		VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_SRGB, tex.image._image, VK_IMAGE_ASPECT_COLOR_BIT, tex.image._mipLevels);
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

std::vector<IndirectBatch> VulkanEngine::compact_draws(RenderObject* objects, int count)
{
	std::vector<IndirectBatch> draws;

	IndirectBatch firstDraw;
	firstDraw.mesh = objects[0].mesh;
	firstDraw.material = objects[0].material;
	firstDraw.first = 0;
	firstDraw.count = 1;

	draws.push_back(firstDraw);

	for (int i = 0; i < count; i++)
	{
		bool sameMesh = objects[i].mesh == draws.back().mesh;
		bool sameMaterial = objects[i].material == draws.back().material;

		if (sameMesh && sameMaterial)
		{
			draws.back().count++;
		}
		else
		{
			IndirectBatch newDraw;
			newDraw.mesh = objects[i].mesh;
			newDraw.material = objects[i].material;
			newDraw.first = i;
			newDraw.count = 1;

			draws.push_back(newDraw);
		}
	}
	return draws;
};

void VulkanEngine::execute_shadow_culling(VkCommandBuffer cmd, RenderObject* first, int count, int cascadesIndex)
{
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	std::vector<IndirectBatch> draws = compact_draws(first, count);
	int frameIndex = _frameNumber % FRAME_OVERLAP;

	void* objectData;
	vmaMapMemory(_allocator, get_current_frame().objectBuffer._allocation, &objectData);

	GPUObjectData* objectSSBO = (GPUObjectData*)objectData;

	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];
		objectSSBO[i].modelMatrix = object.transformMatrix;
		objectSSBO[i].sphereBound = object.mesh->sphereBound;
	}

	vmaUnmapMemory(_allocator, get_current_frame().objectBuffer._allocation);

	void* instanceData;
	vmaMapMemory(_allocator, get_current_frame().instanceBuffer._allocation, &instanceData);

	GPUInstance* instanceSSBO = (GPUInstance*)instanceData;

	for (int i = 0; i < count; i++)
	{
		instanceSSBO[i].objectID = i;
	}
	
	vmaUnmapMemory(_allocator, get_current_frame().instanceBuffer._allocation);

	void* indirectShadowData;
	vmaMapMemory(_allocator, get_current_frame().indirectShadowBuffers[cascadesIndex]._allocation, &indirectShadowData);

	VkDrawIndirectCommand* drawShadowCommands = (VkDrawIndirectCommand*)indirectShadowData;

	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];
		drawShadowCommands[i].vertexCount = object.mesh->_vertices.size();
		drawShadowCommands[i].instanceCount = 1;
		drawShadowCommands[i].firstVertex = 0;
		drawShadowCommands[i].firstInstance = i;
	}

	vmaUnmapMemory(_allocator, get_current_frame().indirectShadowBuffers[cascadesIndex]._allocation);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, get_material("culling")->pipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, get_material("culling")->pipelineLayout, 0, 1, &get_current_frame().cullCascadeDescriptors[cascadesIndex], 0, nullptr);

	glm::vec3 verticalNormal = { 0.0f, 1.0f, 0.0f };
	glm::vec3 horizontalNormal = { 1.0f, 0.0f, 0.0f };
	float yTopBottom = verticalNormal.y;
	float zTopBottom = verticalNormal.z;
	float xLeftRight = horizontalNormal.x;
	float zLeftRight = horizontalNormal.z;

	CullConstants constants;
	constants.view = glm::lookAt(_lightPos, _lightFoc, glm::vec3(0.0f, 1.0f, 0.0f));
	constants.frustum = { zLeftRight, xLeftRight, zTopBottom, yTopBottom };
	constants.distance = get_current_frame().cascades[cascadesIndex].radius;
	constants.zfar = 200.0f;
	constants.znear = 0.5f;

	vkCmdPushConstants(cmd, get_material("culling")->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullConstants), &constants);

	int groupcount = ((count) / 256) + 1;

	vkCmdDispatch(cmd, groupcount, 1, 1);

	VK_CHECK(vkEndCommandBuffer(cmd));
}

void VulkanEngine::execute_culling(VkCommandBuffer cmd, RenderObject* first, int count)
{
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	void* indirectData;
	vmaMapMemory(_allocator, get_current_frame().indirectBuffer._allocation, &indirectData);

	VkDrawIndirectCommand* drawCommands = (VkDrawIndirectCommand*)indirectData;

	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];
		drawCommands[i].vertexCount = object.mesh->_vertices.size();
		drawCommands[i].instanceCount = 1;
		drawCommands[i].firstVertex = 0;
		drawCommands[i].firstInstance = i;
	}

	vmaUnmapMemory(_allocator, get_current_frame().indirectBuffer._allocation);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, get_material("culling")->pipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, get_material("culling")->pipelineLayout, 0, 1, &get_current_frame().cullDescriptor, 0, nullptr);

	glm::vec3 verticalNormal = { 0.0f, glm::cos(glm::radians(45.0f)), glm::sin(glm::radians(45.0f)) };
	glm::vec3 horizontalNormal = glm::normalize(glm::vec3{ glm::cos(glm::radians(45.0f)),
														   0.0f, 
														   glm::sin(glm::radians(45.0f)) * _windowExtent.width / (float)_windowExtent.height });
	float yTopBottom = verticalNormal.y;
	float zTopBottom = verticalNormal.z;
	float xLeftRight = horizontalNormal.x;
	float zLeftRight = horizontalNormal.z;

	CullConstants constants;
	constants.view = glm::lookAt(_camera._camPos, _camera._foc, glm::vec3(0.0f, 1.0f, 0.0f));
	constants.frustum = { zLeftRight, xLeftRight, zTopBottom, yTopBottom };
	constants.distance = 0.0f;
	constants.zfar = _camera.zFar;
	constants.znear = _camera.zNear;

	vkCmdPushConstants(cmd, get_material("culling")->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullConstants), &constants);

	int groupcount = ((count) / 256) + 1;

	vkCmdDispatch(cmd, groupcount, 1, 1);

	std::vector<VkBufferMemoryBarrier> barriers;
	VkBufferMemoryBarrier barrier = vkinit::buffer_barrier(get_current_frame().indirectBuffer._buffer, _graphicsQueueFamily);
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barriers.push_back(barrier);

	for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
		VkBufferMemoryBarrier barrier2 = vkinit::buffer_barrier(get_current_frame().indirectShadowBuffers[i]._buffer, _graphicsQueueFamily);
		barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barriers.push_back(barrier2);
	}
	

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, barriers.size(), barriers.data(), 0, nullptr);

	VK_CHECK(vkEndCommandBuffer(cmd));
}

void VulkanEngine::prepare_depthpass()
{

	VkAttachmentDescription depth_attachment = {};
	depth_attachment.flags = 0;
	depth_attachment.format = _depthFormat;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 0;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 0;
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].dstSubpass = 0;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 1;
	render_pass_info.pAttachments = &depth_attachment;
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;
	render_pass_info.dependencyCount = static_cast<uint32_t>(dependencies.size());
	render_pass_info.pDependencies = &dependencies[0];

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_depthPass));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _depthPass, nullptr);
		});

	VkExtent3D shadowExtent = {
		_shadowExtent.width,
		_shadowExtent.height,
		1
	};

	VmaAllocationCreateInfo img_allocinfo = {};
	img_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	img_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VkImageCreateInfo img_shadow_info = vkinit::image_create_info(_depthFormat, SHADOW_MAP_CASCADE_COUNT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, shadowExtent, VK_SAMPLE_COUNT_1_BIT,1);

	vmaCreateImage(_allocator, &img_shadow_info, &img_allocinfo, &_depth._image, &_depth._allocation, nullptr);

	VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(_depthFormat, _depth._image, VK_IMAGE_ASPECT_DEPTH_BIT,1);
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = SHADOW_MAP_CASCADE_COUNT;

	VK_CHECK(vkCreateImageView(_device, &viewInfo, nullptr, &_depthView));

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		for (uint32_t j = 0; j < SHADOW_MAP_CASCADE_COUNT; j++) {

			VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(_depthFormat, _depth._image, VK_IMAGE_ASPECT_DEPTH_BIT,1);
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = j;
			viewInfo.subresourceRange.layerCount = 1;

			VK_CHECK(vkCreateImageView(_device, &viewInfo, nullptr, &_frames[i].cascades[j].view));

			VkFramebufferCreateInfo framebufferInfo = vkinit::framebuffer_create_info(_depthPass, _shadowExtent);
			framebufferInfo.renderPass = _depthPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = &_frames[i].cascades[j].view;
			framebufferInfo.layers = 1;
			VK_CHECK(vkCreateFramebuffer(_device, &framebufferInfo, nullptr, &_frames[i].cascades[j].frameBuffer));
		

			_mainDeletionQueue.push_function([=]() {
				
				vkDestroyImageView(_device, _frames[i].cascades[j].view, nullptr);
				vkDestroyFramebuffer(_device, _frames[i].cascades[j].frameBuffer, nullptr);
				
				});
		}

	}
	_mainDeletionQueue.push_function([=]() {

		vkDestroyImageView(_device, _depthView, nullptr);
		vmaDestroyImage(_allocator, _depth._image, _depth._allocation);

		});	

}

void VulkanEngine::update_descriptors( RenderObject* first, int count)
{

	float cascadeSplits[SHADOW_MAP_CASCADE_COUNT];

	void* objectData;
	vmaMapMemory(_allocator, get_current_frame().objectBuffer._allocation, &objectData);

	GPUObjectData* objectSSBO = (GPUObjectData*)objectData;

	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];
		objectSSBO[i].modelMatrix = object.transformMatrix;
		objectSSBO[i].sphereBound = object.mesh->sphereBound;
	}

	vmaUnmapMemory(_allocator, get_current_frame().objectBuffer._allocation);
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	_sceneParameters.lightColor = { 3.0,3.0,3.0 };
	_sceneParameters.lightDir = glm::normalize(glm::vec4{ -0.3, 1.0, 1.0, 0.0 });
	_sceneParameters.zNear = 0.1f;
	_sceneParameters.zFar = 200.0f;

	_camera.zNear = 0.5f;
	_camera.zFar = 120.0f;

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

	glm::mat4 view = glm::lookAt(_camera._camPos, _camera._foc, glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), _windowExtent.width / (float)_windowExtent.height, 0.5f, 120.0f);
	projection[1][1] *= -1;
	glm::mat4 viewProjMat = projection * view;
	
	void* data;

	GPUCameraData camData;
	camData.pos = _camera._camPos;
	camData.viewproj = viewProjMat;
	camData.view = view;

	vmaMapMemory(_allocator, get_current_frame().cameraBuffer._allocation, &data);

	memcpy(data, &camData, sizeof(GPUCameraData));

	vmaUnmapMemory(_allocator, get_current_frame().cameraBuffer._allocation);

	void* skyboxData;

	glm::mat4 skyboxView = glm::mat4(glm::mat3(view)); 
	glm::mat4 skyboxProj = glm::perspective(glm::radians(45.0f), _windowExtent.width / (float)_windowExtent.height, 0.5f, 120.0f);
	skyboxProj[1][1] *= -1;
	GPUCameraData skybox;
	skybox.pos = _camera._camPos;
	skybox.viewproj = skyboxProj * skyboxView;
	skybox.view = skyboxView;

	vmaMapMemory(_allocator, get_current_frame().skyboxBuffer._allocation, &skyboxData);

	memcpy(skyboxData, &skybox, sizeof(GPUCameraData));

	vmaUnmapMemory(_allocator, get_current_frame().skyboxBuffer._allocation);

	float nearClip = _camera.zNear;
	float farClip = _camera.zFar;
	float clipRange = farClip - nearClip;

	float minZ = nearClip;
	float maxZ = nearClip + clipRange;
	float range = maxZ - minZ;
	float ratio = maxZ / minZ;

	float cascadeSplitLambda = 0.8f;
	for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
		float p = (i + 1) / static_cast<float>(SHADOW_MAP_CASCADE_COUNT);
		float log = minZ * std::pow(ratio, p);
		float uniform = minZ + range * p;
		float d = cascadeSplitLambda * (log - uniform) + uniform;
		cascadeSplits[i] = (d - nearClip) / clipRange;
	}

	float lastSplitDist = 0.0;
	for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
		float splitDist = cascadeSplits[i];

		glm::vec3 frustumCorners[8] = {
			glm::vec3(-1.0f,  1.0f, 0.0f),
			glm::vec3(1.0f,  1.0f, 0.0f),
			glm::vec3(1.0f, -1.0f, 0.0f),
			glm::vec3(-1.0f, -1.0f, 0.0f),
			glm::vec3(-1.0f,  1.0f,  1.0f),
			glm::vec3(1.0f,  1.0f,  1.0f),
			glm::vec3(1.0f, -1.0f,  1.0f),
			glm::vec3(-1.0f, -1.0f,  1.0f),
		};


		glm::mat4 invCam = glm::inverse(projection * view);
		for (uint32_t j = 0; j < 8; j++) {
			glm::vec4 invCorner = invCam * glm::vec4(frustumCorners[j], 1.0f);
			frustumCorners[j] = invCorner / invCorner.w;
		}

		for (uint32_t j = 0; j < 4; j++) {
			glm::vec3 dist = frustumCorners[j + 4] - frustumCorners[j];
			frustumCorners[j + 4] = frustumCorners[j] + (dist * splitDist);
			frustumCorners[j] = frustumCorners[j] + (dist * lastSplitDist);
		}

		glm::vec3 frustumCenter = glm::vec3(0.0f);
		for (uint32_t j = 0; j < 8; j++) {
			frustumCenter += frustumCorners[j];
		}
		frustumCenter /= 8.0f;

		float radius = 0.0f;
		for (uint32_t j = 0; j < 8; j++) {
			float distance = glm::length(frustumCorners[j] - frustumCenter);
			radius = glm::max(radius, distance);
		}


		glm::vec3 maxExtents = glm::vec3(radius);
		glm::vec3 minExtents = -maxExtents;

		glm::mat4 lightViewMatrix = glm::lookAt(frustumCenter + _sceneParameters.lightDir * (maxExtents + 40.0f), frustumCenter, glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 lightOrthoMatrix = glm::ortho(minExtents.x, maxExtents.x,  maxExtents.y, minExtents.y, 0.0f, maxExtents.z*2.0f+80.0f);

		glm::mat4 shadowMatrix = lightOrthoMatrix * lightViewMatrix;
		glm::vec4 shadowOrigin = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		shadowOrigin = shadowMatrix * shadowOrigin;
		shadowOrigin = shadowOrigin * (float)_shadowExtent.height / 2.0f;

		glm::vec4 roundedOrigin = glm::round(shadowOrigin);
		glm::vec4 roundOffset = roundedOrigin - shadowOrigin;
		roundOffset = roundOffset * 2.0f / (float)_shadowExtent.height;
		roundOffset.z = 0.0f;
		roundOffset.w = 0.0f;

		glm::mat4 shadowProj = lightOrthoMatrix;
		shadowProj[3] += roundOffset;
		lightOrthoMatrix = shadowProj;

		get_current_frame().cascades[i].radius = radius;
		get_current_frame().cascades[i].splitDepth = (_camera.zNear + splitDist * clipRange) * -1.0f;
		get_current_frame().cascades[i].viewProjMatrix = lightOrthoMatrix * lightViewMatrix;

		lastSplitDist = cascadeSplits[i];

		void* cascadeData;
		GPUCameraData lightData;
		lightData.pos = frustumCenter - _sceneParameters.lightDir * minExtents.z;
		lightData.viewproj = lightOrthoMatrix * lightViewMatrix;
		lightData.view = lightViewMatrix;

		vmaMapMemory(_allocator, get_current_frame().cascadesBuffers[i]._allocation, &cascadeData);

		memcpy(cascadeData, &lightData, sizeof(GPUCameraData));

		vmaUnmapMemory(_allocator, get_current_frame().cascadesBuffers[i]._allocation);
	}

	void* cascadesSetData;
	CascadesSet cascadesSet;
	for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
		uint32_t rowIndex = i / 4;
		uint32_t colIndex = i % 4;
		cascadesSet.cascadeSplits[rowIndex][colIndex] = get_current_frame().cascades[i].splitDepth;
		cascadesSet.cascadeViewProjMat[i] = get_current_frame().cascades[i].viewProjMatrix;
		cascadesSet.cascadeSizes[rowIndex][colIndex] = std::pow(2.0f * get_current_frame().cascades[i].radius, 2.0f);
	}
	
	vmaMapMemory(_allocator, get_current_frame().cascadesSetBuffer._allocation, &cascadesSetData);

	memcpy(cascadesSetData, &cascadesSet, sizeof(CascadesSet));

	vmaUnmapMemory(_allocator, get_current_frame().cascadesSetBuffer._allocation);
}


void VulkanEngine::update_csm(VkCommandBuffer cmd, RenderObject* first, int count, int cascadesIndex)
{

	int frameIndex = _frameNumber % FRAME_OVERLAP;

	std::vector<IndirectBatch> draws = compact_draws(first, count);

	VkPipeline pipeline = get_material("directDepth")->pipeline;
	VkPipelineLayout pipeline_layout = get_material("directDepth")->pipelineLayout;

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	uint32_t uniform_offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &get_current_frame().cascades[cascadesIndex].descriptorSet, 1, &uniform_offset);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &get_current_frame().objectDescriptor, 0, nullptr);

	for (IndirectBatch& draw : draws)
	{
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &draw.mesh->_vertexBuffer._buffer, &offset);

		VkDeviceSize indirect_offset = draw.first * sizeof(VkDrawIndirectCommand);
		uint32_t draw_stride = sizeof(VkDrawIndirectCommand);

		vkCmdDrawIndirect(cmd, get_current_frame().indirectShadowBuffers[cascadesIndex]._buffer, indirect_offset, draw.count, draw_stride);
	}
}

void VulkanEngine::draw_gbuffer(VkCommandBuffer cmd, RenderObject* first, int count)
{
	int frameIndex = _frameNumber % FRAME_OVERLAP;

	std::vector<IndirectBatch> draws = compact_draws(first, count);

	for (IndirectBatch& draw : draws)
	{
		VkPipelineLayout layout;
		VkPipeline pipeline;
		if (draw.material->textureSet != VK_NULL_HANDLE) {
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

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 3, 1, &get_current_frame()._csmSet, 0, nullptr);

		if (draw.material->textureSet != VK_NULL_HANDLE) {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 4, 1, &draw.material->textureSet, 0, nullptr);
		}

		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &draw.mesh->_vertexBuffer._buffer, &offset);

		VkDeviceSize indirect_offset = draw.first * sizeof(VkDrawIndirectCommand);
		uint32_t draw_stride = sizeof(VkDrawIndirectCommand);

		vkCmdDrawIndirect(cmd, get_current_frame().indirectBuffer._buffer, indirect_offset, draw.count, draw_stride);
	}
}


void VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject* first, int count)
{
	int frameIndex = _frameNumber % FRAME_OVERLAP;

	std::vector<IndirectBatch> draws = compact_draws(first, count);

	for (IndirectBatch& draw : draws)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline);

		uint32_t uniform_offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;
		
		if (draw.material->pipeline == get_material("Front")->pipeline) {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipelineLayout, 0, 1, &get_current_frame().skyboxDescriptor, 0, nullptr);
		}
		else {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipelineLayout, 0, 1, &get_current_frame().globalDescriptor, 0, nullptr);
		}
		

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipelineLayout, 1, 1, &get_current_frame().cascadesSetDescriptor, 1, &uniform_offset);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipelineLayout, 2, 1, &get_current_frame().objectDescriptor, 0, nullptr);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipelineLayout, 3, 1, &get_current_frame()._csmSet, 0, nullptr);

		if (draw.material->textureSet != VK_NULL_HANDLE) {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipelineLayout, 4, 1, &draw.material->textureSet, 0, nullptr);
			
		}	
		

		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &draw.mesh->_vertexBuffer._buffer, &offset);

		VkDeviceSize indirect_offset = draw.first * sizeof(VkDrawIndirectCommand);
		uint32_t draw_stride = sizeof(VkDrawIndirectCommand);

		vkCmdDrawIndirect(cmd, get_current_frame().indirectBuffer._buffer, indirect_offset, draw.count, draw_stride);
	}
	
}


void VulkanEngine::init_scene()
{
	

	std::vector<std::string> nameVec = {
	"Sidewalk_Plane",
	"road_2_Plane.002",
	"road_2.001_Plane.004",
	"Cube.010_Cube.019",
	"Cube.016_Cube.031",
	"Cube.020_Cube.035",
	"Cube.023_Cube.038",
	"Building_1_Plane.006",
	"Building_1.001_Plane.005",
	"Building_2_Cube.001",
	"Building_2.001_Cube.010",
	"Building_3_Cube.022",
	"Building_4_Cube.009",
	"Building_4.001_Cube.013",
	"Building_5_Cube.027",
	"Building_6_Cube.008",
	"Plane.005_Plane.017",
	"Cube.003_Cube.004",
	"Cube.005",
	"Cube.004_Cube.051",
	"Cube.009_Cube.021",
	"Cube.011_Cube.024",
	"Cube.012_Cube.025",
	"Cube.013_Cube.026",
	"Cube.014_Cube.029",
	"Cube.015_Cube.030",
	"Cube.017_Cube.032",
	"Cube.018_Cube.033",
	"Cube.019_Cube.034",
	"Cube.021_Cube.036",
	"Cube.022_Cube.037",
	"floor",
	"front",
	"left",
	"right",
	"back",
	"top"
	};
	

	const std::unordered_map<std::string, std::string> texMap = {
	{"Sidewalk_Plane", "Side walk" },
		{"road_2_Plane.002","Center road"},
		{"road_2.001_Plane.004", "Road"},
		{"Cube.010_Cube.019", "Bark"},
		{"Cube.016_Cube.031", "Bark"},
		{"Cube.020_Cube.035", "Bark"},
		{"Cube.023_Cube.038", "Bark"},
		{"Cylinder_Cylinder.001", "Stop light"},
		{"Cylinder.001_Cylinder.003", "Stop light"},
		{"Building_1_Plane.006", "Building 1"},
		{"Building_1.001_Plane.005", "Building 1"},
		{"Building_2_Cube.001", "Building 2"},
		{"Building_2.001_Cube.010","Building 2"},
		{"Building_3_Cube.022", "Building 3"},
		{"Building_4_Cube.009", "Building 4"},
		{"Building_4.001_Cube.013", "Building 4"},
		{"Building_5_Cube.027", "Building 5"},
		{"Building_6_Cube.008", "Building 6"},
		{"Plane.005_Plane.017", "Building 7"},
		{"Cube.001", "Building 8"},
		{"Cube.001_Cube.002", "Building 8"},
		{"Cube.002_Cube.003", "Building 8"},
		{"Cube.007_Cube.016", "Building 8"},
		{"Cube.008_Cube.018", "Building 8"},
		{"Cube.003_Cube.004", "Building 9"},
		{"Cube.005", "Building 9"},
		{"Cube.004_Cube.051", "Building 10"},
		{"Cube.006_Cube.012", "Building 11"},
		{"Cube.009_Cube.021", "Leaf 2"},
		{"Cube.011_Cube.024", "Leaf 2"},
		{"Cube.012_Cube.025", "Bark"},
		{"Cube.013_Cube.026", "Leaf 2"},
		{"Cube.014_Cube.029", "Leaf 2"},
		{"Cube.015_Cube.030", "Bark"},
		{"Cube.017_Cube.032", "Leaf 2"},
		{"Cube.018_Cube.033", "Leaf 2"},
		{"Cube.019_Cube.034", "Bark"},
		{"Cube.021_Cube.036", "Leaf 2"},
		{"Cube.022_Cube.037", "Bark"},
		{"floor","Side walk"},
		{"front","Front"},
		{"left","Left"},
		{"right","Right"},
		{"back","Back"},
		{"top","Top"}
		
	};

	for (auto n : nameVec) {
		RenderObject obj;
		obj.mesh = get_mesh(n);
		obj.material = get_material(texMap.at(n));
		obj.transformMatrix = glm::mat4(1.0f);
		_renderables.push_back(obj);
	}
	RenderObject obj1;
	obj1.mesh = get_mesh("Building_5_Cube.027");
	obj1.material = get_material(texMap.at("Building_5_Cube.027"));
	obj1.transformMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(15.0f,0.0f, 0.0f));
	_renderables.push_back(obj1);

	RenderObject obj2;
	obj2.mesh = get_mesh("Plane.005_Plane.017");
	obj2.material = get_material(texMap.at("Plane.005_Plane.017"));
	obj2.transformMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	_renderables.push_back(obj2);

	RenderObject front;
	front.mesh = get_mesh("front");
	front.material = get_material("Front");
	front.transformMatrix = glm::mat4(1.0f);
	_renderables.push_back(front);

	RenderObject left;
	left.mesh = get_mesh("left");
	left.material = get_material("Left");
	left.transformMatrix = glm::mat4(1.0f);
	_renderables.push_back(left);

	RenderObject right;
	right.mesh = get_mesh("right");
	right.material = get_material("Right");
	right.transformMatrix = glm::mat4(1.0f);
	_renderables.push_back(right);

	RenderObject back;
	back.mesh = get_mesh("back");
	back.material = get_material("Back");
	back.transformMatrix = glm::mat4(1.0f);
	_renderables.push_back(back);

	RenderObject top;
	top.mesh = get_mesh("top");
	top.material = get_material("Top");
	top.transformMatrix = glm::mat4(1.0f);
	_renderables.push_back(top);

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	VkSamplerCreateInfo shadowSamplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	shadowSamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

	VkSampler imgSampler;
	vkCreateSampler(_device, &samplerInfo, nullptr, &imgSampler);

	vkCreateSampler(_device, &shadowSamplerInfo, nullptr, &_shadowSampler);

	_mainDeletionQueue.push_function([=]() {
		vkDestroySampler(_device, imgSampler, nullptr);
		vkDestroySampler(_device, _shadowSampler, nullptr);
		});
	


	for (auto name : TEXTURE_NAMES) {
		Material* texMat = get_material(name);
		_descriptorAllocator->allocate(&texMat->textureSet, _singleTextureSetLayout);

		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = _loadedTextures[name].image._sampler;
		imageBufferInfo.imageView = _loadedTextures[name].imageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		std::string skyboxs[] = {"Front","Left","Right","Back","Top"};
		std::string* res = std::find(std::begin(skyboxs), std::end(skyboxs), name);

		if (res != std::end(skyboxs)) {
			imageBufferInfo.sampler = imgSampler;
		}

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_image(0, 1, &imageBufferInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(texMat->textureSet);
	}

	
	VkDescriptorImageInfo csmBufferInfo;
	
	csmBufferInfo.sampler = _shadowSampler;
	csmBufferInfo.imageView = _depthView;
	csmBufferInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
	
	for (int i = 0; i < FRAME_OVERLAP; i++) {
		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_image(0, 1, &csmBufferInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(_frames[i]._csmSet);
	}

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
	_cascadesSetLayout = _descriptorLayoutCache->create_descriptor_layout(&set4info);

	VkDescriptorSetLayoutCreateInfo set5info = {};
	set5info.bindingCount = 5;
	set5info.flags = 0;
	set5info.pNext = nullptr;
	set5info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set5info.pBindings = gBufferBindings;

	_gBufferSetLayout = _descriptorLayoutCache->create_descriptor_layout(&set5info);
	
	const size_t sceneParamBufferSize = FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData));

	_sceneParameterBuffer = create_buffer(sceneParamBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	VkDescriptorSetLayoutBinding cullBind1 = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);
	VkDescriptorSetLayoutBinding cullBind2 = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1);
	VkDescriptorSetLayoutBinding cullBind3 = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2);
	VkDescriptorSetLayoutBinding cullBindings[] = { cullBind1, cullBind2, cullBind3 };

	VkDescriptorSetLayoutCreateInfo set6info = {};
	set6info.bindingCount = 3;
	set6info.flags = 0;
	set6info.pNext = nullptr;
	set6info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set6info.pBindings = cullBindings;

	_cullSetLayout = _descriptorLayoutCache->create_descriptor_layout(&set6info);

	VkDescriptorSetLayoutBinding cascadeBindings = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	VkDescriptorSetLayoutCreateInfo csmSetInfo = {};
	csmSetInfo.bindingCount = 1;
	csmSetInfo.flags = 0;
	csmSetInfo.pNext = nullptr;
	csmSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	csmSetInfo.pBindings = &cascadeBindings;

	_csmSetLayout = _descriptorLayoutCache->create_descriptor_layout(&csmSetInfo);

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		_frames[i].cameraBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].lightBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].skyboxBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].cascadesSetBuffer = create_buffer(sizeof(CascadesSet), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		for (uint32_t j = 0; j < SHADOW_MAP_CASCADE_COUNT; j++) {
			_frames[i].cascadesBuffers[j] = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		}

		const int MAX_OBJECTS = 10000;
		_frames[i].objectBuffer = create_buffer(sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].instanceBuffer = create_buffer(sizeof(GPUInstance) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		const int MAX_COMMANDS = 10000;
		_frames[i].indirectBuffer = create_buffer(sizeof(VkDrawIndirectCommand) * MAX_COMMANDS, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		for (uint32_t j = 0; j < SHADOW_MAP_CASCADE_COUNT; j++) {
			_frames[i].indirectShadowBuffers[j] = create_buffer(sizeof(VkDrawIndirectCommand) * MAX_COMMANDS, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			_descriptorAllocator->allocate(&_frames[i].cullCascadeDescriptors[j], _cullSetLayout);
		}

		_descriptorAllocator->allocate(&_frames[i].lightDescriptor, _lightSetLayout);
		_descriptorAllocator->allocate(&_frames[i].globalDescriptor, _globalSetLayout);
		_descriptorAllocator->allocate(&_frames[i].objectDescriptor, _objectSetLayout);
		_descriptorAllocator->allocate(&_frames[i].cullDescriptor, _cullSetLayout);
		_descriptorAllocator->allocate(&_frames[i].cascadesSetDescriptor, _cascadesSetLayout);
		_descriptorAllocator->allocate(&_frames[i].skyboxDescriptor, _globalSetLayout);

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

		VkDescriptorBufferInfo instanceBufferInfo;
		instanceBufferInfo.buffer = _frames[i].instanceBuffer._buffer;
		instanceBufferInfo.offset = 0;
		instanceBufferInfo.range = sizeof(GPUInstance) * MAX_OBJECTS;

		VkDescriptorBufferInfo indirectBufferInfo;
		indirectBufferInfo.buffer = _frames[i].indirectBuffer._buffer;
		indirectBufferInfo.offset = 0;
		indirectBufferInfo.range = sizeof(VkDrawIndirectCommand) * MAX_OBJECTS;

		std::array<VkDescriptorBufferInfo,SHADOW_MAP_CASCADE_COUNT> indirectShadowBufferInfos;
		for (uint32_t j = 0; j < SHADOW_MAP_CASCADE_COUNT; j++) {
			indirectShadowBufferInfos[j].buffer = _frames[i].indirectShadowBuffers[j]._buffer;
			indirectShadowBufferInfos[j].offset = 0;
			indirectShadowBufferInfos[j].range = sizeof(VkDrawIndirectCommand) * MAX_OBJECTS;
		}
		

		VkDescriptorBufferInfo lightBufferInfo;
		lightBufferInfo.buffer = _frames[i].lightBuffer._buffer;
		lightBufferInfo.offset = 0;
		lightBufferInfo.range = sizeof(GPUCameraData);

		VkDescriptorBufferInfo skyboxBufferInfo;
		skyboxBufferInfo.buffer = _frames[i].skyboxBuffer._buffer;
		skyboxBufferInfo.offset = 0;
		skyboxBufferInfo.range = sizeof(GPUCameraData);

		VkDescriptorBufferInfo cascadesSetBufferInfo;
		cascadesSetBufferInfo.buffer = _frames[i].cascadesSetBuffer._buffer;
		cascadesSetBufferInfo.offset = 0;
		cascadesSetBufferInfo.range = sizeof(CascadesSet);


		std::array<VkDescriptorBufferInfo, SHADOW_MAP_CASCADE_COUNT> cascadesBufferInfo;
		for (uint32_t j = 0; j < SHADOW_MAP_CASCADE_COUNT; j++) {
			cascadesBufferInfo[j].buffer = _frames[i].cascadesBuffers[j]._buffer;
			cascadesBufferInfo[j].offset = 0;
			cascadesBufferInfo[j].range = sizeof(GPUCameraData);

			vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
				.bind_buffer(0, &cascadesBufferInfo[j], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
				.bind_buffer(1, &sceneInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
				.build(_frames[i].cascades[j].descriptorSet);
		}

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_buffer(0, &lightBufferInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.bind_buffer(1, &sceneInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(_frames[i].lightDescriptor);

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_buffer(0, &cascadesSetBufferInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.bind_buffer(1, &sceneInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(_frames[i].cascadesSetDescriptor);


		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_buffer(0, &cameraInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(_frames[i].globalDescriptor);

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_buffer(0, &skyboxBufferInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(_frames[i].skyboxDescriptor);

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_buffer(0, &objectBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
			.build(_frames[i].objectDescriptor);

		vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
			.bind_buffer(0, &objectBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
			.bind_buffer(1, &instanceBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
			.bind_buffer(2, &indirectBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
			.build(_frames[i].cullDescriptor);

		for (uint32_t j = 0; j < SHADOW_MAP_CASCADE_COUNT; j++) {
			vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, _descriptorAllocator)
				.bind_buffer(0, &objectBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
				.bind_buffer(1, &instanceBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
				.bind_buffer(2, &indirectShadowBufferInfos[j], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
				.build(_frames[i].cullCascadeDescriptors[j]);
		}

		
			
	}

	_mainDeletionQueue.push_function([&]() {
	
		vmaDestroyBuffer(_allocator, _sceneParameterBuffer._buffer, _sceneParameterBuffer._allocation);
		for (int i = 0; i < FRAME_OVERLAP; i++)
		{
			vmaDestroyBuffer(_allocator, _frames[i].cameraBuffer._buffer, _frames[i].cameraBuffer._allocation);
			vmaDestroyBuffer(_allocator, _frames[i].lightBuffer._buffer, _frames[i].lightBuffer._allocation);
			vmaDestroyBuffer(_allocator, _frames[i].objectBuffer._buffer, _frames[i].objectBuffer._allocation);
			vmaDestroyBuffer(_allocator, _frames[i].instanceBuffer._buffer, _frames[i].instanceBuffer._allocation);
			vmaDestroyBuffer(_allocator, _frames[i].skyboxBuffer._buffer, _frames[i].skyboxBuffer._allocation);
			vmaDestroyBuffer(_allocator, _frames[i].cascadesSetBuffer._buffer, _frames[i].cascadesSetBuffer._allocation);
			for (uint32_t j = 0; j < SHADOW_MAP_CASCADE_COUNT; j++) {
				vmaDestroyBuffer(_allocator, _frames[i].indirectShadowBuffers[j]._buffer, _frames[i].indirectShadowBuffers[j]._allocation);
				vmaDestroyBuffer(_allocator, _frames[i].cascadesBuffers[j]._buffer, _frames[i].cascadesBuffers[j]._allocation);
			}
			vmaDestroyBuffer(_allocator, _frames[i].indirectBuffer._buffer, _frames[i].indirectBuffer._allocation);
		}
	});
		

}

void VulkanEngine::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    VulkanEngine* myEngine = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    glm::vec3 dir = myEngine->_camera._foc - myEngine->_camera._camPos;
    dir[1] = 0.0f;
    glm::vec3 forward = glm::normalize(dir) * 0.25f;
    glm::vec3 right = glm::cross(glm::normalize(dir), glm::vec3{0.0f, 1.0f, 0.0f}) * 0.25f;
    if (action == GLFW_REPEAT || action == GLFW_PRESS) {
        switch (key)
        {
        case GLFW_KEY_A:
		
			myEngine->_camera._camPos -= right;
			myEngine->_camera._foc -= right;
			myEngine->_camera._oriFoc -= right;
            break;
        case GLFW_KEY_D:
            
			myEngine->_camera._camPos += right;
			myEngine->_camera._foc += right;
			myEngine->_camera._oriFoc += right;
            break;
        case GLFW_KEY_W:
            
			myEngine->_camera._camPos += forward;
			myEngine->_camera._foc += forward;
			myEngine->_camera._oriFoc += forward;
            break;
        case GLFW_KEY_S:
            
			myEngine->_camera._camPos -= forward;
			myEngine->_camera._foc -= forward;
			myEngine->_camera._oriFoc -= forward;
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
        myEngine->_camera._posPress = glm::vec2(xpos, ypos);
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        myEngine->_camera._horAngle = myEngine->_camera._horAngle + myEngine->_camera._horAngleOffset;
        myEngine->_camera._verAngle = myEngine->_camera._verAngle + myEngine->_camera._verAngleOffset;
        myEngine->_camera._horAngleOffset = 0.0f;
        myEngine->_camera._verAngleOffset = 0.0f;
    }
}

void VulkanEngine::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    VulkanEngine* myEngine = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    if (state == GLFW_PRESS) {

        myEngine->_camera._horAngleOffset = (myEngine->_camera._posPress.x - xpos) * 90.0f / (float)myEngine->_windowExtent.width;
        myEngine->_camera._verAngleOffset = (myEngine->_camera._posPress.y - ypos) * 90.0f / (float)myEngine->_windowExtent.height;


        if (myEngine->_camera._verAngle + myEngine->_camera._verAngleOffset > 30.0f) {
            myEngine->_camera._verAngleOffset = 30.0f - myEngine->_camera._verAngle;
        }
        else if (myEngine->_camera._verAngle + myEngine->_camera._verAngleOffset < -30.0f) {
            myEngine->_camera._verAngleOffset = -30.0f - myEngine->_camera._verAngle;
        }

        glm::vec3 trans = -myEngine->_camera._camPos;
        glm::mat4 horizatal_rotate = glm::rotate(glm::mat4(1.0f), glm::radians(myEngine->_camera._horAngle + myEngine->_camera._horAngleOffset), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 vertical_rotate = glm::rotate(glm::mat4(1.0f), glm::radians(myEngine->_camera._verAngle + myEngine->_camera._verAngleOffset), glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec4 focal = (glm::translate(glm::mat4(1.0), -trans) * horizatal_rotate * vertical_rotate * glm::translate(glm::mat4(1.0), trans) * glm::vec4(myEngine->_camera._oriFoc, 1.0f));
        focal /= focal.w;
        myEngine->_camera._foc = glm::vec3(focal.x, focal.y, focal.z);
    }

}

void VulkanEngine::cursor_enter_callback(GLFWwindow* window, int entered)
{
    VulkanEngine* myEngine = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    if (!entered) {
		int horAngleInt = static_cast<int>(myEngine->_camera._horAngle + myEngine->_camera._horAngleOffset);
        myEngine->_camera._horAngle = (myEngine->_camera._horAngle + myEngine->_camera._horAngleOffset) - horAngleInt/360 * 360.0f;
        myEngine->_camera._verAngle = myEngine->_camera._verAngle + myEngine->_camera._verAngleOffset;
        myEngine->_camera._horAngleOffset = 0.0f;
        myEngine->_camera._verAngleOffset = 0.0f;
    }

}

void VulkanEngine::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto myEngine = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    myEngine->framebufferResized = true;
}