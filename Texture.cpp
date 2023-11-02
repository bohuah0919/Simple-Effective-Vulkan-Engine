#include "Texture.h"
#include <iostream>

#include "vk_initializers.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


bool vkutil::load_image_from_file(VulkanEngine& engine, std::string file, AllocatedImage& outImage)
{
	int texWidth, texHeight, texChannels;

	stbi_uc* pixels = stbi_load(file.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

	if (!pixels) {
		std::cout << "Failed to load texture file " << file << std::endl;
		return false;
	}

	void* pixel_ptr = pixels;
	VkDeviceSize imageSize = texWidth * texHeight * 4;

	VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB;

	AllocatedBuffer stagingBuffer = engine.create_buffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

	void* data;
	vmaMapMemory(engine._allocator, stagingBuffer._allocation, &data);

	memcpy(data, pixel_ptr, static_cast<size_t>(imageSize));

	vmaUnmapMemory(engine._allocator, stagingBuffer._allocation);

	stbi_image_free(pixels);

	VkExtent3D imageExtent;
	imageExtent.width = static_cast<uint32_t>(texWidth);
	imageExtent.height = static_cast<uint32_t>(texHeight);
	imageExtent.depth = 1;

	AllocatedImage newImage;
	newImage._mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
	VkImageCreateInfo dimg_info = vkinit::image_create_info(image_format,1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, imageExtent, VK_SAMPLE_COUNT_1_BIT, newImage._mipLevels);


	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	vmaCreateImage(engine._allocator, &dimg_info, &dimg_allocinfo, &newImage._image, &newImage._allocation, nullptr);

	engine.immediate_submit([&](VkCommandBuffer cmd) {
		VkImageSubresourceRange range;
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;

		VkImageMemoryBarrier imageBarrier_toTransfer = {};
		imageBarrier_toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

		imageBarrier_toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarrier_toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toTransfer.image = newImage._image;
		imageBarrier_toTransfer.subresourceRange = range;

		imageBarrier_toTransfer.srcAccessMask = 0;
		imageBarrier_toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toTransfer);

		VkBufferImageCopy copyRegion = {};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;

		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = imageExtent;

		vkCmdCopyBufferToImage(cmd, stagingBuffer._buffer, newImage._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		VkImageMemoryBarrier imageBarrier_toReadable = imageBarrier_toTransfer;

		imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

		imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageBarrier_toReadable.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);

		for (uint32_t i = 1; i < newImage._mipLevels; i++)
		{
			VkImageBlit imageBlit{};

			imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlit.srcSubresource.layerCount = 1;
			imageBlit.srcSubresource.mipLevel = i - 1;
			imageBlit.srcOffsets[1].x = int32_t(texWidth >> (i - 1));
			imageBlit.srcOffsets[1].y = int32_t(texHeight >> (i - 1));
			imageBlit.srcOffsets[1].z = 1;

			imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlit.dstSubresource.layerCount = 1;
			imageBlit.dstSubresource.mipLevel = i;
			imageBlit.dstOffsets[1].x = int32_t(texWidth >> i);
			imageBlit.dstOffsets[1].y = int32_t(texHeight >> i);
			imageBlit.dstOffsets[1].z = 1;

			VkImageSubresourceRange mipSubRange = {};
			mipSubRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			mipSubRange.baseMipLevel = i;
			mipSubRange.levelCount = 1;
			mipSubRange.layerCount = 1;

			imageBarrier_toReadable.subresourceRange = mipSubRange;
			imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

			imageBarrier_toReadable.srcAccessMask = 0;
			imageBarrier_toReadable.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);


			vkCmdBlitImage(
				cmd,
				newImage._image,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				newImage._image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageBlit,
				VK_FILTER_LINEAR);
			
			imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

			imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imageBarrier_toReadable.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);

		}
		imageBarrier_toReadable.subresourceRange = range;
		imageBarrier_toReadable.subresourceRange.levelCount = newImage._mipLevels;
		imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);
		
		});


	engine._mainDeletionQueue.push_function([=]() {

		vmaDestroyImage(engine._allocator, newImage._image, newImage._allocation);
		});

	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(engine._chosenGPU, &properties);

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = (float)newImage._mipLevels;
	samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
	samplerInfo.anisotropyEnable = VK_TRUE;

	vkCreateSampler(engine._device, &samplerInfo, nullptr, &newImage._sampler);

	vmaDestroyBuffer(engine._allocator, stagingBuffer._buffer, stagingBuffer._allocation);

	std::cout << "Texture loaded succesfully " << file << std::endl;

	outImage = newImage;
	return true;


}