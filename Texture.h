#pragma once
#include "vk_types.h"
#include "vk_engine.h"

namespace vkutil {

	bool load_image_from_file(VulkanEngine& engine, std::string file, AllocatedImage& outImage);

}
