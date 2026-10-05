#include "backends/vulkan/vulkan_backend.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace blocktensor {

uint32_t VulkanBuffer::find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("VulkanBuffer: Failed to find suitable memory type!");
}

VulkanBuffer::VulkanBuffer(VkDevice device, VkPhysicalDevice physical_device,
                           size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location)
    : device_(device), bytes_(bytes), dt_(dt), usage_(usage), location_(location) {

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer_) != VK_SUCCESS) {
        throw std::runtime_error("VulkanBuffer: Failed to create buffer!");
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, buffer_, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;

    VkMemoryPropertyFlags props = 0;
    if (location == MemoryLocation::HOST_VISIBLE) {
        props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else {
        props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    try {
        alloc_info.memoryTypeIndex = find_memory_type(physical_device, mem_reqs.memoryTypeBits, props);
    } catch (...) {
        props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        alloc_info.memoryTypeIndex = find_memory_type(physical_device, mem_reqs.memoryTypeBits, props);
    }

    if (vkAllocateMemory(device_, &alloc_info, nullptr, &memory_) != VK_SUCCESS) {
        throw std::runtime_error("VulkanBuffer: Failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device_, buffer_, memory_, 0);
}

VulkanBuffer::~VulkanBuffer() {
    if (mapped_ptr_) {
        vkUnmapMemory(device_, memory_);
    }
    if (buffer_) {
        vkDestroyBuffer(device_, buffer_, nullptr);
    }
    if (memory_) {
        vkFreeMemory(device_, memory_, nullptr);
    }
}

void* VulkanBuffer::map() {
    if (!mapped_ptr_) {
        if (vkMapMemory(device_, memory_, 0, bytes_, 0, &mapped_ptr_) != VK_SUCCESS) {
            throw std::runtime_error("VulkanBuffer: Failed to map memory!");
        }
    }
    return mapped_ptr_;
}

void VulkanBuffer::unmap() {
    if (mapped_ptr_) {
        vkUnmapMemory(device_, memory_);
        mapped_ptr_ = nullptr;
    }
}

void VulkanBuffer::write(const void* src, size_t bytes, size_t offset) {
    void* ptr = map();
    std::memcpy(static_cast<char*>(ptr) + offset, src, bytes);
}

void VulkanBuffer::read(void* dst, size_t bytes, size_t offset) {
    void* ptr = map();
    std::memcpy(dst, static_cast<char*>(ptr) + offset, bytes);
}

VulkanPipeline::VulkanPipeline(VkDevice device, std::string name, VkPipelineLayout layout, VkPipeline pipeline, VkDescriptorSetLayout descriptor_set_layout)
    : device_(device), name_(std::move(name)), layout_(layout), pipeline_(pipeline), descriptor_set_layout_(descriptor_set_layout) {}

VulkanPipeline::~VulkanPipeline() {
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (descriptor_set_layout_) vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
}

VulkanBackend::VulkanBackend() {
    init_vulkan_instance();
    select_physical_device();
    create_logical_device();
    create_command_pool();
    create_descriptor_pool();
}

VulkanBackend::~VulkanBackend() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void VulkanBackend::init_vulkan_instance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "BlockTensor Runtime";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "BlockTensor Engine";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
        throw std::runtime_error("VulkanBackend: Failed to create Vulkan instance!");
    }
}

void VulkanBackend::select_physical_device() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        throw std::runtime_error("VulkanBackend: No GPUs with Vulkan support found!");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    physical_device_ = devices[0];
    for (const auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = dev;
            break;
        }
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);

    info_.name = props.deviceName;
    info_.vendor_id = props.vendorID;
    info_.device_id = props.deviceID;
    info_.is_discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    info_.vendor_name = (props.vendorID == 0x1002) ? "AMD" : "Generic/Mesa";
    info_.max_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
    info_.max_workgroup_size[0] = props.limits.maxComputeWorkGroupSize[0];
    info_.max_workgroup_size[1] = props.limits.maxComputeWorkGroupSize[1];
    info_.max_workgroup_size[2] = props.limits.maxComputeWorkGroupSize[2];
    info_.max_shared_memory_bytes = props.limits.maxComputeSharedMemorySize;

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
    uint64_t vram = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            vram += mem_props.memoryHeaps[i].size;
        }
    }
    info_.total_vram_bytes = vram;
}

void VulkanBackend::create_logical_device() {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family_ = i;
            break;
        }
    }

    if (compute_queue_family_ == VK_QUEUE_FAMILY_IGNORED) {
        throw std::runtime_error("VulkanBackend: No compute queue family found!");
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family_;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.queueCreateInfoCount = 1;

    if (vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_) != VK_SUCCESS) {
        throw std::runtime_error("VulkanBackend: Failed to create logical device!");
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
}

void VulkanBackend::create_command_pool() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        throw std::runtime_error("VulkanBackend: Failed to create command pool!");
    }
}

void VulkanBackend::create_descriptor_pool() {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 1000;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 200;

    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        throw std::runtime_error("VulkanBackend: Failed to create descriptor pool!");
    }
}

std::shared_ptr<IComputeBuffer> VulkanBackend::create_buffer(
    size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location) {
    return std::make_shared<VulkanBuffer>(device_, physical_device_, bytes, dt, usage, location);
}

std::shared_ptr<IComputePipeline> VulkanBackend::create_pipeline_from_spirv(
    const std::string& name,
    const std::vector<uint32_t>& spirv_code,
    size_t buffer_binding_count,
    size_t push_constant_size) {

    VkShaderModuleCreateInfo shader_module_info{};
    shader_module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    shader_module_info.pCode = spirv_code.data();

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &shader_module_info, nullptr, &shader_module) != VK_SUCCESS) {
        throw std::runtime_error("VulkanBackend: Failed to create shader module!");
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(buffer_binding_count);
    for (size_t i = 0; i < buffer_binding_count; ++i) {
        bindings[i].binding = static_cast<uint32_t>(i);
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    VkDescriptorSetLayout descriptor_set_layout;
    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, shader_module, nullptr);
        throw std::runtime_error("VulkanBackend: Failed to create descriptor set layout!");
    }

    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = static_cast<uint32_t>(push_constant_size);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
    if (push_constant_size > 0) {
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;
    }

    VkPipelineLayout pipeline_layout;
    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout, nullptr);
        vkDestroyShaderModule(device_, shader_module, nullptr);
        throw std::runtime_error("VulkanBackend: Failed to create pipeline layout!");
    }

    VkComputePipelineCreateInfo pipeline_create_info{};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_create_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_create_info.stage.module = shader_module;
    pipeline_create_info.stage.pName = "main";
    pipeline_create_info.layout = pipeline_layout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout, nullptr);
        vkDestroyShaderModule(device_, shader_module, nullptr);
        throw std::runtime_error("VulkanBackend: Failed to create compute pipeline!");
    }

    vkDestroyShaderModule(device_, shader_module, nullptr);

    return std::make_shared<VulkanPipeline>(device_, name, pipeline_layout, pipeline, descriptor_set_layout);
}

void VulkanBackend::dispatch(
    const std::shared_ptr<IComputePipeline>& pipeline,
    const std::vector<std::shared_ptr<IComputeBuffer>>& buffers,
    const DispatchDimensions& dims,
    const void* push_constants,
    size_t push_constants_size) {

    auto vk_pipeline = std::dynamic_pointer_cast<VulkanPipeline>(pipeline);
    if (!vk_pipeline) {
        throw std::runtime_error("VulkanBackend::dispatch invalid pipeline");
    }

    VkDescriptorSetLayout set_layout = vk_pipeline->descriptor_set_layout();
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout;

    VkDescriptorSet descriptor_set;
    if (vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set) != VK_SUCCESS) {
        throw std::runtime_error("VulkanBackend::dispatch failed to allocate descriptor set!");
    }

    std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> write_sets(buffers.size());

    for (size_t i = 0; i < buffers.size(); ++i) {
        auto vk_buf = std::dynamic_pointer_cast<VulkanBuffer>(buffers[i]);
        if (!vk_buf) {
            throw std::runtime_error("VulkanBackend::dispatch non-VulkanBuffer passed");
        }

        buffer_infos[i].buffer = vk_buf->vk_buffer();
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = vk_buf->size();

        write_sets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_sets[i].dstSet = descriptor_set;
        write_sets[i].dstBinding = static_cast<uint32_t>(i);
        write_sets[i].dstArrayElement = 0;
        write_sets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write_sets[i].descriptorCount = 1;
        write_sets[i].pBufferInfo = &buffer_infos[i];
    }

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(write_sets.size()), write_sets.data(), 0, nullptr);

    VkCommandBufferAllocateInfo cmd_alloc_info{};
    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc_info.commandPool = command_pool_;
    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd_buffer;
    vkAllocateCommandBuffers(device_, &cmd_alloc_info, &cmd_buffer);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd_buffer, &begin_info);
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline->Layout());
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline->pipeline_layout(), 0, 1, &descriptor_set, 0, nullptr);

    if (push_constants && push_constants_size > 0) {
        vkCmdPushConstants(cmd_buffer, vk_pipeline->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(push_constants_size), push_constants);
    }

    vkCmdDispatch(cmd_buffer, dims.x, dims.y, dims.z);
    vkEndCommandBuffer(cmd_buffer);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;

    vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(compute_queue_);

    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
    vkFreeDescriptorSets(device_, descriptor_pool_, 1, &descriptor_set);
}

void VulkanBackend::synchronize() {
    if (compute_queue_) {
        vkQueueWaitIdle(compute_queue_);
    }
}

} // namespace blocktensor
