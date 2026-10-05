#pragma once

#include "blocktensor/backend.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace blocktensor {

class VulkanBuffer : public IComputeBuffer {
public:
    VulkanBuffer(VkDevice device, VkPhysicalDevice physical_device,
                 size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location);
    ~VulkanBuffer() override;

    size_t size() const override { return bytes_; }
    DataType data_type() const override { return dt_; }
    BufferUsage usage() const override { return usage_; }
    MemoryLocation location() const override { return location_; }

    void* map() override;
    void unmap() override;
    void write(const void* src, size_t bytes, size_t offset = 0) override;
    void read(void* dst, size_t bytes, size_t offset = 0) override;

    VkBuffer vk_buffer() const { return buffer_; }

private:
    uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties);

    VkDevice device_{VK_NULL_HANDLE};
    VkBuffer buffer_{VK_NULL_HANDLE};
    VkDeviceMemory memory_{VK_NULL_HANDLE};
    size_t bytes_{0};
    DataType dt_;
    BufferUsage usage_;
    MemoryLocation location_;
    void* mapped_ptr_{nullptr};
};

class VulkanPipeline : public IComputePipeline {
public:
    VulkanPipeline(VkDevice device, std::string name, VkPipelineLayout layout, VkPipeline pipeline, VkDescriptorSetLayout descriptor_set_layout);
    ~VulkanPipeline() override;

    const std::string& name() const override { return name_; }
    VkPipeline Layout() const { return pipeline_; }
    VkPipelineLayout pipeline_layout() const { return layout_; }
    VkDescriptorSetLayout descriptor_set_layout() const { return descriptor_set_layout_; }

private:
    VkDevice device_{VK_NULL_HANDLE};
    std::string name_;
    VkPipelineLayout layout_{VK_NULL_HANDLE};
    VkPipeline pipeline_{VK_NULL_HANDLE};
    VkDescriptorSetLayout descriptor_set_layout_{VK_NULL_HANDLE};
};

class VulkanBackend : public IComputeBackend {
public:
    VulkanBackend();
    ~VulkanBackend() override;

    std::string backend_name() const override { return "Vulkan_Compute"; }
    const DeviceInfo& device_info() const override { return info_; }

    std::shared_ptr<IComputeBuffer> create_buffer(
        size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location) override;

    std::shared_ptr<IComputePipeline> create_pipeline_from_spirv(
        const std::string& name,
        const std::vector<uint32_t>& spirv_code,
        size_t buffer_binding_count,
        size_t push_constant_size = 0) override;

    void dispatch(
        const std::shared_ptr<IComputePipeline>& pipeline,
        const std::vector<std::shared_ptr<IComputeBuffer>>& buffers,
        const DispatchDimensions& dims,
        const void* push_constants = nullptr,
        size_t push_constants_size = 0) override;

    void synchronize() override;

private:
    void init_vulkan_instance();
    void select_physical_device();
    void create_logical_device();
    void create_command_pool();
    void create_descriptor_pool();

    VkInstance instance_{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue compute_queue_{VK_NULL_HANDLE};
    uint32_t compute_queue_family_{VK_QUEUE_FAMILY_IGNORED};
    VkCommandPool command_pool_{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    DeviceInfo info_;
};

} // namespace blocktensor
