#include "backends/cpu/cpu_backend.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace blocktensor {

size_t get_data_type_size(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::INT32:   return 4;
        case DataType::INT8:    return 1;
        case DataType::INT4:    return 1;
        default: return 1;
    }
}

CPUBuffer::CPUBuffer(size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location)
    : data_(bytes, 0), dt_(dt), usage_(usage), location_(location) {}

void CPUBuffer::write(const void* src, size_t bytes, size_t offset) {
    if (offset + bytes > data_.size()) {
        throw std::runtime_error("CPUBuffer::write out of bounds");
    }
    std::memcpy(data_.data() + offset, src, bytes);
}

void CPUBuffer::read(void* dst, size_t bytes, size_t offset) {
    if (offset + bytes > data_.size()) {
        throw std::runtime_error("CPUBuffer::read out of bounds");
    }
    std::memcpy(dst, data_.data() + offset, bytes);
}

CPUBackend::CPUBackend() {
    info_.name = "Host CPU Reference Engine";
    info_.vendor_name = "Generic CPU";
    info_.vendor_id = 0;
    info_.device_id = 0;
    info_.is_discrete = false;
    info_.total_vram_bytes = 16ULL * 1024 * 1024 * 1024;
    info_.max_workgroup_size[0] = 1024;
    info_.max_workgroup_size[1] = 1024;
    info_.max_workgroup_size[2] = 64;
    info_.max_workgroup_invocations = 1024;
    info_.max_shared_memory_bytes = 65536;
}

std::shared_ptr<IComputeBuffer> CPUBackend::create_buffer(
    size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location) {
    return std::make_shared<CPUBuffer>(bytes, dt, usage, location);
}

void CPUBackend::register_cpu_kernel(const std::string& name, CPUShaderFunc func) {
    kernel_registry_[name] = func;
}

std::shared_ptr<IComputePipeline> CPUBackend::create_pipeline_from_spirv(
    const std::string& name,
    const std::vector<uint32_t>& /*spirv_code*/,
    size_t /*buffer_binding_count*/,
    size_t /*push_constant_size*/) {
    auto it = kernel_registry_.find(name);
    if (it != kernel_registry_.end()) {
        return std::make_shared<CPUPipeline>(name, it->second);
    }
    return std::make_shared<CPUPipeline>(name, [](
        const std::vector<std::shared_ptr<CPUBuffer>>&,
        const DispatchDimensions&,
        const void*) {
    });
}

void CPUBackend::dispatch(
    const std::shared_ptr<IComputePipeline>& pipeline,
    const std::vector<std::shared_ptr<IComputeBuffer>>& buffers,
    const DispatchDimensions& dims,
    const void* push_constants,
    size_t /*push_constants_size*/) {

    auto cpu_pipeline = std::dynamic_pointer_cast<CPUPipeline>(pipeline);
    if (!cpu_pipeline) {
        throw std::runtime_error("CPUBackend::dispatch invalid pipeline type");
    }

    std::vector<std::shared_ptr<CPUBuffer>> cpu_buffers;
    for (const auto& buf : buffers) {
        auto cpu_buf = std::dynamic_pointer_cast<CPUBuffer>(buf);
        if (!cpu_buf) {
            throw std::runtime_error("CPUBackend::dispatch non-CPUBuffer passed");
        }
        cpu_buffers.push_back(cpu_buf);
    }

    cpu_pipeline->func()(cpu_buffers, dims, push_constants);
}

} // namespace blocktensor
