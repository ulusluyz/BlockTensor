#pragma once

#include "blocktensor/backend.hpp"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

namespace blocktensor {

class CPUBuffer : public IComputeBuffer, public std::enable_shared_from_this<CPUBuffer> {
public:
    CPUBuffer(size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location);
    ~CPUBuffer() override = default;

    size_t size() const override { return data_.size(); }
    DataType data_type() const override { return dt_; }
    BufferUsage usage() const override { return usage_; }
    MemoryLocation location() const override { return location_; }

    void* map() override { return data_.data(); }
    void unmap() override {}
    void write(const void* src, size_t bytes, size_t offset = 0) override;
    void read(void* dst, size_t bytes, size_t offset = 0) override;

    uint8_t* raw_data() { return data_.data(); }
    const uint8_t* raw_data() const { return data_.data(); }

private:
    std::vector<uint8_t> data_;
    DataType dt_;
    BufferUsage usage_;
    MemoryLocation location_;
};

using CPUShaderFunc = std::function<void(
    const std::vector<std::shared_ptr<CPUBuffer>>& buffers,
    const DispatchDimensions& dims,
    const void* push_constants)>;

class CPUPipeline : public IComputePipeline {
public:
    CPUPipeline(std::string name, CPUShaderFunc func)
        : name_(std::move(name)), func_(std::move(func)) {}

    const std::string& name() const override { return name_; }
    const CPUShaderFunc& func() const { return func_; }

private:
    std::string name_;
    CPUShaderFunc func_;
};

class CPUBackend : public IComputeBackend {
public:
    CPUBackend();
    ~CPUBackend() override = default;

    std::string backend_name() const override { return "CPU_Reference"; }
    const DeviceInfo& device_info() const override { return info_; }

    std::shared_ptr<IComputeBuffer> create_buffer(
        size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location) override;

    std::shared_ptr<IComputePipeline> create_pipeline_from_spirv(
        const std::string& name,
        const std::vector<uint32_t>& spirv_code,
        size_t buffer_binding_count,
        size_t push_constant_size = 0) override;

    void register_cpu_kernel(const std::string& name, CPUShaderFunc func);

    void dispatch(
        const std::shared_ptr<IComputePipeline>& pipeline,
        const std::vector<std::shared_ptr<IComputeBuffer>>& buffers,
        const DispatchDimensions& dims,
        const void* push_constants = nullptr,
        size_t push_constants_size = 0) override;

    void synchronize() override {}

private:
    DeviceInfo info_;
    std::unordered_map<std::string, CPUShaderFunc> kernel_registry_;
};

} // namespace blocktensor
