#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

namespace blocktensor {

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT8,
    INT4
};

size_t get_data_type_size(DataType dt);

enum class BufferUsage {
    STORAGE,
    STAGING,
    UNIFORM
};

enum class MemoryLocation {
    HOST_VISIBLE,
    DEVICE_LOCAL
};

class IComputeBuffer {
public:
    virtual ~IComputeBuffer() = default;
    virtual size_t size() const = 0;
    virtual DataType data_type() const = 0;
    virtual BufferUsage usage() const = 0;
    virtual MemoryLocation location() const = 0;
    virtual void* map() = 0;
    virtual void unmap() = 0;
    virtual void write(const void* data, size_t bytes, size_t offset = 0) = 0;
    virtual void read(void* data, size_t bytes, size_t offset = 0) = 0;
};

class IComputePipeline {
public:
    virtual ~IComputePipeline() = default;
    virtual const std::string& name() const = 0;
};

struct DispatchDimensions {
    uint32_t x{1};
    uint32_t y{1};
    uint32_t z{1};
};

struct DeviceInfo {
    std::string name;
    std::string vendor_name;
    uint32_t vendor_id{0};
    uint32_t device_id{0};
    bool is_discrete{false};
    uint64_t total_vram_bytes{0};
    uint32_t max_workgroup_size[3]{1024, 1024, 64};
    uint32_t max_workgroup_invocations{1024};
    uint32_t max_shared_memory_bytes{32768};
};

class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;
    virtual std::string backend_name() const = 0;
    virtual const DeviceInfo& device_info() const = 0;

    virtual std::shared_ptr<IComputeBuffer> create_buffer(
        size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location) = 0;

    virtual std::shared_ptr<IComputePipeline> create_pipeline_from_spirv(
        const std::string& name,
        const std::vector<uint32_t>& spirv_code,
        size_t buffer_binding_count,
        size_t push_constant_size = 0) = 0;

    virtual void dispatch(
        const std::shared_ptr<IComputePipeline>& pipeline,
        const std::vector<std::shared_ptr<IComputeBuffer>>& buffers,
        const DispatchDimensions& dims,
        const void* push_constants = nullptr,
        size_t push_constants_size = 0) = 0;

    virtual void synchronize() = 0;
};

} // namespace blocktensor
