#pragma once

#include "blocktensor/backend.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace blocktensor {

class MemoryManager {
public:
    explicit MemoryManager(std::shared_ptr<IComputeBackend> backend);
    ~MemoryManager() = default;

    std::shared_ptr<IComputeBuffer> allocate(
        size_t bytes,
        DataType dt,
        BufferUsage usage = BufferUsage::STORAGE,
        MemoryLocation location = MemoryLocation::DEVICE_LOCAL);

    void copy_host_to_device(
        const void* host_src,
        const std::shared_ptr<IComputeBuffer>& device_dst,
        size_t bytes,
        size_t dst_offset = 0);

    void copy_device_to_host(
        const std::shared_ptr<IComputeBuffer>& device_src,
        void* host_dst,
        size_t bytes,
        size_t src_offset = 0);

    size_t total_allocated_bytes() const { return total_allocated_bytes_; }

private:
    std::shared_ptr<IComputeBackend> backend_;
    size_t total_allocated_bytes_{0};
    std::mutex mutex_;
};

} // namespace blocktensor
