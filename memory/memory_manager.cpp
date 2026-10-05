#include "memory/memory_manager.hpp"
#include <stdexcept>
#include <cstring>

namespace blocktensor {

MemoryManager::MemoryManager(std::shared_ptr<IComputeBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_) {
        throw std::invalid_argument("MemoryManager: Backend pointer cannot be null");
    }
}

std::shared_ptr<IComputeBuffer> MemoryManager::allocate(
    size_t bytes, DataType dt, BufferUsage usage, MemoryLocation location) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto buffer = backend_->create_buffer(bytes, dt, usage, location);
    total_allocated_bytes_ += bytes;
    return buffer;
}

void MemoryManager::copy_host_to_device(
    const void* host_src,
    const std::shared_ptr<IComputeBuffer>& device_dst,
    size_t bytes,
    size_t dst_offset) {

    if (!device_dst) {
        throw std::invalid_argument("MemoryManager::copy_host_to_device null buffer");
    }

    if (device_dst->location() == MemoryLocation::HOST_VISIBLE) {
        device_dst->write(host_src, bytes, dst_offset);
    } else {
        auto staging = backend_->create_buffer(
            bytes, device_dst->data_type(), BufferUsage::STAGING, MemoryLocation::HOST_VISIBLE);
        staging->write(host_src, bytes, 0);

        void* src_ptr = staging->map();
        device_dst->write(src_ptr, bytes, dst_offset);
        staging->unmap();
    }
}

void MemoryManager::copy_device_to_host(
    const std::shared_ptr<IComputeBuffer>& device_src,
    void* host_dst,
    size_t bytes,
    size_t src_offset) {

    if (!device_src) {
        throw std::invalid_argument("MemoryManager::copy_device_to_host null buffer");
    }

    if (device_src->location() == MemoryLocation::HOST_VISIBLE) {
        device_src->read(host_dst, bytes, src_offset);
    } else {
        auto staging = backend_->create_buffer(
            bytes, device_src->data_type(), BufferUsage::STAGING, MemoryLocation::HOST_VISIBLE);
        void* dst_ptr = staging->map();
        device_src->read(dst_ptr, bytes, src_offset);
        staging->unmap();
        staging->read(host_dst, bytes, 0);
    }
}

} // namespace blocktensor
