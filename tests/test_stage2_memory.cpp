#include <iostream>
#include <cassert>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "memory/memory_manager.hpp"

void test_memory_manager_with_backend(const std::shared_ptr<blocktensor::IComputeBackend>& backend) {
    std::cout << "[TEST] Testing MemoryManager with backend: " << backend->backend_name() << std::endl;

    blocktensor::MemoryManager mem(backend);

    size_t alloc_size = 4096;
    auto buf = mem.allocate(alloc_size, blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    assert(buf != nullptr);
    assert(buf->size() == alloc_size);
    assert(mem.total_allocated_bytes() == alloc_size);

    std::vector<float> input(1024);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i * 2.5f);
    }

    mem.copy_host_to_device(input.data(), buf, input.size() * sizeof(float));

    std::vector<float> output(1024, 0.0f);
    mem.copy_device_to_host(buf, output.data(), output.size() * sizeof(float));

    for (size_t i = 0; i < input.size(); ++i) {
        assert(input[i] == output[i]);
    }

    std::cout << "  MemoryManager PASSED for " << backend->backend_name() << std::endl;
}

int main() {
    std::cout << "=== Running Stage 2 Memory Management Tests ===" << std::endl;

    auto cpu_backend = std::make_shared<blocktensor::CPUBackend>();
    test_memory_manager_with_backend(cpu_backend);

    auto vulkan_backend = std::make_shared<blocktensor::VulkanBackend>();
    test_memory_manager_with_backend(vulkan_backend);

    std::cout << "=== Stage 2 Memory Management Tests ALL PASSED ===" << std::endl;
    return 0;
}
