#include <iostream>
#include <cassert>
#include "backends/cpu/cpu_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"

void test_cpu_backend() {
    std::cout << "[TEST] Testing CPU Backend Initialization..." << std::endl;
    blocktensor::CPUBackend cpu;
    const auto& info = cpu.device_info();
    std::cout << "  Device Name: " << info.name << std::endl;
    assert(cpu.backend_name() == "CPU_Reference");

    auto buf = cpu.create_buffer(1024, blocktensor::DataType::FLOAT32,
                                blocktensor::BufferUsage::STORAGE,
                                blocktensor::MemoryLocation::HOST_VISIBLE);
    assert(buf != nullptr);
    assert(buf->size() == 1024);

    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dst[4] = {0};
    buf->write(src, sizeof(src));
    buf->read(dst, sizeof(dst));

    for (int i = 0; i < 4; ++i) {
        assert(src[i] == dst[i]);
    }
    std::cout << "  CPU Backend PASSED!" << std::endl;
}

void test_vulkan_backend() {
    std::cout << "[TEST] Testing Vulkan Backend Initialization..." << std::endl;
    try {
        blocktensor::VulkanBackend vk;
        const auto& info = vk.device_info();
        std::cout << "  Vulkan Device Name: " << info.name << " (" << info.vendor_name << ")" << std::endl;
        assert(vk.backend_name() == "Vulkan_Compute");

        auto buf = vk.create_buffer(1024, blocktensor::DataType::FLOAT32,
                                    blocktensor::BufferUsage::STORAGE,
                                    blocktensor::MemoryLocation::HOST_VISIBLE);
        assert(buf != nullptr);
        assert(buf->size() == 1024);

        float src[4] = {10.0f, 20.0f, 30.0f, 40.0f};
        float dst[4] = {0};
        buf->write(src, sizeof(src));
        buf->read(dst, sizeof(dst));

        for (int i = 0; i < 4; ++i) {
            assert(src[i] == dst[i]);
        }
        std::cout << "  Vulkan Backend PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  Vulkan Backend test error: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    std::cout << "=== Running Stage 1 Backend Tests ===" << std::endl;
    test_cpu_backend();
    test_vulkan_backend();
    std::cout << "=== Stage 1 Backend Tests ALL PASSED ===" << std::endl;
    return 0;
}
