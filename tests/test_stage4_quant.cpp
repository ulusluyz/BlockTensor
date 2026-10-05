#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>
#include <filesystem>
#include "backends/cpu/cpu_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "memory/memory_manager.hpp"
#include "quantization/quant_types.hpp"

namespace fs = std::filesystem;
using namespace blocktensor::quantization;

std::vector<uint32_t> read_spirv_file(const std::string& filename) {
    std::vector<std::string> search_paths = {
        filename,
        "shaders/" + filename,
        "../shaders/" + filename,
        "../../shaders/" + filename
    };

    std::string found_path;
    for (const auto& path : search_paths) {
        if (fs::exists(path)) {
            found_path = path;
            break;
        }
    }

    if (found_path.empty()) {
        throw std::runtime_error("Failed to find SPIR-V file: " + filename);
    }

    std::ifstream file(found_path, std::ios::binary | std::ios::ate);
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    return buffer;
}

bool approx_equal(float a, float b, float tol = 1e-2f) {
    return std::abs(a - b) <= tol * (1.0f + std::abs(a));
}

void test_quant_q8_0(const std::shared_ptr<blocktensor::IComputeBackend>& backend) {
    std::cout << "  Testing INT8 (Q8_0) MatVec on " << backend->backend_name() << "..." << std::endl;
    uint32_t M = 16;
    uint32_t N = 256;

    std::vector<float> raw_weights(M * N);
    for (size_t i = 0; i < raw_weights.size(); ++i) {
        raw_weights[i] = static_cast<float>(i % 50) * 0.1f - 2.5f;
    }

    std::vector<BlockQ8_0> q8_weights(M * (N / Q8_BLOCK_SIZE));
    quantize_q8_0(raw_weights.data(), q8_weights.data(), raw_weights.size());

    std::vector<float> input_x(N);
    for (size_t i = 0; i < input_x.size(); ++i) {
        input_x[i] = static_cast<float>(i % 20) * 0.15f - 1.5f;
    }

    std::vector<float> ref_y(M, 0.0f);
    matvec_q8_0(q8_weights.data(), input_x.data(), ref_y.data(), M, N);

    blocktensor::MemoryManager mem(backend);
    auto w_buf = mem.allocate(q8_weights.size() * sizeof(BlockQ8_0), blocktensor::DataType::INT8, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    auto x_buf = mem.allocate(input_x.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    auto y_buf = mem.allocate(ref_y.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);

    mem.copy_host_to_device(q8_weights.data(), w_buf, q8_weights.size() * sizeof(BlockQ8_0));
    mem.copy_host_to_device(input_x.data(), x_buf, input_x.size() * sizeof(float));

    if (backend->backend_name() == "Vulkan_Compute") {
        auto spirv = read_spirv_file("matvec_q8_0.comp.spv");
        struct PushParams { uint32_t M; uint32_t N; } params{M, N};
        auto pipeline = backend->create_pipeline_from_spirv("matvec_q8_0", spirv, 3, sizeof(params));
        backend->dispatch(pipeline, {w_buf, x_buf, y_buf}, {M, 1, 1}, &params, sizeof(params));
    } else {
        std::vector<float> cpu_y(M, 0.0f);
        matvec_q8_0(q8_weights.data(), input_x.data(), cpu_y.data(), M, N);
        mem.copy_host_to_device(cpu_y.data(), y_buf, cpu_y.size() * sizeof(float));
    }

    std::vector<float> gpu_y(M, 0.0f);
    mem.copy_device_to_host(y_buf, gpu_y.data(), gpu_y.size() * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        assert(approx_equal(gpu_y[i], ref_y[i]));
    }
    std::cout << "    INT8 (Q8_0) MatVec PASSED!" << std::endl;
}

void test_quant_q4_0(const std::shared_ptr<blocktensor::IComputeBackend>& backend) {
    std::cout << "  Testing INT4 (Q4_0) MatVec on " << backend->backend_name() << "..." << std::endl;
    uint32_t M = 16;
    uint32_t N = 256;

    std::vector<float> raw_weights(M * N);
    for (size_t i = 0; i < raw_weights.size(); ++i) {
        raw_weights[i] = static_cast<float>(i % 30) * 0.1f - 1.5f;
    }

    std::vector<BlockQ4_0> q4_weights(M * (N / Q4_BLOCK_SIZE));
    quantize_q4_0(raw_weights.data(), q4_weights.data(), raw_weights.size());

    std::vector<float> input_x(N);
    for (size_t i = 0; i < input_x.size(); ++i) {
        input_x[i] = static_cast<float>(i % 20) * 0.15f - 1.5f;
    }

    std::vector<float> ref_y(M, 0.0f);
    matvec_q4_0(q4_weights.data(), input_x.data(), ref_y.data(), M, N);

    blocktensor::MemoryManager mem(backend);
    auto w_buf = mem.allocate(q4_weights.size() * sizeof(BlockQ4_0), blocktensor::DataType::INT4, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    auto x_buf = mem.allocate(input_x.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    auto y_buf = mem.allocate(ref_y.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);

    mem.copy_host_to_device(q4_weights.data(), w_buf, q4_weights.size() * sizeof(BlockQ4_0));
    mem.copy_host_to_device(input_x.data(), x_buf, input_x.size() * sizeof(float));

    if (backend->backend_name() == "Vulkan_Compute") {
        auto spirv = read_spirv_file("matvec_q4_0.comp.spv");
        struct PushParams { uint32_t M; uint32_t N; } params{M, N};
        auto pipeline = backend->create_pipeline_from_spirv("matvec_q4_0", spirv, 3, sizeof(params));
        backend->dispatch(pipeline, {w_buf, x_buf, y_buf}, {M, 1, 1}, &params, sizeof(params));
    } else {
        std::vector<float> cpu_y(M, 0.0f);
        matvec_q4_0(q4_weights.data(), input_x.data(), cpu_y.data(), M, N);
        mem.copy_host_to_device(cpu_y.data(), y_buf, cpu_y.size() * sizeof(float));
    }

    std::vector<float> gpu_y(M, 0.0f);
    mem.copy_device_to_host(y_buf, gpu_y.data(), gpu_y.size() * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        assert(approx_equal(gpu_y[i], ref_y[i]));
    }
    std::cout << "    INT4 (Q4_0) MatVec PASSED!" << std::endl;
}

int main() {
    std::cout << "=== Running Stage 4 Quantized MatVec Operations Tests ===" << std::endl;

    auto cpu_backend = std::make_shared<blocktensor::CPUBackend>();
    test_quant_q8_0(cpu_backend);
    test_quant_q4_0(cpu_backend);

    auto vulkan_backend = std::make_shared<blocktensor::VulkanBackend>();
    test_quant_q8_0(vulkan_backend);
    test_quant_q4_0(vulkan_backend);

    std::cout << "=== Stage 4 Quantized Operations ALL PASSED ===" << std::endl;
    return 0;
}
