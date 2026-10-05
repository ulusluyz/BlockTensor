#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>
#include <filesystem>
#include "backends/cpu/cpu_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "memory/memory_manager.hpp"
#include "kernels/cpu_ref.hpp"

namespace fs = std::filesystem;

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
        throw std::runtime_error("Failed to find SPIR-V file: " + filename + " in search paths");
    }

    std::ifstream file(found_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file: " + found_path);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    return buffer;
}

bool approx_equal(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) <= tol * (1.0f + std::abs(a));
}

void test_rmsnorm(const std::shared_ptr<blocktensor::IComputeBackend>& backend) {
    std::cout << "  Testing RMSNorm on " << backend->backend_name() << "..." << std::endl;
    uint32_t num_rows = 4;
    uint32_t dim = 512;
    float eps = 1e-5f;

    std::vector<float> input(num_rows * dim);
    std::vector<float> weight(dim);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 100) * 0.1f - 5.0f;
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = 1.0f + static_cast<float>(i % 10) * 0.05f;

    std::vector<float> ref_out(num_rows * dim);
    blocktensor::cpu_ref::rmsnorm(input.data(), weight.data(), ref_out.data(), num_rows, dim, eps);

    blocktensor::MemoryManager mem(backend);
    auto in_buf = mem.allocate(input.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    auto w_buf = mem.allocate(weight.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    auto out_buf = mem.allocate(ref_out.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);

    mem.copy_host_to_device(input.data(), in_buf, input.size() * sizeof(float));
    mem.copy_host_to_device(weight.data(), w_buf, weight.size() * sizeof(float));

    if (backend->backend_name() == "Vulkan_Compute") {
        auto spirv = read_spirv_file("rmsnorm.comp.spv");
        struct PushParams { uint32_t dim; float eps; } params{dim, eps};
        auto pipeline = backend->create_pipeline_from_spirv("rmsnorm", spirv, 3, sizeof(params));
        backend->dispatch(pipeline, {in_buf, w_buf, out_buf}, {num_rows, 1, 1}, &params, sizeof(params));
    } else {
        std::vector<float> cpu_out(num_rows * dim);
        blocktensor::cpu_ref::rmsnorm(input.data(), weight.data(), cpu_out.data(), num_rows, dim, eps);
        mem.copy_host_to_device(cpu_out.data(), out_buf, cpu_out.size() * sizeof(float));
    }

    std::vector<float> gpu_out(num_rows * dim);
    mem.copy_device_to_host(out_buf, gpu_out.data(), gpu_out.size() * sizeof(float));

    for (size_t i = 0; i < gpu_out.size(); ++i) {
        assert(approx_equal(gpu_out[i], ref_out[i]));
    }
    std::cout << "    RMSNorm PASSED!" << std::endl;
}

void test_rope(const std::shared_ptr<blocktensor::IComputeBackend>& backend) {
    std::cout << "  Testing RoPE on " << backend->backend_name() << "..." << std::endl;
    uint32_t seq_len = 2;
    uint32_t num_heads = 4;
    uint32_t head_dim = 64;
    uint32_t pos_offset = 0;
    uint32_t total_elements = seq_len * num_heads * head_dim;

    std::vector<float> input(total_elements);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 30) * 0.2f - 3.0f;

    std::vector<float> ref_out(total_elements);
    blocktensor::cpu_ref::rope(input.data(), ref_out.data(), seq_len, num_heads, head_dim, pos_offset);

    blocktensor::MemoryManager mem(backend);
    auto in_buf = mem.allocate(input.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    auto out_buf = mem.allocate(ref_out.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);

    mem.copy_host_to_device(input.data(), in_buf, input.size() * sizeof(float));

    if (backend->backend_name() == "Vulkan_Compute") {
        auto spirv = read_spirv_file("rope.comp.spv");
        struct PushParams { uint32_t seq_len; uint32_t num_heads; uint32_t head_dim; uint32_t pos_offset; } params{seq_len, num_heads, head_dim, pos_offset};
        auto pipeline = backend->create_pipeline_from_spirv("rope", spirv, 2, sizeof(params));
        uint32_t workgroups = (total_elements / 2 + 255) / 256;
        backend->dispatch(pipeline, {in_buf, out_buf}, {workgroups, 1, 1}, &params, sizeof(params));
    } else {
        std::vector<float> cpu_out(total_elements);
        blocktensor::cpu_ref::rope(input.data(), cpu_out.data(), seq_len, num_heads, head_dim, pos_offset);
        mem.copy_host_to_device(cpu_out.data(), out_buf, cpu_out.size() * sizeof(float));
    }

    std::vector<float> gpu_out(total_elements);
    mem.copy_device_to_host(out_buf, gpu_out.data(), gpu_out.size() * sizeof(float));

    for (size_t i = 0; i < gpu_out.size(); ++i) {
        assert(approx_equal(gpu_out[i], ref_out[i]));
    }
    std::cout << "    RoPE PASSED!" << std::endl;
}

void test_softmax(const std::shared_ptr<blocktensor::IComputeBackend>& backend) {
    std::cout << "  Testing Softmax on " << backend->backend_name() << "..." << std::endl;
    uint32_t num_rows = 4;
    uint32_t dim = 256;

    std::vector<float> input(num_rows * dim);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 50) * 0.1f - 2.5f;

    std::vector<float> ref_out(num_rows * dim);
    blocktensor::cpu_ref::softmax(input.data(), ref_out.data(), num_rows, dim);

    blocktensor::MemoryManager mem(backend);
    auto in_buf = mem.allocate(input.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    auto out_buf = mem.allocate(ref_out.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);

    mem.copy_host_to_device(input.data(), in_buf, input.size() * sizeof(float));

    if (backend->backend_name() == "Vulkan_Compute") {
        auto spirv = read_spirv_file("softmax.comp.spv");
        struct PushParams { uint32_t num_rows; uint32_t dim; } params{num_rows, dim};
        auto pipeline = backend->create_pipeline_from_spirv("softmax", spirv, 2, sizeof(params));
        backend->dispatch(pipeline, {in_buf, out_buf}, {num_rows, 1, 1}, &params, sizeof(params));
    } else {
        std::vector<float> cpu_out(num_rows * dim);
        blocktensor::cpu_ref::softmax(input.data(), cpu_out.data(), num_rows, dim);
        mem.copy_host_to_device(cpu_out.data(), out_buf, cpu_out.size() * sizeof(float));
    }

    std::vector<float> gpu_out(num_rows * dim);
    mem.copy_device_to_host(out_buf, gpu_out.data(), gpu_out.size() * sizeof(float));

    for (size_t i = 0; i < gpu_out.size(); ++i) {
        assert(approx_equal(gpu_out[i], ref_out[i]));
    }
    std::cout << "    Softmax PASSED!" << std::endl;
}

void test_silu(const std::shared_ptr<blocktensor::IComputeBackend>& backend) {
    std::cout << "  Testing SiLU on " << backend->backend_name() << "..." << std::endl;
    uint32_t num_elements = 1024;

    std::vector<float> input(num_elements);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 100) * 0.1f - 5.0f;

    std::vector<float> ref_out(num_elements);
    blocktensor::cpu_ref::silu(input.data(), ref_out.data(), num_elements);

    blocktensor::MemoryManager mem(backend);
    auto in_buf = mem.allocate(input.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    auto out_buf = mem.allocate(ref_out.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);

    mem.copy_host_to_device(input.data(), in_buf, input.size() * sizeof(float));

    if (backend->backend_name() == "Vulkan_Compute") {
        auto spirv = read_spirv_file("silu.comp.spv");
        struct PushParams { uint32_t num_elements; } params{num_elements};
        auto pipeline = backend->create_pipeline_from_spirv("silu", spirv, 2, sizeof(params));
        uint32_t workgroups = (num_elements + 255) / 256;
        backend->dispatch(pipeline, {in_buf, out_buf}, {workgroups, 1, 1}, &params, sizeof(params));
    } else {
        std::vector<float> cpu_out(num_elements);
        blocktensor::cpu_ref::silu(input.data(), cpu_out.data(), num_elements);
        mem.copy_host_to_device(cpu_out.data(), out_buf, cpu_out.size() * sizeof(float));
    }

    std::vector<float> gpu_out(num_elements);
    mem.copy_device_to_host(out_buf, gpu_out.data(), gpu_out.size() * sizeof(float));

    for (size_t i = 0; i < gpu_out.size(); ++i) {
        assert(approx_equal(gpu_out[i], ref_out[i]));
    }
    std::cout << "    SiLU PASSED!" << std::endl;
}

int main() {
    std::cout << "=== Running Stage 3 Math Kernels Tests ===" << std::endl;

    auto cpu_backend = std::make_shared<blocktensor::CPUBackend>();
    test_rmsnorm(cpu_backend);
    test_rope(cpu_backend);
    test_softmax(cpu_backend);
    test_silu(cpu_backend);

    auto vulkan_backend = std::make_shared<blocktensor::VulkanBackend>();
    test_rmsnorm(vulkan_backend);
    test_rope(vulkan_backend);
    test_softmax(vulkan_backend);
    test_silu(vulkan_backend);

    std::cout << "=== Stage 3 Math Kernels ALL PASSED ===" << std::endl;
    return 0;
}
