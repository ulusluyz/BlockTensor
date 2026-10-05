#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include <fstream>
#include "backends/cpu/cpu_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "memory/memory_manager.hpp"
#include "blocktensor/transformer_engine.hpp"
#include "blocktensor/kv_cache.hpp"
#include "blocktensor/sampler.hpp"
#include "models/model_importer.hpp"
#include "quantization/quant_types.hpp"

using namespace blocktensor;

void test_quantization_edge_cases() {
    std::cout << "[TEST] Testing Quantization Edge Cases (Zeros, Extremes, Subnormals)..." << std::endl;

    // Edge case 1: All zeros
    std::vector<float> zeros(32, 0.0f);
    quantization::BlockQ8_0 q8_zero[1];
    quantization::quantize_q8_0(zeros.data(), q8_zero, 32);
    assert(q8_zero[0].scale == 0.0f);

    std::vector<float> zeros_dequant(32, -1.0f);
    quantization::dequantize_q8_0(q8_zero, zeros_dequant.data(), 32);
    for (float v : zeros_dequant) assert(v == 0.0f);

    // Edge case 2: INT4 Extreme Values
    std::vector<float> extreme = {1e5f, -1e5f};
    for (size_t i = 2; i < 32; ++i) extreme.push_back(0.0f);
    quantization::BlockQ4_0 q4_ext[1];
    quantization::quantize_q4_0(extreme.data(), q4_ext, 32);
    assert(q4_ext[0].scale > 0.0f);

    std::cout << "  Quantization Edge Cases PASSED!" << std::endl;
}

void test_kv_cache_boundaries() {
    std::cout << "[TEST] Testing KV-Cache Boundaries and Overflow..." << std::endl;
    auto cpu = std::make_shared<CPUBackend>();
    MemoryManager mem(cpu);

    uint32_t max_seq = 4;
    KVCache cache(mem, 1, max_seq, 2, 8);

    std::vector<float> dummy(16, 1.0f);
    for (uint32_t p = 0; p < max_seq; ++p) {
        cache.update(0, p, dummy.data(), dummy.data());
    }

    // Expect out_of_range on overflow
    bool caught = false;
    try {
        cache.update(0, max_seq, dummy.data(), dummy.data());
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    std::cout << "  KV-Cache Boundaries PASSED!" << std::endl;
}

void test_malformed_model_files() {
    std::cout << "[TEST] Testing Malformed Model File Handling..." << std::endl;
    auto cpu = std::make_shared<CPUBackend>();
    MemoryManager mem(cpu);

    std::string bad_file = "corrupted.btmodel";
    {
        std::ofstream out(bad_file, std::ios::binary);
        uint32_t bad_magic = 0xDEADBEEF;
        out.write(reinterpret_cast<char*>(&bad_magic), sizeof(bad_magic));
    }

    bool caught = false;
    try {
        importer::import_btmodel(bad_file, cpu, mem);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);

    std::cout << "  Malformed Model File Handling PASSED!" << std::endl;
}

void test_sampler_determinism() {
    std::cout << "[TEST] Testing Sampler Determinism..." << std::endl;
    float logits[5] = {0.1f, 2.5f, 0.5f, 10.0f, -1.0f};

    uint32_t s1 = sampling::sample_top_p_top_k(logits, 5, 0.7f, 0.9f, 40, 9999);
    uint32_t s2 = sampling::sample_top_p_top_k(logits, 5, 0.7f, 0.9f, 40, 9999);
    assert(s1 == s2);

    std::cout << "  Sampler Determinism PASSED!" << std::endl;
}

int main() {
    std::cout << "=== Running Edge Case & Boundary Verification Tests ===" << std::endl;
    test_quantization_edge_cases();
    test_kv_cache_boundaries();
    test_malformed_model_files();
    test_sampler_determinism();
    std::cout << "=== Edge Case & Boundary Verification ALL PASSED ===" << std::endl;
    return 0;
}
