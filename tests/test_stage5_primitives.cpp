#include <iostream>
#include <cassert>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "memory/memory_manager.hpp"
#include "blocktensor/sampler.hpp"
#include "blocktensor/kv_cache.hpp"

void test_sampling() {
    std::cout << "  Testing Sampling Primitives..." << std::endl;
    uint32_t vocab_size = 5;
    float logits[5] = {1.0f, 2.5f, 0.1f, 10.0f, -3.0f};

    uint32_t greedy_id = blocktensor::sampling::sample_greedy(logits, vocab_size);
    assert(greedy_id == 3);

    uint32_t sample_id = blocktensor::sampling::sample_top_p_top_k(logits, vocab_size, 0.7f, 0.9f, 40, 12345);
    assert(sample_id == 3);

    std::cout << "    Sampling Primitives PASSED!" << std::endl;
}

void test_kv_cache() {
    std::cout << "  Testing KV-Cache Management..." << std::endl;
    auto cpu_backend = std::make_shared<blocktensor::CPUBackend>();
    blocktensor::MemoryManager mem(cpu_backend);

    uint32_t num_layers = 2;
    uint32_t max_seq_len = 16;
    uint32_t kv_heads = 2;
    uint32_t head_dim = 8;

    blocktensor::KVCache cache(mem, num_layers, max_seq_len, kv_heads, head_dim);

    std::vector<float> k_in = {1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16};
    std::vector<float> v_in = {16,15,14,13,12,11,10,9, 8,7,6,5,4,3,2,1};

    cache.update(0, 0, k_in.data(), v_in.data());

    std::vector<float> k_out(16, 0.0f);
    mem.copy_device_to_host(cache.k_cache(0), k_out.data(), 16 * sizeof(float), 0);

    for (size_t i = 0; i < 16; ++i) {
        assert(k_out[i] == k_in[i]);
    }

    std::cout << "    KV-Cache PASSED!" << std::endl;
}

int main() {
    std::cout << "=== Running Stage 5 Primitives Tests ===" << std::endl;
    test_sampling();
    test_kv_cache();
    std::cout << "=== Stage 5 Primitives ALL PASSED ===" << std::endl;
    return 0;
}
