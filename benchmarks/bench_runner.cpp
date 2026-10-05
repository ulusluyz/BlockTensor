#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include "backends/vulkan/vulkan_backend.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "memory/memory_manager.hpp"
#include "blocktensor/transformer_engine.hpp"
#include "blocktensor/kv_cache.hpp"

blocktensor::TransformerWeights create_bench_weights(
    const blocktensor::ModelConfig& cfg,
    blocktensor::MemoryManager& mem) {

    blocktensor::TransformerWeights weights;
    weights.config = cfg;
    weights.quant_type = blocktensor::DataType::INT8;

    std::vector<float> embed(cfg.vocab_size * cfg.dim, 0.1f);
    weights.embed_tokens_w = mem.allocate(embed.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    mem.copy_host_to_device(embed.data(), weights.embed_tokens_w, embed.size() * sizeof(float));

    std::vector<float> norm_w(cfg.dim, 1.0f);
    weights.output_norm_w = mem.allocate(norm_w.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    mem.copy_host_to_device(norm_w.data(), weights.output_norm_w, norm_w.size() * sizeof(float));

    std::vector<float> raw_head(cfg.vocab_size * cfg.dim, 0.05f);
    std::vector<blocktensor::quantization::BlockQ8_0> qhead(cfg.vocab_size * (cfg.dim / blocktensor::quantization::Q8_BLOCK_SIZE));
    blocktensor::quantization::quantize_q8_0(raw_head.data(), qhead.data(), raw_head.size());
    weights.output_head_w = mem.allocate(qhead.size() * sizeof(blocktensor::quantization::BlockQ8_0), blocktensor::DataType::INT8, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    mem.copy_host_to_device(qhead.data(), weights.output_head_w, qhead.size() * sizeof(blocktensor::quantization::BlockQ8_0));

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        blocktensor::LayerWeights layer;

        layer.attn_norm_w = mem.allocate(cfg.dim * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
        mem.copy_host_to_device(norm_w.data(), layer.attn_norm_w, norm_w.size() * sizeof(float));

        layer.ffn_norm_w = mem.allocate(cfg.dim * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
        mem.copy_host_to_device(norm_w.data(), layer.ffn_norm_w, norm_w.size() * sizeof(float));

        auto allocate_proj = [&](uint32_t rows, uint32_t cols) {
            std::vector<float> raw(rows * cols, 0.02f);
            std::vector<blocktensor::quantization::BlockQ8_0> qw(rows * (cols / blocktensor::quantization::Q8_BLOCK_SIZE));
            blocktensor::quantization::quantize_q8_0(raw.data(), qw.data(), raw.size());
            auto buf = mem.allocate(qw.size() * sizeof(blocktensor::quantization::BlockQ8_0), blocktensor::DataType::INT8, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
            mem.copy_host_to_device(qw.data(), buf, qw.size() * sizeof(blocktensor::quantization::BlockQ8_0));
            return buf;
        };

        layer.q_proj_w = allocate_proj(cfg.dim, cfg.dim);
        layer.k_proj_w = allocate_proj(cfg.num_kv_heads * cfg.head_dim, cfg.dim);
        layer.v_proj_w = allocate_proj(cfg.num_kv_heads * cfg.head_dim, cfg.dim);
        layer.o_proj_w = allocate_proj(cfg.dim, cfg.dim);

        layer.gate_proj_w = allocate_proj(cfg.hidden_dim, cfg.dim);
        layer.up_proj_w   = allocate_proj(cfg.hidden_dim, cfg.dim);
        layer.down_proj_w = allocate_proj(cfg.dim, cfg.hidden_dim);

        weights.layers.push_back(layer);
    }

    return weights;
}

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "  BlockTensor Benchmark Utility                         " << std::endl;
    std::cout << "=========================================================" << std::endl;

    std::shared_ptr<blocktensor::IComputeBackend> backend;
    try {
        backend = std::make_shared<blocktensor::VulkanBackend>();
    } catch (...) {
        backend = std::make_shared<blocktensor::CPUBackend>();
    }

    std::cout << "Backend: " << backend->backend_name() << std::endl;
    std::cout << "Device:  " << backend->device_info().name << std::endl;

    auto t_load_start = std::chrono::high_resolution_clock::now();
    blocktensor::MemoryManager mem(backend);

    blocktensor::ModelConfig cfg;
    cfg.vocab_size = 1000;
    cfg.dim = 256;
    cfg.hidden_dim = 512;
    cfg.num_layers = 4;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 64;
    cfg.max_seq_len = 128;

    auto weights = create_bench_weights(cfg, mem);
    blocktensor::TransformerEngine engine(backend, weights);
    blocktensor::KVCache kv_cache(mem, cfg.num_layers, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim);

    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    std::cout << "\n[BENCHMARK RESULTS - MEASURED]" << std::endl;
    std::cout << "  Model Loading Time:       " << load_ms << " ms" << std::endl;
    std::cout << "  Allocated Memory:         " << (mem.total_allocated_bytes() / 1024.0 / 1024.0) << " MB" << std::endl;

    uint32_t prompt_tokens = 16;
    auto t_prompt_start = std::chrono::high_resolution_clock::now();
    for (uint32_t pos = 0; pos < prompt_tokens; ++pos) {
        engine.forward_single_token(pos % cfg.vocab_size, pos, kv_cache);
    }
    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    double prompt_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();

    std::cout << "  Prompt Processing (16 tok): " << prompt_ms << " ms (" << (prompt_tokens / (prompt_ms / 1000.0)) << " tok/s)" << std::endl;

    uint32_t gen_tokens = 32;
    auto t_gen_start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < gen_tokens; ++i) {
        engine.forward_single_token(i % cfg.vocab_size, prompt_tokens + i, kv_cache);
    }
    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();

    std::cout << "  Token Generation (32 tok):  " << gen_ms << " ms (" << (gen_tokens / (gen_ms / 1000.0)) << " tok/s)" << std::endl;
    std::cout << "  Time-To-First-Token (TTFT): " << (prompt_ms / prompt_tokens) << " ms/tok" << std::endl;

    return 0;
}
