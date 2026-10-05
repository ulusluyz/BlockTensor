#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include "backends/cpu/cpu_backend.hpp"
#include "memory/memory_manager.hpp"
#include "blocktensor/transformer_engine.hpp"
#include "blocktensor/kv_cache.hpp"
#include "quantization/quant_types.hpp"

blocktensor::TransformerWeights create_synthetic_weights(
    const blocktensor::ModelConfig& cfg,
    blocktensor::MemoryManager& mem,
    blocktensor::DataType quant_type) {

    blocktensor::TransformerWeights weights;
    weights.config = cfg;
    weights.quant_type = quant_type;

    std::vector<float> embed(cfg.vocab_size * cfg.dim, 0.1f);
    weights.embed_tokens_w = mem.allocate(embed.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    mem.copy_host_to_device(embed.data(), weights.embed_tokens_w, embed.size() * sizeof(float));

    std::vector<float> norm_w(cfg.dim, 1.0f);
    weights.output_norm_w = mem.allocate(norm_w.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    mem.copy_host_to_device(norm_w.data(), weights.output_norm_w, norm_w.size() * sizeof(float));

    std::vector<float> raw_head(cfg.vocab_size * cfg.dim, 0.05f);
    if (quant_type == blocktensor::DataType::INT8) {
        std::vector<blocktensor::quantization::BlockQ8_0> qhead(cfg.vocab_size * (cfg.dim / blocktensor::quantization::Q8_BLOCK_SIZE));
        blocktensor::quantization::quantize_q8_0(raw_head.data(), qhead.data(), raw_head.size());
        weights.output_head_w = mem.allocate(qhead.size() * sizeof(blocktensor::quantization::BlockQ8_0), blocktensor::DataType::INT8, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
        mem.copy_host_to_device(qhead.data(), weights.output_head_w, qhead.size() * sizeof(blocktensor::quantization::BlockQ8_0));
    } else {
        std::vector<blocktensor::quantization::BlockQ4_0> qhead(cfg.vocab_size * (cfg.dim / blocktensor::quantization::Q4_BLOCK_SIZE));
        blocktensor::quantization::quantize_q4_0(raw_head.data(), qhead.data(), raw_head.size());
        weights.output_head_w = mem.allocate(qhead.size() * sizeof(blocktensor::quantization::BlockQ4_0), blocktensor::DataType::INT4, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
        mem.copy_host_to_device(qhead.data(), weights.output_head_w, qhead.size() * sizeof(blocktensor::quantization::BlockQ4_0));
    }

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        blocktensor::LayerWeights layer;

        layer.attn_norm_w = mem.allocate(cfg.dim * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
        mem.copy_host_to_device(norm_w.data(), layer.attn_norm_w, norm_w.size() * sizeof(float));

        layer.ffn_norm_w = mem.allocate(cfg.dim * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
        mem.copy_host_to_device(norm_w.data(), layer.ffn_norm_w, norm_w.size() * sizeof(float));

        auto allocate_proj = [&](uint32_t rows, uint32_t cols) {
            std::vector<float> raw(rows * cols, 0.02f);
            if (quant_type == blocktensor::DataType::INT8) {
                std::vector<blocktensor::quantization::BlockQ8_0> qw(rows * (cols / blocktensor::quantization::Q8_BLOCK_SIZE));
                blocktensor::quantization::quantize_q8_0(raw.data(), qw.data(), raw.size());
                auto buf = mem.allocate(qw.size() * sizeof(blocktensor::quantization::BlockQ8_0), blocktensor::DataType::INT8, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
                mem.copy_host_to_device(qw.data(), buf, qw.size() * sizeof(blocktensor::quantization::BlockQ8_0));
                return buf;
            } else {
                std::vector<blocktensor::quantization::BlockQ4_0> qw(rows * (cols / blocktensor::quantization::Q4_BLOCK_SIZE));
                blocktensor::quantization::quantize_q4_0(raw.data(), qw.data(), raw.size());
                auto buf = mem.allocate(qw.size() * sizeof(blocktensor::quantization::BlockQ4_0), blocktensor::DataType::INT4, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
                mem.copy_host_to_device(qw.data(), buf, qw.size() * sizeof(blocktensor::quantization::BlockQ4_0));
                return buf;
            }
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

void test_transformer_engine(blocktensor::DataType quant_type) {
    std::cout << "[TEST] Testing Transformer Execution Engine ("
              << (quant_type == blocktensor::DataType::INT8 ? "INT8" : "INT4") << ")..." << std::endl;

    auto cpu_backend = std::make_shared<blocktensor::CPUBackend>();
    blocktensor::MemoryManager mem(cpu_backend);

    blocktensor::ModelConfig cfg;
    cfg.vocab_size = 100;
    cfg.dim = 64;
    cfg.hidden_dim = 128;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 16;
    cfg.max_seq_len = 32;

    auto weights = create_synthetic_weights(cfg, mem, quant_type);
    blocktensor::TransformerEngine engine(cpu_backend, weights);
    blocktensor::KVCache kv_cache(mem, cfg.num_layers, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim);

    uint32_t prompt_tokens[3] = {5, 12, 42};
    for (uint32_t pos = 0; pos < 3; ++pos) {
        auto logits = engine.forward_single_token(prompt_tokens[pos], pos, kv_cache);
        assert(logits.size() == cfg.vocab_size);
        std::cout << "  Pos " << pos << " Token " << prompt_tokens[pos]
                  << " -> Logits size " << logits.size() << " Sample Logit[0]: " << logits[0] << std::endl;
    }

    std::cout << "  Transformer Execution Engine PASSED!" << std::endl;
}

int main() {
    std::cout << "=== Running Stages 6 & 7 Transformer Engine Tests ===" << std::endl;
    test_transformer_engine(blocktensor::DataType::INT8);
    test_transformer_engine(blocktensor::DataType::INT4);
    std::cout << "=== Stages 6 & 7 Transformer Engine ALL PASSED ===" << std::endl;
    return 0;
}
