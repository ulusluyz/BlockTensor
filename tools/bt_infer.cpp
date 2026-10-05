#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include "backends/vulkan/vulkan_backend.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "memory/memory_manager.hpp"
#include "blocktensor/transformer_engine.hpp"
#include "blocktensor/tokenizer.hpp"
#include "blocktensor/sampler.hpp"
#include "models/model_importer.hpp"
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

int main(int argc, char** argv) {
    std::cout << "=========================================================" << std::endl;
    std::cout << "  BlockTensor - Experimental Vulkan AI Inference Runtime  " << std::endl;
    std::cout << "=========================================================" << std::endl;

    std::shared_ptr<blocktensor::IComputeBackend> backend;
    try {
        backend = std::make_shared<blocktensor::VulkanBackend>();
        std::cout << "[INFO] Using Vulkan Compute Backend (" << backend->device_info().name << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[WARN] Vulkan initialization failed: " << e.what() << std::endl;
        std::cout << "[INFO] Falling back to CPU Reference Backend." << std::endl;
        backend = std::make_shared<blocktensor::CPUBackend>();
    }

    blocktensor::MemoryManager mem(backend);

    blocktensor::ModelConfig cfg;
    cfg.vocab_size = 10;
    cfg.dim = 64;
    cfg.hidden_dim = 128;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 16;
    cfg.max_seq_len = 32;

    std::cout << "[INFO] Creating synthetic quantized INT8 model weights..." << std::endl;
    auto weights = create_synthetic_weights(cfg, mem, blocktensor::DataType::INT8);

    std::vector<std::string> vocab = {"<unk>", "<s>", "</s>", "hello", "world", "blocktensor", "vulkan", "amd", "inference", "quantized"};
    blocktensor::Tokenizer tokenizer;
    tokenizer.build_vocab(vocab);

    blocktensor::TransformerEngine engine(backend, weights);
    blocktensor::KVCache kv_cache(mem, cfg.num_layers, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim);

    std::string prompt = "hello blocktensor vulkan";
    std::cout << "[PROMPT] \"" << prompt << "\"" << std::endl;

    auto input_tokens = tokenizer.encode(prompt);
    std::cout << "[TOKENS] Input token IDs: ";
    for (auto id : input_tokens) std::cout << id << " ";
    std::cout << std::endl;

    uint32_t pos = 0;
    for (auto token_id : input_tokens) {
        engine.forward_single_token(token_id, pos, kv_cache);
        pos++;
    }

    uint32_t max_gen_tokens = 5;
    uint32_t current_token = input_tokens.back();
    std::vector<uint32_t> generated_tokens;

    std::cout << "[GENERATION] ";
    for (uint32_t g = 0; g < max_gen_tokens; ++g) {
        auto logits = engine.forward_single_token(current_token, pos, kv_cache);
        uint32_t next_token = blocktensor::sampling::sample_top_p_top_k(
            logits.data(), cfg.vocab_size, 0.7f, 0.9f, 40, 100 + g);

        generated_tokens.push_back(next_token);
        std::cout << tokenizer.decode(next_token) << " ";
        current_token = next_token;
        pos++;
    }
    std::cout << std::endl;
    std::cout << "[COMPLETED] Generated " << generated_tokens.size() << " tokens successfully!" << std::endl;

    return 0;
}
