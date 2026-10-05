#include <iostream>
#include <vector>
#include <cassert>
#include "backends/cpu/cpu_backend.hpp"
#include "memory/memory_manager.hpp"
#include "blocktensor/tokenizer.hpp"
#include "models/model_importer.hpp"

void test_tokenizer() {
    std::cout << "[TEST] Testing Tokenizer..." << std::endl;
    std::vector<std::string> vocab = {"<unk>", "<s>", "</s>", "blocktensor", "vulkan", "ai"};
    blocktensor::Tokenizer tokenizer;
    tokenizer.build_vocab(vocab);

    auto tokens = tokenizer.encode("blocktensor vulkan");
    assert(tokens.size() == 2);
    assert(tokens[0] == 3);
    assert(tokens[1] == 4);

    std::string decoded = tokenizer.decode(tokens);
    assert(decoded == "blocktensor vulkan");
    std::cout << "  Tokenizer PASSED!" << std::endl;
}

void test_model_importer_exporter() {
    std::cout << "[TEST] Testing Model Importer & Exporter (.btmodel)..." << std::endl;
    auto cpu_backend = std::make_shared<blocktensor::CPUBackend>();
    blocktensor::MemoryManager mem(cpu_backend);

    blocktensor::ModelConfig cfg;
    cfg.vocab_size = 10;
    cfg.dim = 32;
    cfg.hidden_dim = 64;
    cfg.num_layers = 1;

    blocktensor::TransformerWeights original_weights;
    original_weights.config = cfg;
    original_weights.quant_type = blocktensor::DataType::INT8;

    std::vector<float> embed(cfg.vocab_size * cfg.dim, 0.5f);
    original_weights.embed_tokens_w = mem.allocate(embed.size() * sizeof(float), blocktensor::DataType::FLOAT32, blocktensor::BufferUsage::STORAGE, blocktensor::MemoryLocation::HOST_VISIBLE);
    mem.copy_host_to_device(embed.data(), original_weights.embed_tokens_w, embed.size() * sizeof(float));

    original_weights.output_norm_w = original_weights.embed_tokens_w;
    original_weights.output_head_w = original_weights.embed_tokens_w;

    blocktensor::LayerWeights layer;
    layer.attn_norm_w = original_weights.embed_tokens_w;
    layer.ffn_norm_w  = original_weights.embed_tokens_w;
    layer.q_proj_w    = original_weights.embed_tokens_w;
    layer.k_proj_w    = original_weights.embed_tokens_w;
    layer.v_proj_w    = original_weights.embed_tokens_w;
    layer.o_proj_w    = original_weights.embed_tokens_w;
    layer.gate_proj_w = original_weights.embed_tokens_w;
    layer.up_proj_w   = original_weights.embed_tokens_w;
    layer.down_proj_w = original_weights.embed_tokens_w;
    original_weights.layers.push_back(layer);

    std::string test_file = "test_export.btmodel";
    blocktensor::importer::export_btmodel(test_file, original_weights, mem);

    auto imported_weights = blocktensor::importer::import_btmodel(test_file, cpu_backend, mem);
    assert(imported_weights.config.vocab_size == cfg.vocab_size);
    assert(imported_weights.config.dim == cfg.dim);
    assert(imported_weights.layers.size() == 1);

    std::cout << "  Model Importer & Exporter PASSED!" << std::endl;
}

int main() {
    std::cout << "=== Running Stages 8 & 9 Integration Tests ===" << std::endl;
    test_tokenizer();
    test_model_importer_exporter();
    std::cout << "=== Stages 8 & 9 Integration Tests ALL PASSED ===" << std::endl;
    return 0;
}
