#pragma once

#include "blocktensor/model_config.hpp"
#include "blocktensor/backend.hpp"
#include "memory/memory_manager.hpp"
#include "quantization/quant_types.hpp"
#include <memory>
#include <vector>

namespace blocktensor {

struct LayerWeights {
    std::shared_ptr<IComputeBuffer> attn_norm_w;
    std::shared_ptr<IComputeBuffer> ffn_norm_w;

    std::shared_ptr<IComputeBuffer> q_proj_w;
    std::shared_ptr<IComputeBuffer> k_proj_w;
    std::shared_ptr<IComputeBuffer> v_proj_w;
    std::shared_ptr<IComputeBuffer> o_proj_w;

    std::shared_ptr<IComputeBuffer> gate_proj_w;
    std::shared_ptr<IComputeBuffer> up_proj_w;
    std::shared_ptr<IComputeBuffer> down_proj_w;
};

struct TransformerWeights {
    ModelConfig config;
    DataType quant_type{DataType::INT8};

    std::shared_ptr<IComputeBuffer> embed_tokens_w;
    std::shared_ptr<IComputeBuffer> output_norm_w;
    std::shared_ptr<IComputeBuffer> output_head_w;

    std::vector<LayerWeights> layers;
};

} // namespace blocktensor
