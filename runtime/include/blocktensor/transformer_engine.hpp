#pragma once

#include "blocktensor/transformer_weights.hpp"
#include "blocktensor/kv_cache.hpp"
#include "blocktensor/backend.hpp"
#include "memory/memory_manager.hpp"
#include <memory>
#include <vector>

namespace blocktensor {

class TransformerEngine {
public:
    TransformerEngine(std::shared_ptr<IComputeBackend> backend, const TransformerWeights& weights);
    ~TransformerEngine() = default;

    std::vector<float> forward_single_token(uint32_t token_id, uint32_t pos, KVCache& kv_cache);

private:
    void execute_layer(uint32_t layer_idx, uint32_t pos, KVCache& kv_cache);

    void dispatch_matvec(
        const std::shared_ptr<IComputeBuffer>& W,
        const std::shared_ptr<IComputeBuffer>& X,
        const std::shared_ptr<IComputeBuffer>& Y,
        uint32_t M, uint32_t N);

    void dispatch_rmsnorm(
        const std::shared_ptr<IComputeBuffer>& X,
        const std::shared_ptr<IComputeBuffer>& W,
        const std::shared_ptr<IComputeBuffer>& Out,
        uint32_t num_rows, uint32_t dim, float eps);

    std::shared_ptr<IComputeBackend> backend_;
    MemoryManager mem_;
    TransformerWeights weights_;
    ModelConfig cfg_;

    std::shared_ptr<IComputeBuffer> x_buf_;
    std::shared_ptr<IComputeBuffer> norm_x_buf_;
    std::shared_ptr<IComputeBuffer> q_buf_;
    std::shared_ptr<IComputeBuffer> k_buf_;
    std::shared_ptr<IComputeBuffer> v_buf_;
    std::shared_ptr<IComputeBuffer> attn_out_buf_;
    std::shared_ptr<IComputeBuffer> gate_buf_;
    std::shared_ptr<IComputeBuffer> up_buf_;
    std::shared_ptr<IComputeBuffer> mlp_out_buf_;
    std::shared_ptr<IComputeBuffer> logits_buf_;

    // Vulkan compute pipelines
    std::shared_ptr<IComputePipeline> rmsnorm_pipeline_;
    std::shared_ptr<IComputePipeline> rope_pipeline_;
    std::shared_ptr<IComputePipeline> softmax_pipeline_;
    std::shared_ptr<IComputePipeline> silu_pipeline_;
    std::shared_ptr<IComputePipeline> matvec_pipeline_;
};

} // namespace blocktensor
