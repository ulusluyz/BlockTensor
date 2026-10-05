#pragma once

#include "blocktensor/backend.hpp"
#include "memory/memory_manager.hpp"
#include <memory>
#include <vector>

namespace blocktensor {

class KVCache {
public:
    KVCache(MemoryManager& memory_mgr, uint32_t num_layers, uint32_t max_seq_len, uint32_t kv_heads, uint32_t head_dim);

    uint32_t current_seq_len() const { return current_seq_len_; }
    uint32_t max_seq_len() const { return max_seq_len_; }

    std::shared_ptr<IComputeBuffer> k_cache(uint32_t layer) const { return k_caches_[layer]; }
    std::shared_ptr<IComputeBuffer> v_cache(uint32_t layer) const { return v_caches_[layer]; }

    void update(uint32_t layer, uint32_t pos, const float* k_data, const float* v_data);
    void advance_pos() { current_seq_len_++; }

private:
    uint32_t num_layers_;
    uint32_t max_seq_len_;
    uint32_t kv_heads_;
    uint32_t head_dim_;
    uint32_t current_seq_len_{0};

    std::vector<std::shared_ptr<IComputeBuffer>> k_caches_;
    std::vector<std::shared_ptr<IComputeBuffer>> v_caches_;
    MemoryManager& memory_mgr_;
};

} // namespace blocktensor
