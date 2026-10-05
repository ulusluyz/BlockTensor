#include "blocktensor/kv_cache.hpp"
#include <stdexcept>

namespace blocktensor {

KVCache::KVCache(MemoryManager& memory_mgr, uint32_t num_layers, uint32_t max_seq_len, uint32_t kv_heads, uint32_t head_dim)
    : num_layers_(num_layers), max_seq_len_(max_seq_len), kv_heads_(kv_heads), head_dim_(head_dim), memory_mgr_(memory_mgr) {

    size_t bytes_per_layer = static_cast<size_t>(max_seq_len) * kv_heads * head_dim * sizeof(float);

    for (uint32_t l = 0; l < num_layers; ++l) {
        k_caches_.push_back(memory_mgr_.allocate(bytes_per_layer, DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE));
        v_caches_.push_back(memory_mgr_.allocate(bytes_per_layer, DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE));
    }
}

void KVCache::update(uint32_t layer, uint32_t pos, const float* k_data, const float* v_data) {
    if (layer >= num_layers_) throw std::out_of_range("KVCache layer out of bounds");
    if (pos >= max_seq_len_) throw std::out_of_range("KVCache pos out of bounds");

    size_t element_offset = static_cast<size_t>(pos) * kv_heads_ * head_dim_;
    size_t byte_offset = element_offset * sizeof(float);
    size_t bytes_to_write = static_cast<size_t>(kv_heads_) * head_dim_ * sizeof(float);

    memory_mgr_.copy_host_to_device(k_data, k_caches_[layer], bytes_to_write, byte_offset);
    memory_mgr_.copy_host_to_device(v_data, v_caches_[layer], bytes_to_write, byte_offset);
}

} // namespace blocktensor
