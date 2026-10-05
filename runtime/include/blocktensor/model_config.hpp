#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace blocktensor {

struct ModelConfig {
    uint32_t vocab_size{32000};
    uint32_t dim{512};
    uint32_t hidden_dim{1376};
    uint32_t num_layers{4};
    uint32_t num_heads{8};
    uint32_t num_kv_heads{8};
    uint32_t head_dim{64};
    uint32_t max_seq_len{512};
    float rms_norm_eps{1e-5f};
};

} // namespace blocktensor
