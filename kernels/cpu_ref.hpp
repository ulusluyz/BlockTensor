#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace blocktensor::cpu_ref {

inline void rmsnorm(const float* input, const float* weight, float* output, uint32_t num_rows, uint32_t dim, float eps = 1e-5f) {
    for (uint32_t r = 0; r < num_rows; ++r) {
        const float* in_row = input + r * dim;
        float* out_row = output + r * dim;

        float sum_sq = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            sum_sq += in_row[d] * in_row[d];
        }

        float mean_sq = sum_sq / static_cast<float>(dim);
        float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (uint32_t d = 0; d < dim; ++d) {
            out_row[d] = in_row[d] * inv_rms * weight[d];
        }
    }
}

inline void rope(const float* input, float* output, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim, uint32_t pos_offset = 0) {
    uint32_t half_dim = head_dim / 2;

    for (uint32_t s = 0; s < seq_len; ++s) {
        uint32_t pos = s + pos_offset;
        for (uint32_t h = 0; h < num_heads; ++h) {
            const float* in_head = input + (s * num_heads + h) * head_dim;
            float* out_head = output + (s * num_heads + h) * head_dim;

            for (uint32_t i = 0; i < half_dim; ++i) {
                float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(head_dim));
                float val = static_cast<float>(pos) * freq;
                float cos_val = std::cos(val);
                float sin_val = std::sin(val);

                float x1 = in_head[i];
                float x2 = in_head[i + half_dim];

                out_head[i]            = x1 * cos_val - x2 * sin_val;
                out_head[i + half_dim] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

inline void softmax(const float* input, float* output, uint32_t num_rows, uint32_t dim) {
    for (uint32_t r = 0; r < num_rows; ++r) {
        const float* in_row = input + r * dim;
        float* out_row = output + r * dim;

        float max_val = *std::max_element(in_row, in_row + dim);

        float sum_exp = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            out_row[d] = std::exp(in_row[d] - max_val);
            sum_exp += out_row[d];
        }

        for (uint32_t d = 0; d < dim; ++d) {
            out_row[d] /= sum_exp;
        }
    }
}

inline void silu(const float* input, float* output, uint32_t num_elements) {
    for (uint32_t i = 0; i < num_elements; ++i) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

} // namespace blocktensor::cpu_ref
