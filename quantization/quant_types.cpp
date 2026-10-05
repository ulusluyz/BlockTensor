#include "quantization/quant_types.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace blocktensor::quantization {

void quantize_q8_0(const float* src, BlockQ8_0* dst, size_t num_elements) {
    if (num_elements % Q8_BLOCK_SIZE != 0) {
        throw std::invalid_argument("quantize_q8_0: num_elements must be a multiple of block size");
    }

    size_t num_blocks = num_elements / Q8_BLOCK_SIZE;

    for (size_t b = 0; b < num_blocks; ++b) {
        const float* block_src = src + b * Q8_BLOCK_SIZE;
        BlockQ8_0& block_dst = dst[b];

        float amax = 0.0f;
        for (size_t i = 0; i < Q8_BLOCK_SIZE; ++i) {
            amax = std::max(amax, std::abs(block_src[i]));
        }

        float scale = amax / 127.0f;
        block_dst.scale = scale;

        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
        for (size_t i = 0; i < Q8_BLOCK_SIZE; ++i) {
            int q = static_cast<int>(std::round(block_src[i] * inv_scale));
            q = std::clamp(q, -128, 127);
            block_dst.qs[i] = static_cast<int8_t>(q);
        }
    }
}

void dequantize_q8_0(const BlockQ8_0* src, float* dst, size_t num_elements) {
    size_t num_blocks = num_elements / Q8_BLOCK_SIZE;

    for (size_t b = 0; b < num_blocks; ++b) {
        const BlockQ8_0& block_src = src[b];
        float* block_dst = dst + b * Q8_BLOCK_SIZE;

        float scale = block_src.scale;
        for (size_t i = 0; i < Q8_BLOCK_SIZE; ++i) {
            block_dst[i] = static_cast<float>(block_src.qs[i]) * scale;
        }
    }
}

void quantize_q4_0(const float* src, BlockQ4_0* dst, size_t num_elements) {
    if (num_elements % Q4_BLOCK_SIZE != 0) {
        throw std::invalid_argument("quantize_q4_0: num_elements must be a multiple of block size");
    }

    size_t num_blocks = num_elements / Q4_BLOCK_SIZE;

    for (size_t b = 0; b < num_blocks; ++b) {
        const float* block_src = src + b * Q4_BLOCK_SIZE;
        BlockQ4_0& block_dst = dst[b];

        float amax = 0.0f;
        for (size_t i = 0; i < Q4_BLOCK_SIZE; ++i) {
            amax = std::max(amax, std::abs(block_src[i]));
        }

        float scale = amax / 7.0f;
        block_dst.scale = scale;

        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
        for (size_t i = 0; i < Q4_BLOCK_SIZE / 2; ++i) {
            int q0 = static_cast<int>(std::round(block_src[2 * i] * inv_scale)) + 8;
            int q1 = static_cast<int>(std::round(block_src[2 * i + 1] * inv_scale)) + 8;

            q0 = std::clamp(q0, 0, 15);
            q1 = std::clamp(q1, 0, 15);

            block_dst.qs[i] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
}

void dequantize_q4_0(const BlockQ4_0* src, float* dst, size_t num_elements) {
    size_t num_blocks = num_elements / Q4_BLOCK_SIZE;

    for (size_t b = 0; b < num_blocks; ++b) {
        const BlockQ4_0& block_src = src[b];
        float* block_dst = dst + b * Q4_BLOCK_SIZE;

        float scale = block_src.scale;
        for (size_t i = 0; i < Q4_BLOCK_SIZE / 2; ++i) {
            uint8_t packed = block_src.qs[i];
            int q0 = static_cast<int>(packed & 0x0F) - 8;
            int q1 = static_cast<int>((packed >> 4) & 0x0F) - 8;

            block_dst[2 * i]     = static_cast<float>(q0) * scale;
            block_dst[2 * i + 1] = static_cast<float>(q1) * scale;
        }
    }
}

void matvec_q8_0(const BlockQ8_0* W, const float* X, float* Y, uint32_t M, uint32_t N) {
    uint32_t blocks_per_row = N / Q8_BLOCK_SIZE;

    for (uint32_t row = 0; row < M; ++row) {
        float sum = 0.0f;
        const BlockQ8_0* row_W = W + row * blocks_per_row;

        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const BlockQ8_0& block = row_W[b];
            float scale = block.scale;
            const float* x_block = X + b * Q8_BLOCK_SIZE;

            float block_sum = 0.0f;
            for (uint32_t i = 0; i < Q8_BLOCK_SIZE; ++i) {
                block_sum += static_cast<float>(block.qs[i]) * x_block[i];
            }
            sum += block_sum * scale;
        }
        Y[row] = sum;
    }
}

void matvec_q4_0(const BlockQ4_0* W, const float* X, float* Y, uint32_t M, uint32_t N) {
    uint32_t blocks_per_row = N / Q4_BLOCK_SIZE;

    for (uint32_t row = 0; row < M; ++row) {
        float sum = 0.0f;
        const BlockQ4_0* row_W = W + row * blocks_per_row;

        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const BlockQ4_0& block = row_W[b];
            float scale = block.scale;
            const float* x_block = X + b * Q4_BLOCK_SIZE;

            float block_sum = 0.0f;
            for (uint32_t i = 0; i < Q4_BLOCK_SIZE / 2; ++i) {
                uint8_t packed = block.qs[i];
                int q0 = static_cast<int>(packed & 0x0F) - 8;
                int q1 = static_cast<int>((packed >> 4) & 0x0F) - 8;

                block_sum += static_cast<float>(q0) * x_block[2 * i] + static_cast<float>(q1) * x_block[2 * i + 1];
            }
            sum += block_sum * scale;
        }
        Y[row] = sum;
    }
}

} // namespace blocktensor::quantization
