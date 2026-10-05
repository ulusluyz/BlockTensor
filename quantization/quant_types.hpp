#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

namespace blocktensor::quantization {

constexpr uint32_t Q8_BLOCK_SIZE = 32;
constexpr uint32_t Q4_BLOCK_SIZE = 32;

struct alignas(4) BlockQ8_0 {
    float scale;
    int8_t qs[Q8_BLOCK_SIZE];
};

struct alignas(4) BlockQ4_0 {
    float scale;
    uint8_t qs[Q4_BLOCK_SIZE / 2];
};

void quantize_q8_0(const float* src, BlockQ8_0* dst, size_t num_elements);
void dequantize_q8_0(const BlockQ8_0* src, float* dst, size_t num_elements);

void quantize_q4_0(const float* src, BlockQ4_0* dst, size_t num_elements);
void dequantize_q4_0(const BlockQ4_0* src, float* dst, size_t num_elements);

void matvec_q8_0(const BlockQ8_0* W, const float* X, float* Y, uint32_t M, uint32_t N);
void matvec_q4_0(const BlockQ4_0* W, const float* X, float* Y, uint32_t M, uint32_t N);

} // namespace blocktensor::quantization
