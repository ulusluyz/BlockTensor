#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "kernels/cpu_ref.hpp"
#include "quantization/quant_types.hpp"

using namespace blocktensor;

int main() {
    std::cout << "=== Numerical Precision & Error Analysis ===" << std::endl;

    // 1. RMSNorm Error
    std::vector<float> in(512), w(512, 1.0f), out1(512), out2(512);
    for (size_t i = 0; i < 512; ++i) in[i] = static_cast<float>(i % 100) * 0.1f - 5.0f;
    cpu_ref::rmsnorm(in.data(), w.data(), out1.data(), 1, 512, 1e-5f);
    cpu_ref::rmsnorm(in.data(), w.data(), out2.data(), 1, 512, 1e-5f);

    float max_err_rms = 0.0f, sum_err_rms = 0.0f;
    for (size_t i = 0; i < 512; ++i) {
        float err = std::abs(out1[i] - out2[i]);
        if (err > max_err_rms) max_err_rms = err;
        sum_err_rms += err;
    }

    // 2. Q8_0 MatVec Error vs Float32
    std::vector<float> raw_w(16 * 256), x(256), float_y(16, 0.0f), q8_y(16, 0.0f);
    for (size_t i = 0; i < raw_w.size(); ++i) raw_w[i] = static_cast<float>(i % 50) * 0.1f - 2.5f;
    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i % 20) * 0.15f - 1.5f;

    for (size_t r = 0; r < 16; ++r) {
        for (size_t c = 0; c < 256; ++c) float_y[r] += raw_w[r * 256 + c] * x[c];
    }

    std::vector<quantization::BlockQ8_0> q8_blocks(16 * (256 / 32));
    quantization::quantize_q8_0(raw_w.data(), q8_blocks.data(), raw_w.size());
    quantization::matvec_q8_0(q8_blocks.data(), x.data(), q8_y.data(), 16, 256);

    float max_err_q8 = 0.0f, sum_err_q8 = 0.0f;
    for (size_t i = 0; i < 16; ++i) {
        float err = std::abs(float_y[i] - q8_y[i]);
        if (err > max_err_q8) max_err_q8 = err;
        sum_err_q8 += err;
    }

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "RMSNorm Max Error:  " << max_err_rms << " | Mean Error: " << (sum_err_rms / 512.0f) << std::endl;
    std::cout << "Q8_0    Max Error:  " << max_err_q8  << " | Mean Error: " << (sum_err_q8 / 16.0f) << std::endl;

    return 0;
}
