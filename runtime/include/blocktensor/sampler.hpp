#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

namespace blocktensor::sampling {

inline uint32_t sample_greedy(const float* logits, uint32_t vocab_size) {
    return static_cast<uint32_t>(std::distance(logits, std::max_element(logits, logits + vocab_size)));
}

inline uint32_t sample_top_p_top_k(
    const float* logits,
    uint32_t vocab_size,
    float temperature = 0.7f,
    float top_p = 0.9f,
    uint32_t top_k = 40,
    uint64_t seed = 42) {

    if (temperature <= 0.0f) {
        return sample_greedy(logits, vocab_size);
    }

    struct TokenLogit {
        uint32_t id;
        float logit;
    };

    std::vector<TokenLogit> candidate_tokens(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        candidate_tokens[i] = {i, logits[i] / temperature};
    }

    std::sort(candidate_tokens.begin(), candidate_tokens.end(), [](const TokenLogit& a, const TokenLogit& b) {
        return a.logit > b.logit;
    });

    if (top_k > 0 && top_k < candidate_tokens.size()) {
        candidate_tokens.resize(top_k);
    }

    float max_l = candidate_tokens[0].logit;
    float sum_exp = 0.0f;
    std::vector<float> probs(candidate_tokens.size());
    for (size_t i = 0; i < candidate_tokens.size(); ++i) {
        probs[i] = std::exp(candidate_tokens[i].logit - max_l);
        sum_exp += probs[i];
    }
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum_exp;
    }

    float cumsum = 0.0f;
    size_t cutoff = probs.size();
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    probs.resize(cutoff);
    candidate_tokens.resize(cutoff);

    float p_sum = 0.0f;
    for (float p : probs) p_sum += p;
    for (float& p : probs) p /= p_sum;

    std::mt19937 gen(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float r = dis(gen);

    float acc = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (r <= acc) {
            return candidate_tokens[i].id;
        }
    }

    return candidate_tokens.back().id;
}

} // namespace blocktensor::sampling
