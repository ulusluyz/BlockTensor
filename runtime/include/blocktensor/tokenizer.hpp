#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace blocktensor {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    void build_vocab(const std::vector<std::string>& vocab);

    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(uint32_t token_id) const;
    std::string decode(const std::vector<uint32_t>& tokens) const;

    uint32_t vocab_size() const { return static_cast<uint32_t>(id_to_token_.size()); }
    uint32_t bos_token_id() const { return bos_id_; }
    uint32_t eos_token_id() const { return eos_id_; }

private:
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    uint32_t bos_id_{1};
    uint32_t eos_id_{2};
    uint32_t unk_id_{0};
};

} // namespace blocktensor
