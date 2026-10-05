#include "blocktensor/tokenizer.hpp"
#include <sstream>

namespace blocktensor {

void Tokenizer::build_vocab(const std::vector<std::string>& vocab) {
    id_to_token_ = vocab;
    token_to_id_.clear();
    for (size_t i = 0; i < vocab.size(); ++i) {
        token_to_id_[vocab[i]] = static_cast<uint32_t>(i);
    }
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> tokens;
    if (text.empty()) return tokens;

    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        auto it = token_to_id_.find(word);
        if (it != token_to_id_.end()) {
            tokens.push_back(it->second);
        } else {
            for (char c : word) {
                std::string s(1, c);
                auto cit = token_to_id_.find(s);
                if (cit != token_to_id_.end()) {
                    tokens.push_back(cit->second);
                } else {
                    tokens.push_back(unk_id_);
                }
            }
        }
    }
    return tokens;
}

std::string Tokenizer::decode(uint32_t token_id) const {
    if (token_id < id_to_token_.size()) {
        return id_to_token_[token_id];
    }
    return "<unk>";
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += " ";
        result += decode(tokens[i]);
    }
    return result;
}

} // namespace blocktensor
