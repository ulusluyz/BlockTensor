#include "models/model_importer.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace blocktensor::importer {

constexpr uint32_t BTMODEL_MAGIC = 0x42544d44;

void export_btmodel(const std::string& filepath, const TransformerWeights& weights, MemoryManager& mem) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Failed to open file for export: " + filepath);

    out.write(reinterpret_cast<const char*>(&BTMODEL_MAGIC), sizeof(BTMODEL_MAGIC));
    out.write(reinterpret_cast<const char*>(&weights.config), sizeof(ModelConfig));
    uint32_t qt = static_cast<uint32_t>(weights.quant_type);
    out.write(reinterpret_cast<const char*>(&qt), sizeof(qt));

    auto write_buf = [&](const std::shared_ptr<IComputeBuffer>& buf) {
        size_t sz = buf->size();
        out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        std::vector<uint8_t> data(sz);
        mem.copy_device_to_host(buf, data.data(), sz);
        out.write(reinterpret_cast<const char*>(data.data()), sz);
    };

    write_buf(weights.embed_tokens_w);
    write_buf(weights.output_norm_w);
    write_buf(weights.output_head_w);

    for (const auto& layer : weights.layers) {
        write_buf(layer.attn_norm_w);
        write_buf(layer.ffn_norm_w);
        write_buf(layer.q_proj_w);
        write_buf(layer.k_proj_w);
        write_buf(layer.v_proj_w);
        write_buf(layer.o_proj_w);
        write_buf(layer.gate_proj_w);
        write_buf(layer.up_proj_w);
        write_buf(layer.down_proj_w);
    }
}

TransformerWeights import_btmodel(const std::string& filepath, std::shared_ptr<IComputeBackend> backend, MemoryManager& mem) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Failed to open file for import: " + filepath);

    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != BTMODEL_MAGIC) throw std::runtime_error("Invalid .btmodel magic header");

    TransformerWeights weights;
    in.read(reinterpret_cast<char*>(&weights.config), sizeof(ModelConfig));
    uint32_t qt = 0;
    in.read(reinterpret_cast<char*>(&qt), sizeof(qt));
    weights.quant_type = static_cast<DataType>(qt);

    auto read_buf = [&](DataType dt) {
        size_t sz = 0;
        in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        std::vector<uint8_t> data(sz);
        in.read(reinterpret_cast<char*>(data.data()), sz);

        auto buf = mem.allocate(sz, dt, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);
        mem.copy_host_to_device(data.data(), buf, sz);
        return buf;
    };

    weights.embed_tokens_w = read_buf(DataType::FLOAT32);
    weights.output_norm_w  = read_buf(DataType::FLOAT32);
    weights.output_head_w  = read_buf(weights.quant_type);

    for (uint32_t l = 0; l < weights.config.num_layers; ++l) {
        LayerWeights layer;
        layer.attn_norm_w = read_buf(DataType::FLOAT32);
        layer.ffn_norm_w  = read_buf(DataType::FLOAT32);
        layer.q_proj_w    = read_buf(weights.quant_type);
        layer.k_proj_w    = read_buf(weights.quant_type);
        layer.v_proj_w    = read_buf(weights.quant_type);
        layer.o_proj_w    = read_buf(weights.quant_type);
        layer.gate_proj_w = read_buf(weights.quant_type);
        layer.up_proj_w   = read_buf(weights.quant_type);
        layer.down_proj_w = read_buf(weights.quant_type);
        weights.layers.push_back(layer);
    }

    return weights;
}

} // namespace blocktensor::importer
