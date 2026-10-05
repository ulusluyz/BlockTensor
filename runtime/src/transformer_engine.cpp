#include "blocktensor/transformer_engine.hpp"
#include "kernels/cpu_ref.hpp"
#include "quantization/quant_types.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace blocktensor {

static std::vector<uint32_t> load_spirv(const std::string& filename) {
    std::vector<std::string> paths = {
        filename,
        "shaders/" + filename,
        "../shaders/" + filename,
        "../../shaders/" + filename,
        "build/shaders/" + filename
    };
    std::string found;
    for (const auto& p : paths) {
        if (fs::exists(p)) { found = p; break; }
    }
    if (found.empty()) throw std::runtime_error("TransformerEngine: Cannot find SPIR-V file " + filename);

    std::ifstream file(found, std::ios::binary | std::ios::ate);
    size_t sz = file.tellg();
    std::vector<uint32_t> buf(sz / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

TransformerEngine::TransformerEngine(std::shared_ptr<IComputeBackend> backend, const TransformerWeights& weights)
    : backend_(std::move(backend)), mem_(backend_), weights_(weights), cfg_(weights.config) {

    x_buf_ = mem_.allocate(cfg_.dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);
    norm_x_buf_ = mem_.allocate(cfg_.dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);

    q_buf_ = mem_.allocate(cfg_.dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);
    k_buf_ = mem_.allocate(cfg_.num_kv_heads * cfg_.head_dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);
    v_buf_ = mem_.allocate(cfg_.num_kv_heads * cfg_.head_dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);

    attn_out_buf_ = mem_.allocate(cfg_.dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);

    gate_buf_ = mem_.allocate(cfg_.hidden_dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);
    up_buf_ = mem_.allocate(cfg_.hidden_dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);
    mlp_out_buf_ = mem_.allocate(cfg_.dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);

    logits_buf_ = mem_.allocate(cfg_.vocab_size * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);

    if (backend_->backend_name() == "Vulkan_Compute") {
        rmsnorm_pipeline_ = backend_->create_pipeline_from_spirv("rmsnorm", load_spirv("rmsnorm.comp.spv"), 3, 8);
        rope_pipeline_     = backend_->create_pipeline_from_spirv("rope", load_spirv("rope.comp.spv"), 2, 16);
        softmax_pipeline_  = backend_->create_pipeline_from_spirv("softmax", load_spirv("softmax.comp.spv"), 2, 8);
        silu_pipeline_     = backend_->create_pipeline_from_spirv("silu", load_spirv("silu.comp.spv"), 2, 4);

        if (weights_.quant_type == DataType::INT8) {
            matvec_pipeline_ = backend_->create_pipeline_from_spirv("matvec_q8_0", load_spirv("matvec_q8_0.comp.spv"), 3, 8);
        } else {
            matvec_pipeline_ = backend_->create_pipeline_from_spirv("matvec_q4_0", load_spirv("matvec_q4_0.comp.spv"), 3, 8);
        }
    }
}

void TransformerEngine::dispatch_matvec(
    const std::shared_ptr<IComputeBuffer>& W,
    const std::shared_ptr<IComputeBuffer>& X,
    const std::shared_ptr<IComputeBuffer>& Y,
    uint32_t M, uint32_t N) {

    if (backend_->backend_name() == "Vulkan_Compute") {
        struct PushParams { uint32_t M; uint32_t N; } params{M, N};
        backend_->dispatch(matvec_pipeline_, {W, X, Y}, {M, 1, 1}, &params, sizeof(params));
    } else {
        std::vector<float> x_host(N);
        std::vector<float> y_host(M, 0.0f);
        mem_.copy_device_to_host(X, x_host.data(), N * sizeof(float));

        if (weights_.quant_type == DataType::INT8) {
            std::vector<quantization::BlockQ8_0> w_host(M * (N / quantization::Q8_BLOCK_SIZE));
            mem_.copy_device_to_host(W, w_host.data(), w_host.size() * sizeof(quantization::BlockQ8_0));
            quantization::matvec_q8_0(w_host.data(), x_host.data(), y_host.data(), M, N);
        } else {
            std::vector<quantization::BlockQ4_0> w_host(M * (N / quantization::Q4_BLOCK_SIZE));
            mem_.copy_device_to_host(W, w_host.data(), w_host.size() * sizeof(quantization::BlockQ4_0));
            quantization::matvec_q4_0(w_host.data(), x_host.data(), y_host.data(), M, N);
        }
        mem_.copy_host_to_device(y_host.data(), Y, M * sizeof(float));
    }
}

void TransformerEngine::dispatch_rmsnorm(
    const std::shared_ptr<IComputeBuffer>& X,
    const std::shared_ptr<IComputeBuffer>& W,
    const std::shared_ptr<IComputeBuffer>& Out,
    uint32_t num_rows, uint32_t dim, float eps) {

    if (backend_->backend_name() == "Vulkan_Compute") {
        struct PushParams { uint32_t dim; float eps; } params{dim, eps};
        backend_->dispatch(rmsnorm_pipeline_, {X, W, Out}, {num_rows, 1, 1}, &params, sizeof(params));
    } else {
        std::vector<float> x_host(num_rows * dim), w_host(dim), out_host(num_rows * dim);
        mem_.copy_device_to_host(X, x_host.data(), x_host.size() * sizeof(float));
        mem_.copy_device_to_host(W, w_host.data(), w_host.size() * sizeof(float));
        cpu_ref::rmsnorm(x_host.data(), w_host.data(), out_host.data(), num_rows, dim, eps);
        mem_.copy_host_to_device(out_host.data(), Out, out_host.size() * sizeof(float));
    }
}

void TransformerEngine::execute_layer(uint32_t layer_idx, uint32_t pos, KVCache& kv_cache) {
    const auto& layer = weights_.layers[layer_idx];

    dispatch_rmsnorm(x_buf_, layer.attn_norm_w, norm_x_buf_, 1, cfg_.dim, cfg_.rms_norm_eps);

    dispatch_matvec(layer.q_proj_w, norm_x_buf_, q_buf_, cfg_.dim, cfg_.dim);
    dispatch_matvec(layer.k_proj_w, norm_x_buf_, k_buf_, cfg_.num_kv_heads * cfg_.head_dim, cfg_.dim);
    dispatch_matvec(layer.v_proj_w, norm_x_buf_, v_buf_, cfg_.num_kv_heads * cfg_.head_dim, cfg_.dim);

    if (backend_->backend_name() == "Vulkan_Compute") {
        struct PushParams { uint32_t seq_len; uint32_t num_heads; uint32_t head_dim; uint32_t pos_offset; };
        PushParams q_params{1, cfg_.num_heads, cfg_.head_dim, pos};
        uint32_t q_workgroups = ((cfg_.num_heads * cfg_.head_dim) / 2 + 255) / 256;
        backend_->dispatch(rope_pipeline_, {q_buf_, q_buf_}, {q_workgroups, 1, 1}, &q_params, sizeof(q_params));

        PushParams k_params{1, cfg_.num_kv_heads, cfg_.head_dim, pos};
        uint32_t k_workgroups = ((cfg_.num_kv_heads * cfg_.head_dim) / 2 + 255) / 256;
        backend_->dispatch(rope_pipeline_, {k_buf_, k_buf_}, {k_workgroups, 1, 1}, &k_params, sizeof(k_params));
    } else {
        std::vector<float> q_host(cfg_.dim), k_host(cfg_.num_kv_heads * cfg_.head_dim);
        mem_.copy_device_to_host(q_buf_, q_host.data(), q_host.size() * sizeof(float));
        mem_.copy_device_to_host(k_buf_, k_host.data(), k_host.size() * sizeof(float));
        cpu_ref::rope(q_host.data(), q_host.data(), 1, cfg_.num_heads, cfg_.head_dim, pos);
        cpu_ref::rope(k_host.data(), k_host.data(), 1, cfg_.num_kv_heads, cfg_.head_dim, pos);
        mem_.copy_host_to_device(q_host.data(), q_buf_, q_host.size() * sizeof(float));
        mem_.copy_host_to_device(k_host.data(), k_buf_, k_host.size() * sizeof(float));
    }

    std::vector<float> k_host(cfg_.num_kv_heads * cfg_.head_dim);
    std::vector<float> v_host(cfg_.num_kv_heads * cfg_.head_dim);
    mem_.copy_device_to_host(k_buf_, k_host.data(), k_host.size() * sizeof(float));
    mem_.copy_device_to_host(v_buf_, v_host.data(), v_host.size() * sizeof(float));
    kv_cache.update(layer_idx, pos, k_host.data(), v_host.data());

    uint32_t seq_len_ctx = pos + 1;
    std::vector<float> q_rope(cfg_.dim);
    mem_.copy_device_to_host(q_buf_, q_rope.data(), q_rope.size() * sizeof(float));

    std::vector<float> k_cache_all(seq_len_ctx * cfg_.num_kv_heads * cfg_.head_dim);
    std::vector<float> v_cache_all(seq_len_ctx * cfg_.num_kv_heads * cfg_.head_dim);
    mem_.copy_device_to_host(kv_cache.k_cache(layer_idx), k_cache_all.data(), k_cache_all.size() * sizeof(float));
    mem_.copy_device_to_host(kv_cache.v_cache(layer_idx), v_cache_all.data(), v_cache_all.size() * sizeof(float));

    std::vector<float> attn_out(cfg_.dim, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

    for (uint32_t h = 0; h < cfg_.num_heads; ++h) {
        uint32_t kv_h = h / (cfg_.num_heads / cfg_.num_kv_heads);
        const float* q_h = q_rope.data() + h * cfg_.head_dim;

        std::vector<float> scores(seq_len_ctx);
        for (uint32_t t = 0; t < seq_len_ctx; ++t) {
            const float* k_t = k_cache_all.data() + (t * cfg_.num_kv_heads + kv_h) * cfg_.head_dim;
            float score = 0.0f;
            for (uint32_t d = 0; d < cfg_.head_dim; ++d) {
                score += q_h[d] * k_t[d];
            }
            scores[t] = score * scale;
        }

        cpu_ref::softmax(scores.data(), scores.data(), 1, seq_len_ctx);

        float* out_h = attn_out.data() + h * cfg_.head_dim;
        for (uint32_t t = 0; t < seq_len_ctx; ++t) {
            const float* v_t = v_cache_all.data() + (t * cfg_.num_kv_heads + kv_h) * cfg_.head_dim;
            float p = scores[t];
            for (uint32_t d = 0; d < cfg_.head_dim; ++d) {
                out_h[d] += p * v_t[d];
            }
        }
    }

    mem_.copy_host_to_device(attn_out.data(), attn_out_buf_, attn_out.size() * sizeof(float));

    std::shared_ptr<IComputeBuffer> o_out_buf = mem_.allocate(cfg_.dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);
    dispatch_matvec(layer.o_proj_w, attn_out_buf_, o_out_buf, cfg_.dim, cfg_.dim);

    std::vector<float> x_curr(cfg_.dim), o_out_curr(cfg_.dim);
    mem_.copy_device_to_host(x_buf_, x_curr.data(), x_curr.size() * sizeof(float));
    mem_.copy_device_to_host(o_out_buf, o_out_curr.data(), o_out_curr.size() * sizeof(float));
    for (uint32_t d = 0; d < cfg_.dim; ++d) x_curr[d] += o_out_curr[d];
    mem_.copy_host_to_device(x_curr.data(), x_buf_, x_curr.size() * sizeof(float));

    dispatch_rmsnorm(x_buf_, layer.ffn_norm_w, norm_x_buf_, 1, cfg_.dim, cfg_.rms_norm_eps);
    dispatch_matvec(layer.gate_proj_w, norm_x_buf_, gate_buf_, cfg_.hidden_dim, cfg_.dim);
    dispatch_matvec(layer.up_proj_w, norm_x_buf_, up_buf_, cfg_.hidden_dim, cfg_.dim);

    if (backend_->backend_name() == "Vulkan_Compute") {
        struct PushParams { uint32_t num_elements; } params{cfg_.hidden_dim};
        uint32_t workgroups = (cfg_.hidden_dim + 255) / 256;
        backend_->dispatch(silu_pipeline_, {gate_buf_, gate_buf_}, {workgroups, 1, 1}, &params, sizeof(params));
    } else {
        std::vector<float> gate_host(cfg_.hidden_dim);
        mem_.copy_device_to_host(gate_buf_, gate_host.data(), gate_host.size() * sizeof(float));
        cpu_ref::silu(gate_host.data(), gate_host.data(), cfg_.hidden_dim);
        mem_.copy_host_to_device(gate_host.data(), gate_buf_, gate_host.size() * sizeof(float));
    }

    std::vector<float> silu_gate_host(cfg_.hidden_dim), up_host(cfg_.hidden_dim), mlp_act(cfg_.hidden_dim);
    mem_.copy_device_to_host(gate_buf_, silu_gate_host.data(), silu_gate_host.size() * sizeof(float));
    mem_.copy_device_to_host(up_buf_, up_host.data(), up_host.size() * sizeof(float));
    for (uint32_t i = 0; i < cfg_.hidden_dim; ++i) mlp_act[i] = silu_gate_host[i] * up_host[i];

    std::shared_ptr<IComputeBuffer> mlp_act_buf = mem_.allocate(cfg_.hidden_dim * sizeof(float), DataType::FLOAT32, BufferUsage::STORAGE, MemoryLocation::HOST_VISIBLE);
    mem_.copy_host_to_device(mlp_act.data(), mlp_act_buf, mlp_act.size() * sizeof(float));

    dispatch_matvec(layer.down_proj_w, mlp_act_buf, mlp_out_buf_, cfg_.dim, cfg_.hidden_dim);

    std::vector<float> mlp_out_curr(cfg_.dim);
    mem_.copy_device_to_host(mlp_out_buf_, mlp_out_curr.data(), mlp_out_curr.size() * sizeof(float));
    for (uint32_t d = 0; d < cfg_.dim; ++d) x_curr[d] += mlp_out_curr[d];
    mem_.copy_host_to_device(x_curr.data(), x_buf_, x_curr.size() * sizeof(float));
}

std::vector<float> TransformerEngine::forward_single_token(uint32_t token_id, uint32_t pos, KVCache& kv_cache) {
    if (token_id >= cfg_.vocab_size) {
        throw std::out_of_range("Token ID out of vocab bounds");
    }

    std::vector<float> embed_table(cfg_.vocab_size * cfg_.dim);
    mem_.copy_device_to_host(weights_.embed_tokens_w, embed_table.data(), embed_table.size() * sizeof(float));

    std::vector<float> x(cfg_.dim);
    std::memcpy(x.data(), embed_table.data() + token_id * cfg_.dim, cfg_.dim * sizeof(float));
    mem_.copy_host_to_device(x.data(), x_buf_, x.size() * sizeof(float));

    for (uint32_t l = 0; l < cfg_.num_layers; ++l) {
        execute_layer(l, pos, kv_cache);
    }

    dispatch_rmsnorm(x_buf_, weights_.output_norm_w, norm_x_buf_, 1, cfg_.dim, cfg_.rms_norm_eps);
    dispatch_matvec(weights_.output_head_w, norm_x_buf_, logits_buf_, cfg_.vocab_size, cfg_.dim);

    std::vector<float> logits(cfg_.vocab_size);
    mem_.copy_device_to_host(logits_buf_, logits.data(), logits.size() * sizeof(float));

    return logits;
}

} // namespace blocktensor
