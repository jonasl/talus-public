#include "talus_npu.h"
#include "npu_gemm.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace talus::npu {
void LaunchEncodeActivationsF16(const std::uint16_t* x_f16, std::size_t ldx,
                                std::uint32_t tokens, std::uint32_t valid_tokens,
                                std::uint32_t k_slice, std::uint32_t k_offset,
                                std::uint32_t k_valid, void* dst, hipStream_t stream);
void LaunchEncodeWeightsQ8(const void* w_q8_0, std::uint32_t k_total, std::uint32_t row0,
                           std::uint32_t cols, std::uint32_t valid_cols, std::uint32_t k_slice,
                           std::uint32_t k_offset, std::uint32_t k_valid, void* dst,
                           hipStream_t stream);
void LaunchEncodeWeightsQ8Pair(const void* wa_q8_0, const void* wb_q8_0, std::uint32_t k_total,
                               std::uint32_t cols, std::uint32_t valid_cols, std::uint32_t k_slice,
                               std::uint32_t k_offset, std::uint32_t k_valid, void* dst,
                               hipStream_t stream);
void LaunchEncodeWeightsF16(const std::uint16_t* w_f16, std::uint32_t k_total, std::uint32_t row0,
                            std::uint32_t cols, std::uint32_t valid_cols, std::uint32_t k_slice,
                            std::uint32_t k_offset, std::uint32_t k_valid, void* dst,
                            hipStream_t stream);
void LaunchEncodeWeightsBF16(const uint16_t* w_bf16, std::uint32_t k_total, std::uint32_t row0,
                             std::uint32_t cols, std::uint32_t valid_cols, std::uint32_t k_slice,
                             std::uint32_t k_offset, std::uint32_t k_valid, void* dst,
                             hipStream_t stream);
void LaunchDecodeOutput(const void* src, float* out, std::uint32_t tokens,
                        std::uint32_t valid_tokens, std::uint32_t cols, std::size_t out_pitch,
                        std::uint32_t col0, bool accumulate, std::uint32_t valid_cols,
                        hipStream_t stream);
void LaunchSwiglu(const void* gate_c, const void* up_c, std::uint16_t* out_f16,
                  std::uint32_t tokens, std::uint32_t valid_tokens, std::uint32_t cols,
                  std::uint32_t valid_cols, hipStream_t stream);
void LaunchSwigluPacked(const void* packed_c, std::uint16_t* out_f16,
                        std::uint32_t tokens, std::uint32_t valid_tokens, std::uint32_t cols,
                        std::uint32_t valid_cols, hipStream_t stream);
}  // namespace talus::npu

struct talus_npu_ctx {
    std::unique_ptr<talus::npu::NpuGemm> gemm;
};

namespace {
void PutErr(char* err, size_t cap, const std::string& msg) {
    if (err && cap > 0) {
        std::snprintf(err, cap, "%s", msg.c_str());
    }
}
bool EnvFlag(const char* name, bool def_val) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return def_val;
    return v[0] != '0';
}
}  // namespace

extern "C" {

void talus_npu_config_default(talus_npu_config_t* out) {
    if (!out) return;
    *out = talus_npu_config_t{};
    out->tokens = 4096;
    out->k = 2560;
    out->n = 1024;
    out->timeout_ms = 8000;
    if (const char* t = std::getenv("TALUS_NPU_TIMEOUT_MS")) {
        out->timeout_ms = static_cast<uint32_t>(std::max(1, std::atoi(t)));
    } else if (const char* t = std::getenv("GUFO_NPU_TIMEOUT_MS")) {
        out->timeout_ms = static_cast<uint32_t>(std::max(1, std::atoi(t)));
    }
    out->isolate = EnvFlag("TALUS_NPU_ISOLATE", false) || EnvFlag("GUFO_NPU_ISOLATE", false) ? 1 : 0;
    out->keepalive = (EnvFlag("TALUS_NPU_KEEPALIVE", true) && EnvFlag("GUFO_NPU_KEEPALIVE", true)) ? 1 : 0;
    const bool req_held = EnvFlag("TALUS_NPU_REQUIRE_HELD_FCLK", false);
    out->allow_unheld_fclk = (!req_held && EnvFlag("TALUS_NPU_ALLOW_UNHELD_FCLK", true)) ? 1 : 0;
}

int32_t talus_npu_create(const talus_npu_config_t* cfg, talus_npu_ctx** out_ctx, char* err, size_t err_cap) {
    if (!cfg || !out_ctx || !cfg->xclbin_path || !cfg->insts_path) {
        PutErr(err, err_cap, "talus_npu_create: invalid null argument");
        return -1;
    }
    *out_ctx = nullptr;
    talus::npu::NpuGemm::Config c;
    c.xclbin = cfg->xclbin_path;
    c.insts = cfg->insts_path;
    if (cfg->kernel_name && cfg->kernel_name[0]) c.kernel_name = cfg->kernel_name;
    c.tokens = cfg->tokens ? cfg->tokens : 4096;
    c.k = cfg->k ? cfg->k : 2560;
    c.n = cfg->n ? cfg->n : 1024;
    c.timeout_ms = cfg->timeout_ms ? cfg->timeout_ms : 8000;
    c.isolate = cfg->isolate != 0;
    c.keepalive = cfg->keepalive != 0;
    c.allow_unheld_fclk = cfg->allow_unheld_fclk != 0;

    std::string e;
    auto g = talus::npu::NpuGemm::Create(c, &e);
    if (!g) {
        PutErr(err, err_cap, e);
        return -1;
    }
    auto* ctx = new talus_npu_ctx();
    ctx->gemm = std::move(g);
    *out_ctx = ctx;
    return 0;
}

void talus_npu_destroy(talus_npu_ctx* ctx) {
    delete ctx;
}

int32_t talus_npu_healthy(const talus_npu_ctx* ctx) {
    return (ctx && ctx->gemm && ctx->gemm->Healthy()) ? 1 : 0;
}

uint32_t talus_npu_tokens(const talus_npu_ctx* ctx) { return ctx && ctx->gemm ? ctx->gemm->tokens() : 0; }
uint32_t talus_npu_k(const talus_npu_ctx* ctx) { return ctx && ctx->gemm ? ctx->gemm->k() : 0; }
uint32_t talus_npu_n(const talus_npu_ctx* ctx) { return ctx && ctx->gemm ? ctx->gemm->n() : 0; }
size_t talus_npu_a_bytes(const talus_npu_ctx* ctx) { return ctx && ctx->gemm ? ctx->gemm->ABytes() : 0; }
size_t talus_npu_b_bytes(const talus_npu_ctx* ctx) { return ctx && ctx->gemm ? ctx->gemm->BBytes() : 0; }
size_t talus_npu_c_bytes(const talus_npu_ctx* ctx) { return ctx && ctx->gemm ? ctx->gemm->CBytes() : 0; }

int32_t talus_npu_alloc_buffer(talus_npu_ctx* ctx, size_t bytes, char* err, size_t err_cap) {
    if (!ctx || !ctx->gemm) {
        PutErr(err, err_cap, "talus_npu_alloc_buffer: null ctx");
        return -1;
    }
    std::string e;
    int id = ctx->gemm->NewBuffer(bytes, &e);
    if (id < 0) PutErr(err, err_cap, e);
    return id;
}

void* talus_npu_host_ptr(const talus_npu_ctx* ctx, int32_t buf_id) {
    return (ctx && ctx->gemm && buf_id >= 0) ? ctx->gemm->Host(buf_id) : nullptr;
}
void* talus_npu_device_ptr(const talus_npu_ctx* ctx, int32_t buf_id) {
    return (ctx && ctx->gemm && buf_id >= 0) ? ctx->gemm->Device(buf_id) : nullptr;
}
void* talus_npu_staged_ptr(const talus_npu_ctx* ctx, int32_t buf_id) {
    return (ctx && ctx->gemm && buf_id >= 0) ? ctx->gemm->Staged(buf_id) : nullptr;
}

void talus_npu_finalize_weights(talus_npu_ctx* ctx, int32_t buf_id) {
    if (ctx && ctx->gemm && buf_id >= 0) ctx->gemm->FinalizeWeights(buf_id);
}

void* talus_npu_stage_stream(const talus_npu_ctx* ctx) {
    return (ctx && ctx->gemm) ? static_cast<void*>(ctx->gemm->StageStream()) : nullptr;
}

int32_t talus_npu_submit(talus_npu_ctx* ctx, void* ready_stream, int32_t a_id,
                         const int32_t* b_ids, const int32_t* c_ids, int32_t count) {
    if (!ctx || !ctx->gemm || count <= 0 || !b_ids || !c_ids) return 0;
    return ctx->gemm->Submit(static_cast<hipStream_t>(ready_stream), a_id, b_ids, c_ids, count) ? 1 : 0;
}

int32_t talus_npu_register_kernel(talus_npu_ctx* ctx, const char* kernel_name, const char* insts_path,
                                  uint32_t tokens, uint32_t k, uint32_t n, char* err, size_t err_cap) {
    if (!ctx || !ctx->gemm || !kernel_name || !insts_path) {
        PutErr(err, err_cap, "talus_npu_register_kernel: invalid null argument");
        return -1;
    }
    std::string e;
    int id = ctx->gemm->RegisterKernel(kernel_name, insts_path, tokens, k, n, &e);
    if (id < 0) PutErr(err, err_cap, e);
    return id;
}

int32_t talus_npu_find_kernel(const talus_npu_ctx* ctx, uint32_t tokens, uint32_t k, uint32_t n) {
    return (ctx && ctx->gemm) ? ctx->gemm->FindKernel(tokens, k, n) : -1;
}

int32_t talus_npu_submit_kernel(talus_npu_ctx* ctx, int32_t kernel_id, void* ready_stream, int32_t a_id,
                                const int32_t* b_ids, const int32_t* c_ids, int32_t count) {
    if (!ctx || !ctx->gemm || count <= 0 || !b_ids || !c_ids) return 0;
    return ctx->gemm->SubmitKernel(kernel_id, static_cast<hipStream_t>(ready_stream), a_id, b_ids, c_ids, count) ? 1 : 0;
}

int32_t talus_npu_wait(talus_npu_ctx* ctx, void* consumer_stream) {
    if (!ctx || !ctx->gemm) return 0;
    return ctx->gemm->Wait(static_cast<hipStream_t>(consumer_stream)) ? 1 : 0;
}

void talus_npu_get_stats(const talus_npu_ctx* ctx, talus_npu_stats_t* out) {
    if (ctx && ctx->gemm && out) ctx->gemm->Stats(out);
}

int32_t talus_npu_check_env_safety(char* diag, size_t diag_cap) {
    std::string msg;
    bool held = false;
    bool safe = talus::npu::NpuGemm::CheckEnvSafety(&msg, &held);
    PutErr(diag, diag_cap, msg);
    return safe ? 1 : 0;
}

void talus_npu_encode_activations_f16(const uint16_t* x_f16, size_t ldx, uint32_t tokens,
                                      uint32_t valid_tokens, uint32_t k_slice, uint32_t k_offset,
                                      uint32_t k_valid, void* dst_dev, void* hip_stream) {
    talus::npu::LaunchEncodeActivationsF16(x_f16, ldx, tokens, valid_tokens, k_slice, k_offset,
                                           k_valid, dst_dev, static_cast<hipStream_t>(hip_stream));
}

void talus_npu_encode_weights_q8(const void* w_q8_0, uint32_t k_total, uint32_t row0,
                                 uint32_t cols, uint32_t valid_cols, uint32_t k_slice, uint32_t k_offset,
                                 uint32_t k_valid, void* dst_dev, void* hip_stream) {
    talus::npu::LaunchEncodeWeightsQ8(w_q8_0, k_total, row0, cols, valid_cols, k_slice, k_offset, k_valid, dst_dev,
                                      static_cast<hipStream_t>(hip_stream));
}

void talus_npu_encode_weights_q8_pair(const void* w_a_q8_0, const void* w_b_q8_0, uint32_t k_total,
                                      uint32_t cols, uint32_t valid_cols, uint32_t k_slice,
                                      uint32_t k_offset, uint32_t k_valid, void* dst_dev, void* hip_stream) {
    talus::npu::LaunchEncodeWeightsQ8Pair(w_a_q8_0, w_b_q8_0, k_total, cols, valid_cols, k_slice, k_offset, k_valid,
                                          dst_dev, static_cast<hipStream_t>(hip_stream));
}

void talus_npu_encode_weights_f16(const uint16_t* w_f16, uint32_t k_total, uint32_t row0,
                                  uint32_t cols, uint32_t valid_cols, uint32_t k_slice, uint32_t k_offset,
                                  uint32_t k_valid, void* dst_dev, void* hip_stream) {
    talus::npu::LaunchEncodeWeightsF16(w_f16, k_total, row0, cols, valid_cols, k_slice, k_offset, k_valid, dst_dev,
                                       static_cast<hipStream_t>(hip_stream));
}

void talus_npu_encode_weights_bf16(const uint16_t* w_bf16, uint32_t k_total, uint32_t row0,
                                   uint32_t cols, uint32_t valid_cols, uint32_t k_slice, uint32_t k_offset,
                                   uint32_t k_valid, void* dst_dev, void* hip_stream) {
    talus::npu::LaunchEncodeWeightsBF16(w_bf16, k_total, row0, cols, valid_cols, k_slice, k_offset, k_valid, dst_dev,
                                        static_cast<hipStream_t>(hip_stream));
}

void talus_npu_decode_output(const void* src_dev, float* out_dev, uint32_t tokens,
                             uint32_t valid_tokens, uint32_t cols, size_t out_pitch, uint32_t col0,
                             int32_t accumulate, uint32_t valid_cols, void* hip_stream) {
    talus::npu::LaunchDecodeOutput(src_dev, out_dev, tokens, valid_tokens, cols, out_pitch, col0,
                                   accumulate != 0, valid_cols, static_cast<hipStream_t>(hip_stream));
}

void talus_npu_swiglu(const void* gate_c_dev, const void* up_c_dev, uint16_t* out_f16_dev,
                      uint32_t tokens, uint32_t valid_tokens, uint32_t cols, uint32_t valid_cols,
                      void* hip_stream) {
    talus::npu::LaunchSwiglu(gate_c_dev, up_c_dev, out_f16_dev, tokens, valid_tokens, cols,
                             valid_cols, static_cast<hipStream_t>(hip_stream));
}

void talus_npu_swiglu_packed(const void* packed_c_dev, uint16_t* out_f16_dev,
                             uint32_t tokens, uint32_t valid_tokens, uint32_t cols,
                             uint32_t valid_cols, void* hip_stream) {
    talus::npu::LaunchSwigluPacked(packed_c_dev, out_f16_dev, tokens, valid_tokens, cols,
                                   valid_cols, static_cast<hipStream_t>(hip_stream));
}

}  // extern "C"
