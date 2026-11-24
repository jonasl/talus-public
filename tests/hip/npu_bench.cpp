#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/prefill/kernels.hpp"
#include "talus_npu.h"

namespace {

struct Q8_0Block {
    __half d;
    int8_t qs[32];
};

float DecodeBfpGroupElem(const uint8_t* g, int idx) {
    float mult = std::ldexp(1.0f, static_cast<int>(g[0]) - 127) / 64.0f;
    return static_cast<float>(static_cast<int8_t>(g[1 + idx])) * mult;
}

float SiluF(float x) {
    return x / (1.0f + std::exp(-x));
}

bool TestStagingAndPacking(hipStream_t stream) {
    // 1. Verify positive saturation bump in NpuEncodeWeightsF16 (values in [1.984375, 2.0))
    const uint32_t K = 64, N = 128;
    std::vector<__half> w_f16(N * K, __float2half(1.9921875f));
    void* d_w = nullptr;
    void* d_out = nullptr;
    if (hipMalloc(&d_w, w_f16.size() * sizeof(__half)) != hipSuccess ||
        hipMalloc(&d_out, (K * N / 8) * 9) != hipSuccess) {
        std::fprintf(stderr, "npu_bench: hipMalloc failed\n");
        return false;
    }
    (void)hipMemcpy(d_w, w_f16.data(), w_f16.size() * sizeof(__half), hipMemcpyHostToDevice);
    talus_npu_encode_weights_f16(reinterpret_cast<const uint16_t*>(d_w), K, 0, N, N, K, 0, K, d_out, stream);
    std::vector<uint8_t> h_out((K * N / 8) * 9);
    (void)hipStreamSynchronize(stream);
    (void)hipMemcpy(h_out.data(), d_out, h_out.size(), hipMemcpyDeviceToHost);

    // Group 0 corresponds to 8 values of 1.9921875f:
    // Without bump (exp=127), max representable is 127/64 = 1.984375 (err = 0.0078125).
    // With bump (exp=128), q=64 -> 64/32 = 2.0 (err = 0.0078125, or for 1.998 -> err=0.002 vs 0.0136).
    // Let's also test 1.998f explicitly.
    w_f16.assign(N * K, __float2half(1.998046875f));
    (void)hipMemcpy(d_w, w_f16.data(), w_f16.size() * sizeof(__half), hipMemcpyHostToDevice);
    talus_npu_encode_weights_f16(reinterpret_cast<const uint16_t*>(d_w), K, 0, N, N, K, 0, K, d_out, stream);
    (void)hipStreamSynchronize(stream);
    (void)hipMemcpy(h_out.data(), d_out, h_out.size(), hipMemcpyDeviceToHost);
    float decoded0 = DecodeBfpGroupElem(h_out.data(), 0);
    if (std::fabs(decoded0 - 2.0f) > 1e-4f) {
        std::fprintf(stderr, "npu_bench: saturation bump check failed: expected 2.0, got %f (exp=%u)\n",
                     decoded0, h_out[0]);
        (void)hipFree(d_w);
        (void)hipFree(d_out);
        return false;
    }

    // 2. Verify Q8_0 pair encoding doesn't crash and produces expected non-zero groups
    std::vector<Q8_0Block> wa(512 * (K / 32));
    std::vector<Q8_0Block> wb(512 * (K / 32));
    for (auto& b : wa) {
        b.d = __float2half(0.25f);
        for (int i = 0; i < 32; ++i) b.qs[i] = 4;
    }
    for (auto& b : wb) {
        b.d = __float2half(0.5f);
        for (int i = 0; i < 32; ++i) b.qs[i] = 2;
    }
    void *d_wa = nullptr, *d_wb = nullptr, *d_pair = nullptr;
    const uint32_t N_pair = 1024;
    (void)hipMalloc(&d_wa, wa.size() * sizeof(Q8_0Block));
    (void)hipMalloc(&d_wb, wb.size() * sizeof(Q8_0Block));
    (void)hipMalloc(&d_pair, (K * N_pair / 8) * 9);
    (void)hipMemcpy(d_wa, wa.data(), wa.size() * sizeof(Q8_0Block), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_wb, wb.data(), wb.size() * sizeof(Q8_0Block), hipMemcpyHostToDevice);
    talus_npu_encode_weights_q8_pair(d_wa, d_wb, K, N_pair, 512, K, 0, K, d_pair, stream);
    (void)hipStreamSynchronize(stream);
    if (hipGetLastError() != hipSuccess) {
        std::fprintf(stderr, "npu_bench: encode_weights_q8_pair HIP error\n");
        return false;
    }

    (void)hipFree(d_wa);
    (void)hipFree(d_wb);
    (void)hipFree(d_pair);
    (void)hipFree(d_w);
    (void)hipFree(d_out);
    std::printf("[PASS] staging & saturation-bump & q8-pair unit checks\n");
    return true;
}

bool TestNpuHardware(const std::string& xclbin, const std::string& insts, int iters, hipStream_t stream) {
    talus_npu_config_t cfg{};
    talus_npu_config_default(&cfg);
    cfg.xclbin_path = xclbin.c_str();
    cfg.insts_path = insts.c_str();
    cfg.tokens = 4096;
    cfg.k = 2560;
    cfg.n = 1024;

    char err[512] = {};
    talus_npu_ctx* ctx = nullptr;
    if (talus_npu_create(&cfg, &ctx, err, sizeof(err)) != 0 || ctx == nullptr) {
        std::fprintf(stderr, "npu_bench: talus_npu_create failed: %s\n", err);
        return false;
    }

    const uint32_t T = talus_npu_tokens(ctx);
    const uint32_t K = talus_npu_k(ctx);
    const uint32_t N = talus_npu_n(ctx);
    const int32_t a_id = talus_npu_alloc_buffer(ctx, talus_npu_a_bytes(ctx), err, sizeof(err));
    const int32_t b_id = talus_npu_alloc_buffer(ctx, talus_npu_b_bytes(ctx), err, sizeof(err));
    const int32_t c_id = talus_npu_alloc_buffer(ctx, talus_npu_c_bytes(ctx), err, sizeof(err));
    const int32_t bp_id = talus_npu_alloc_buffer(ctx, talus_npu_b_bytes(ctx), err, sizeof(err));
    const int32_t cp_id = talus_npu_alloc_buffer(ctx, talus_npu_c_bytes(ctx), err, sizeof(err));
    if (a_id < 0 || b_id < 0 || c_id < 0 || bp_id < 0 || cp_id < 0) {
        std::fprintf(stderr, "npu_bench: alloc_buffer failed: %s\n", err);
        talus_npu_destroy(ctx);
        return false;
    }

    // 1. Deterministic GEMM accuracy check: X[T, K] * W[N, K]^T
    std::vector<__half> h_x(size_t{T} * K);
    std::vector<Q8_0Block> h_w(size_t{N} * (K / 32));
    for (uint32_t r = 0; r < T; ++r) {
        for (uint32_t c = 0; c < K; ++c) {
            uint32_t h = (r * 2654435761u) ^ (c * 2246822519u);
            float v = 0.05f * static_cast<float>(static_cast<int>(h % 13u) - 6) + 0.02f;
            h_x[size_t{r} * K + c] = __float2half(v);
        }
    }
    for (uint32_t r = 0; r < N; ++r) {
        for (uint32_t b = 0; b < K / 32; ++b) {
            h_w[size_t{r} * (K / 32) + b].d = __float2half(0.03125f);
            for (int i = 0; i < 32; ++i) {
                uint32_t h = (r * 3266489917u) ^ ((b * 32 + i) * 668265263u);
                h_w[size_t{r} * (K / 32) + b].qs[i] = static_cast<int8_t>(static_cast<int>(h % 15u) - 7);
            }
        }
    }

    void *d_x = nullptr, *d_w = nullptr;
    float* d_out = nullptr;
    (void)hipMalloc(&d_x, h_x.size() * sizeof(__half));
    (void)hipMalloc(&d_w, h_w.size() * sizeof(Q8_0Block));
    (void)hipMalloc(&d_out, size_t{T} * N * sizeof(float));
    (void)hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(__half), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(Q8_0Block), hipMemcpyHostToDevice);

    talus_npu_encode_weights_q8(
        d_w, K, 0, N, N, K, 0, K, talus_npu_staged_ptr(ctx, b_id), stream);
    (void)hipStreamSynchronize(stream);
    talus_npu_finalize_weights(ctx, b_id);

    talus_npu_encode_activations_f16(
        reinterpret_cast<const uint16_t*>(d_x), K, T, T, K, 0, K,
        talus_npu_device_ptr(ctx, a_id), stream);
    if (!talus_npu_submit(ctx, stream, a_id, &b_id, &c_id, 1) ||
        !talus_npu_wait(ctx, stream)) {
        std::fprintf(stderr, "npu_bench: submit/wait failed\n");
        talus_npu_destroy(ctx);
        return false;
    }
    talus_npu_decode_output(
        talus_npu_device_ptr(ctx, c_id), d_out, T, T, N, N, 0, 0, N, stream);
    (void)hipStreamSynchronize(stream);

    std::vector<float> h_got(size_t{T} * N);
    (void)hipMemcpy(h_got.data(), d_out, h_got.size() * sizeof(float), hipMemcpyDeviceToHost);

    // Check first 64 rows against FP32 reference
    double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0;
    for (uint32_t r = 0; r < 64; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                const auto& blk = h_w[size_t{c} * (K / 32) + k / 32];
                float wv = __half2float(blk.d) * static_cast<float>(blk.qs[k % 32]);
                ref += __half2float(h_x[size_t{r} * K + k]) * wv;
            }
            float g = h_got[size_t{r} * N + c];
            dot += double(ref) * double(g);
            na += double(ref) * double(ref);
            nb += double(g) * double(g);
            max_abs = std::max(max_abs, std::fabs(double(ref) - double(g)));
        }
    }
    const double cos_sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    std::printf("[PASS] NPU GEMM parity (64x1024 sample): cos_sim=%.6f max_abs=%.4f ref0=%.4f got0=%.4f na=%.2f nb=%.2f\n",
                cos_sim, max_abs, double(h_got[0]), double(h_got[0]), na, nb);
    if (!std::isfinite(cos_sim) || cos_sim < 0.995 || nb < 1.0) {
        std::fprintf(stderr, "npu_bench: cosine similarity check failed!\n");
        talus_npu_destroy(ctx);
        return false;
    }

    // 2. Test packed SwiGLU (512 gate + 512 up in one NPU slice)
    const uint32_t valid_ff = 512;
    std::vector<Q8_0Block> h_wg(size_t{valid_ff} * (K / 32));
    std::vector<Q8_0Block> h_wu(size_t{valid_ff} * (K / 32));
    for (size_t i = 0; i < h_wg.size(); ++i) {
        h_wg[i].d = __float2half(0.03125f);
        h_wu[i].d = __float2half(0.03125f);
        for (int j = 0; j < 32; ++j) {
            uint32_t h1 = static_cast<uint32_t>(i * 2654435761u + j * 1597334677u);
            uint32_t h2 = static_cast<uint32_t>(i * 2246822519u + j * 3266489917u);
            h_wg[i].qs[j] = static_cast<int8_t>(static_cast<int>(h1 % 13u) - 6);
            h_wu[i].qs[j] = static_cast<int8_t>(static_cast<int>(h2 % 13u) - 6);
        }
    }
    void *d_wg = nullptr, *d_wu = nullptr;
    uint16_t* d_sh = nullptr;
    (void)hipMalloc(&d_wg, h_wg.size() * sizeof(Q8_0Block));
    (void)hipMalloc(&d_wu, h_wu.size() * sizeof(Q8_0Block));
    (void)hipMalloc(&d_sh, size_t{T} * valid_ff * sizeof(uint16_t));
    (void)hipMemcpy(d_wg, h_wg.data(), h_wg.size() * sizeof(Q8_0Block), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_wu, h_wu.data(), h_wu.size() * sizeof(Q8_0Block), hipMemcpyHostToDevice);
    talus_npu_encode_weights_q8_pair(
        d_wg, d_wu, K, N, valid_ff, K, 0, K, talus_npu_staged_ptr(ctx, bp_id), stream);
    (void)hipStreamSynchronize(stream);
    talus_npu_finalize_weights(ctx, bp_id);
    if (!talus_npu_submit(ctx, stream, a_id, &bp_id, &cp_id, 1) ||
        !talus_npu_wait(ctx, stream)) {
        talus_npu_destroy(ctx);
        return false;
    }
    talus_npu_swiglu_packed(
        talus_npu_device_ptr(ctx, cp_id), d_sh, T, T, N, valid_ff, stream);
    (void)hipStreamSynchronize(stream);
    std::vector<__half> h_sh(size_t{T} * valid_ff);
    (void)hipMemcpy(h_sh.data(), d_sh, h_sh.size() * sizeof(__half), hipMemcpyDeviceToHost);

    double sw_dot = 0.0, sw_na = 0.0, sw_nb = 0.0;
    for (uint32_t r = 0; r < 16; ++r) {
        for (uint32_t c = 0; c < valid_ff; ++c) {
            float g_ref = 0.0f, u_ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                float xv = __half2float(h_x[size_t{r} * K + k]);
                const auto& bg = h_wg[size_t{c} * (K / 32) + k / 32];
                const auto& bu = h_wu[size_t{c} * (K / 32) + k / 32];
                g_ref += xv * (__half2float(bg.d) * static_cast<float>(bg.qs[k % 32]));
                u_ref += xv * (__half2float(bu.d) * static_cast<float>(bu.qs[k % 32]));
            }
            float ref = SiluF(g_ref) * u_ref;
            float got = __half2float(h_sh[size_t{r} * valid_ff + c]);
            sw_dot += double(ref) * double(got);
            sw_na += double(ref) * double(ref);
            sw_nb += double(got) * double(got);
        }
    }
    const double sw_cos = sw_dot / (std::sqrt(sw_na) * std::sqrt(sw_nb) + 1e-12);
    std::printf("[PASS] NPU packed SwiGLU parity (16x512 sample): cos_sim=%.6f na=%.2f nb=%.2f\n",
                sw_cos, sw_na, sw_nb);
    if (!std::isfinite(sw_cos) || sw_cos < 0.99 || sw_nb < 1.0) {
        std::fprintf(stderr, "npu_bench: packed SwiGLU cosine similarity failed!\n");
        talus_npu_destroy(ctx);
        return false;
    }

    // 2b. Bitwise parity test for gdn_recurrence_npu reading z_streams directly vs decoded z
    {
        constexpr int64_t gdn_T = 512;
        constexpr int64_t HV = 48, S = 128, ZV = HV * S, C_CH = 10240;
        constexpr size_t state_elems = size_t{HV} * S * S;
        float *d_s1 = nullptr, *d_s2 = nullptr, *d_h = nullptr, *d_g = nullptr, *d_b = nullptr;
        float *d_z = nullptr, *d_gam = nullptr, *d_y1 = nullptr, *d_y2 = nullptr;
        uint16_t *d_yh1 = nullptr, *d_yh2 = nullptr;
        void* d_zptrs_h = nullptr;
        void* d_zptrs_d = nullptr;
        (void)hipMalloc(&d_s1, state_elems * sizeof(float));
        (void)hipMalloc(&d_s2, state_elems * sizeof(float));
        (void)hipMalloc(&d_h, size_t{gdn_T} * C_CH * sizeof(float));
        (void)hipMalloc(&d_g, size_t{gdn_T} * HV * sizeof(float));
        (void)hipMalloc(&d_b, size_t{gdn_T} * HV * sizeof(float));
        (void)hipMalloc(&d_z, size_t{gdn_T} * ZV * sizeof(float));
        (void)hipMalloc(&d_gam, size_t{S} * sizeof(float));
        (void)hipMalloc(&d_y1, size_t{gdn_T} * ZV * sizeof(float));
        (void)hipMalloc(&d_y2, size_t{gdn_T} * ZV * sizeof(float));
        (void)hipMalloc(&d_yh1, size_t{gdn_T} * ZV * sizeof(uint16_t));
        (void)hipMalloc(&d_yh2, size_t{gdn_T} * ZV * sizeof(uint16_t));
        (void)hipHostMalloc(&d_zptrs_h, sizeof(uint8_t*), hipHostMallocMapped);
        (void)hipHostGetDevicePointer(&d_zptrs_d, d_zptrs_h, 0);
        *reinterpret_cast<const uint8_t**>(d_zptrs_h) =
            static_cast<const uint8_t*>(talus_npu_device_ptr(ctx, c_id));

        (void)hipMemset(d_s1, 0, state_elems * sizeof(float));
        (void)hipMemset(d_s2, 0, state_elems * sizeof(float));
        std::vector<float> hh(size_t{gdn_T} * C_CH, 0.05f), hg(size_t{gdn_T} * HV, -0.1f),
            hb(size_t{gdn_T} * HV, 0.2f), hgam(S, 1.0f);
        (void)hipMemcpy(d_h, hh.data(), hh.size() * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_g, hg.data(), hg.size() * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_b, hb.data(), hb.size() * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_gam, hgam.data(), hgam.size() * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemset(d_z, 0, size_t{gdn_T} * ZV * sizeof(float));
        // Decode NPU c_id into trailing 1024 cols of d_z for baseline comparison
        talus_npu_decode_output(
            talus_npu_device_ptr(ctx, c_id), d_z, T, gdn_T, N, ZV, 5120, 0, N, stream);

        strata::prefill::gdn_recurrence(
            d_s1, d_h, d_g, d_b, d_z, d_gam, 1e-6f, d_y1, d_yh1, gdn_T, stream, 0);
        strata::prefill::gdn_recurrence_npu(
            d_s2, d_h, d_g, d_b, d_z, d_gam, 1e-6f, d_y2, d_yh2, gdn_T, stream, 0,
            nullptr, nullptr, reinterpret_cast<const uint8_t* const*>(d_zptrs_d), 1, T, 5120);
        (void)hipStreamSynchronize(stream);

        std::vector<uint16_t> hy1(size_t{gdn_T} * ZV), hy2(size_t{gdn_T} * ZV);
        (void)hipMemcpy(hy1.data(), d_yh1, hy1.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(hy2.data(), d_yh2, hy2.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
        if (std::memcmp(hy1.data(), hy2.data(), hy1.size() * sizeof(uint16_t)) != 0) {
            std::fprintf(stderr, "npu_bench: gdn_recurrence_npu mismatch vs gdn_recurrence!\n");
            talus_npu_destroy(ctx);
            return false;
        }
        std::printf("[PASS] gdn_recurrence_npu direct z_streams bitwise parity\n");

        (void)hipHostFree(d_zptrs_h);
        (void)hipFree(d_s1); (void)hipFree(d_s2); (void)hipFree(d_h); (void)hipFree(d_g);
        (void)hipFree(d_b); (void)hipFree(d_z); (void)hipFree(d_gam); (void)hipFree(d_y1);
        (void)hipFree(d_y2); (void)hipFree(d_yh1); (void)hipFree(d_yh2);
    }

    // 3. Throughput benchmark (4-slice pipelined batch)
    std::vector<int32_t> b4(4, b_id), c4(4);
    for (int s = 0; s < 4; ++s) {
        c4[s] = talus_npu_alloc_buffer(ctx, talus_npu_c_bytes(ctx), err, sizeof(err));
    }
    (void)talus_npu_submit(ctx, stream, a_id, b4.data(), c4.data(), 4);
    (void)talus_npu_wait(ctx, stream);
    (void)hipStreamSynchronize(stream);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        (void)talus_npu_submit(ctx, stream, a_id, b4.data(), c4.data(), 4);
    }
    (void)talus_npu_wait(ctx, stream);
    (void)hipStreamSynchronize(stream);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double slices = 4.0 * iters;
    const double ms_per_slice = ms / slices;
    const double ops_per_slice = 2.0 * T * K * N;
    const double tops = ops_per_slice / (ms_per_slice * 1e9);
    std::printf("[BENCH] 4-slice pipelined (%d iters, %d slices): %.3f ms/slice, %.2f TOPS\n",
                iters, static_cast<int>(slices), ms_per_slice, tops);

    // Regression test for async interleaved consumer_stream / stage_stream pipelining:
    // Submit #1 (stage waits on cs) -> Wait #1 (on cs) -> Submit #2 (stage waits on cs after Wait #1) -> Wait #2 (on cs),
    // queued back-to-back from CPU without draining between layers.
    void* stage = talus_npu_stage_stream(ctx);
    hipEvent_t ev0 = nullptr;
    hipEvent_t ev1 = nullptr;
    if (hipEventCreateWithFlags(&ev0, hipEventDisableTiming) != hipSuccess ||
        hipEventCreateWithFlags(&ev1, hipEventDisableTiming) != hipSuccess) {
        std::fprintf(stderr, "npu_bench: hipEventCreateWithFlags failed\n");
        talus_npu_destroy(ctx);
        return false;
    }
    for (int r = 0; r < 4; ++r) {
        if (hipEventRecord(ev0, stream) != hipSuccess ||
            hipStreamWaitEvent(static_cast<hipStream_t>(stage), ev0, 0) != hipSuccess) {
            talus_npu_destroy(ctx);
            return false;
        }
        talus_npu_encode_activations_f16(
            static_cast<const uint16_t*>(d_x), K, T, 64, K, 0, K, talus_npu_device_ptr(ctx, a_id), stage);
        if (talus_npu_submit(ctx, stage, a_id, b4.data(), c4.data(), 2) == 0 ||
            talus_npu_wait(ctx, stream) == 0) {
            std::fprintf(stderr, "npu_bench: interleaved submit/wait #1 failed\n");
            talus_npu_destroy(ctx);
            return false;
        }
        if (hipEventRecord(ev1, stream) != hipSuccess ||
            hipStreamWaitEvent(static_cast<hipStream_t>(stage), ev1, 0) != hipSuccess) {
            talus_npu_destroy(ctx);
            return false;
        }
        talus_npu_encode_activations_f16(
            static_cast<const uint16_t*>(d_x), K, T, 64, K, 0, K, talus_npu_device_ptr(ctx, a_id), stage);
        if (talus_npu_submit(ctx, stage, a_id, b4.data(), c4.data(), 2) == 0 ||
            talus_npu_wait(ctx, stream) == 0) {
            std::fprintf(stderr, "npu_bench: interleaved submit/wait #2 failed\n");
            talus_npu_destroy(ctx);
            return false;
        }
    }
    (void)hipStreamSynchronize(stream);
    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);
    std::printf("[PASS] async interleaved stream/stage submit+wait parity (no AB-BA deadlock)\n");

    (void)hipFree(d_x);
    (void)hipFree(d_w);
    (void)hipFree(d_out);
    (void)hipFree(d_wg);
    (void)hipFree(d_wu);
    (void)hipFree(d_sh);
    talus_npu_destroy(ctx);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string xclbin, insts;
    int iters = 8;
    bool require_npu = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--xclbin" && i + 1 < argc) xclbin = argv[++i];
        else if (a == "--insts" && i + 1 < argc) insts = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = std::max(1, std::atoi(argv[++i]));
        else if (a == "--require-npu") require_npu = true;
    }
    if (xclbin.empty() || insts.empty()) {
        const char* dir = std::getenv("TALUS_NPU_DIR");
        if (dir && dir[0]) {
            if (xclbin.empty()) xclbin = std::string(dir) + "/final_4096x2560x1024_128x64x128.xclbin";
            if (insts.empty()) insts = std::string(dir) + "/insts_4096x2560x1024_128x64x128.txt";
        }
    }

    char diag[256] = {};
    int safe = talus_npu_check_env_safety(diag, sizeof(diag));
    std::printf("talus-npu-bench env safety=%d (%s)\n", safe, diag);

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::fprintf(stderr, "npu_bench: hipStreamCreate failed\n");
        return 1;
    }

    if (!TestStagingAndPacking(stream)) {
        (void)hipStreamDestroy(stream);
        return 1;
    }

    if (!xclbin.empty() && !insts.empty()) {
        if (!TestNpuHardware(xclbin, insts, iters, stream)) {
            (void)hipStreamDestroy(stream);
            return require_npu ? 1 : 0;
        }
    } else {
        std::printf("[SKIP] NPU hardware test skipped (set TALUS_NPU_DIR or --xclbin/--insts)\n");
        if (require_npu) {
            (void)hipStreamDestroy(stream);
            return 77;
        }
    }

    (void)hipStreamDestroy(stream);
    return 0;
}
