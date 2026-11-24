#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "strata/prefill/gemm.hpp"

namespace talus::npu {

struct WeightDesc {
    const void* native_data{nullptr};
    int native_type{-1};
    const void* bf16_data{nullptr};
    bool is_bf16{false};
    int64_t ne0{0};
    int64_t ne1{0};
};

class PrefillNpu {
public:
    static PrefillNpu* GetOrCreate(std::string* err);
    static bool EnabledInEnv();

    ~PrefillNpu();
    PrefillNpu(const PrefillNpu&) = delete;
    PrefillNpu& operator=(const PrefillNpu&) = delete;

    bool Available() const noexcept;
    bool Eligible(int64_t t_n) const noexcept;
    void SetContext(const void* wt_ptr);

    // GDN: offloads trailing columns of `attn_gate.weight` (`z`: [t_n, 6144]).
    // On success returns true, sets `z_gpu_cols_out` (columns GPU must compute into `z`).
    bool StartGdnZ(
        void* cs,
        strata::prefill::Gemm* gemm,
        int64_t layer,
        const WeightDesc& wg,
        const uint16_t* mixed_h,
        int64_t t_n,
        int64_t* z_gpu_cols_out
    );
    void GdnRecurrence(
        float* state, const float* h, const float* gate, const float* beta, const float* z,
        const float* gamma, float eps, float* y, uint16_t* y16, int64_t t, void* stream,
        int64_t ld16
    );
    void WaitGdnZ(void* cs);

    // QSA: offloads trailing columns of `attn_q.weight` (`qf`: [t_n, 12288]).
    bool StartQsaQ(
        void* cs,
        strata::prefill::Gemm* gemm,
        int64_t layer,
        const WeightDesc& wq,
        const uint16_t* mixed_h,
        int64_t t_n,
        int64_t* q_gpu_cols_out
    );
    bool FinishQsaQ(void* cs, float* qf_dev, int64_t t_n);

    // MoE shared expert: offloads `ffn_gate_shexp` + `ffn_up_shexp` and fuses SiLU+mul directly to FP16 `sh_h`.
    bool StartMoeShexp(
        void* cs,
        strata::prefill::Gemm* gemm,
        int64_t layer,
        const WeightDesc& wsg,
        const WeightDesc& wsu,
        const uint16_t* mixed_h,
        int64_t t_n
    );
    bool FinishMoeShexp(void* cs, uint16_t* sh_h_dev, int64_t t_n);

    void EndChunk();

    static void WaitCallback(void* arg);
    void* RawCtx() const noexcept;

private:
    PrefillNpu();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace talus::npu
