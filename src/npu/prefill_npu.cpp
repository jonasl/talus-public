#include "prefill_npu.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/prefill/kernels.hpp"
#include "talus_npu.h"

namespace talus::npu {
namespace {

constexpr uint32_t kNpuTokens = 4096;
constexpr uint32_t kNpuK = 2560;
constexpr uint32_t kNpuN = 1024;
constexpr int64_t kMaxGroups = 4;
constexpr uint32_t kZCols = 6144;
constexpr uint32_t kZSlicesMax = 6;
constexpr uint32_t kQCols = 12288;
constexpr uint32_t kQSlicesMax = 12;
constexpr uint32_t kShexpCols = 640;
constexpr size_t kReadyEventPoolSize = 256;

enum class OpKind : uint8_t {
    GdnZ = 0,
    QsaQ = 1,
    ShexpGate = 2,
    ShexpUp = 3,
};

struct SliceKey {
    int64_t layer;
    uint8_t op;
    uint32_t slice;
    bool operator==(const SliceKey& o) const noexcept {
        return layer == o.layer && op == o.op && slice == o.slice;
    }
};

struct SliceKeyHash {
    size_t operator()(const SliceKey& k) const noexcept {
        return (static_cast<size_t>(k.layer) * 1315423911u) ^
               (static_cast<size_t>(k.op) << 16) ^
               static_cast<size_t>(k.slice);
    }
};

struct CachedWeight {
    int32_t buf_id{-1};
    const void* src_ptr{nullptr};
    int src_type{-1};
};

int EnvInt(const char* name, int def_val) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return def_val;
    return std::atoi(v);
}

std::string ResolveDir() {
    const char* d = std::getenv("TALUS_NPU_DIR");
    if (!d || !d[0]) d = std::getenv("GUFO_NPU_DIR");
    return (d && d[0]) ? std::string(d) : std::string();
}

}  // namespace

struct PrefillNpu::Impl {
    talus_npu_ctx* ctx{nullptr};
    std::vector<hipEvent_t> ready_evs;
    size_t ready_ev_cursor{0};
    hipStream_t weight_stream{nullptr};
    int64_t min_t{2048};
    int64_t max_groups{kMaxGroups};

    int z_slices_cfg{-1};  // -1 = auto
    int q_slices_cfg{-1};  // -1 = auto
    bool shexp_on{true};
    bool auto_tune{true};

    // Active slice counts (may be adjusted by auto-tuner)
    uint32_t z_slices{2};
    uint32_t q_slices{4};

    // Pooled A and C NPU buffers sized for up to `alloc_groups` groups of 4096 tokens
    int64_t alloc_groups{0};
    std::vector<int32_t> a_bufs;   // [alloc_groups]
    std::vector<int32_t> c_pool;   // [alloc_groups * kQSlicesMax]
    const uint8_t** z_ptrs_host{nullptr};
    const uint8_t** z_ptrs_dev{nullptr};
    uint16_t* w_scratch_f16{nullptr};

    const void* current_wt{nullptr};
    std::unordered_map<SliceKey, CachedWeight, SliceKeyHash> weight_cache;

    // Active in-flight tracking for current half-layer
    std::atomic<bool> op_inflight{false};
    int64_t active_groups{0};
    uint32_t active_z_gpu_cols{kZCols};
    uint32_t active_q_gpu_cols{kQCols};
    uint32_t active_q_slices{0};

    // Auto-tune telemetry
    double last_wait_us{0.0};
    uint64_t last_waits{0};
    int gdn_calls_in_chunk{0};
    int qsa_calls_in_chunk{0};
    double gdn_wait_us_accum{0.0};
    int gdn_wait_samples{0};
    double qsa_wait_us_accum{0.0};
    int qsa_wait_samples{0};

    std::mutex mutex;

    ~Impl() {
        if (w_scratch_f16) (void)hipFree(w_scratch_f16);
        if (z_ptrs_host) (void)hipHostFree(const_cast<uint8_t**>(z_ptrs_host));
        for (hipEvent_t ev : ready_evs) {
            if (ev) (void)hipEventDestroy(ev);
        }
        if (weight_stream) (void)hipStreamDestroy(weight_stream);
        if (ctx) talus_npu_destroy(ctx);
    }

    hipEvent_t NextReadyEvent() {
        hipEvent_t ev = ready_evs[ready_ev_cursor % ready_evs.size()];
        ++ready_ev_cursor;
        return ev;
    }

    bool EnsureCapacity(int64_t groups) {
        if (groups <= alloc_groups) return true;
        char err[256] = {};
        while (alloc_groups < groups) {
            int32_t a_id = talus_npu_alloc_buffer(ctx, talus_npu_a_bytes(ctx), err, sizeof(err));
            if (a_id < 0) {
                std::fprintf(stderr, "talus npu: alloc A failed: %s\n", err);
                return false;
            }
            a_bufs.push_back(a_id);
            const size_t c_bytes = talus_npu_c_bytes(ctx);
            for (uint32_t s = 0; s < kQSlicesMax; ++s) {
                int32_t c_id = talus_npu_alloc_buffer(ctx, c_bytes, err, sizeof(err));
                if (c_id < 0) {
                    std::fprintf(stderr, "talus npu: alloc C failed: %s\n", err);
                    return false;
                }
                c_pool.push_back(c_id);
            }
            ++alloc_groups;
        }
        if (z_ptrs_host == nullptr) {
            void* h = nullptr;
            void* d = nullptr;
            const size_t bytes = static_cast<size_t>(kMaxGroups * kZSlicesMax) * sizeof(uint8_t*);
            if (hipHostMalloc(&h, bytes, hipHostMallocMapped) != hipSuccess ||
                hipHostGetDevicePointer(&d, h, 0) != hipSuccess) {
                if (h) (void)hipHostFree(h);
                return false;
            }
            z_ptrs_host = reinterpret_cast<const uint8_t**>(h);
            z_ptrs_dev = reinterpret_cast<const uint8_t**>(d);
        }
        for (int64_t g = 0; g < alloc_groups; ++g) {
            for (uint32_t s = 0; s < kZSlicesMax; ++s) {
                const int32_t c_id = c_pool[static_cast<size_t>(g * kQSlicesMax + s)];
                z_ptrs_host[g * kZSlicesMax + s] =
                    static_cast<const uint8_t*>(talus_npu_staged_ptr(ctx, c_id));
            }
        }
        return true;
    }

    int32_t EnsureWeightSlice(
        [[maybe_unused]] void* cs,
        [[maybe_unused]] strata::prefill::Gemm* gemm,
        int64_t layer,
        OpKind op,
        uint32_t slice,
        uint32_t row0,
        uint32_t valid_cols,
        const WeightDesc& w
    ) {
        const void* src_ptr = w.native_data ? w.native_data : w.bf16_data;
        const int src_type = w.native_data ? w.native_type : (w.is_bf16 ? 30 : -1);
        if (!src_ptr || src_type < 0 || w.ne0 != kNpuK) return -1;

        SliceKey key{layer, static_cast<uint8_t>(op), slice};
        auto it = weight_cache.find(key);
        if (it != weight_cache.end() && it->second.src_ptr == src_ptr && it->second.src_type == src_type) {
            return it->second.buf_id;
        }

        int32_t buf_id = (it != weight_cache.end()) ? it->second.buf_id : -1;
        if (buf_id < 0) {
            char err[256] = {};
            buf_id = talus_npu_alloc_buffer(ctx, talus_npu_b_bytes(ctx), err, sizeof(err));
            if (buf_id < 0) {
                std::fprintf(stderr, "talus npu: alloc weight buffer failed: %s\n", err);
                return -1;
            }
        }

        void* dst_dev = talus_npu_staged_ptr(ctx, buf_id);
        void* stream_use = weight_stream;
        if (src_type == 8 && w.native_data != nullptr) {
            talus_npu_encode_weights_q8(
                w.native_data, kNpuK, row0, kNpuN, valid_cols, kNpuK, 0, kNpuK, dst_dev, stream_use);
            (void)hipStreamSynchronize(static_cast<hipStream_t>(stream_use));
        } else if (src_type == 1) {
            talus_npu_encode_weights_f16(
                static_cast<const uint16_t*>(src_ptr), kNpuK, row0, kNpuN, valid_cols,
                kNpuK, 0, kNpuK, dst_dev, stream_use);
            (void)hipStreamSynchronize(static_cast<hipStream_t>(stream_use));
        } else if (src_type == 30) {
            talus_npu_encode_weights_bf16(
                static_cast<const uint16_t*>(src_ptr), kNpuK, row0, kNpuN, valid_cols,
                kNpuK, 0, kNpuK, dst_dev, stream_use);
            (void)hipStreamSynchronize(static_cast<hipStream_t>(stream_use));
        } else {
            // Generic GGML quant fallback (Q4_K, Q6_K, IQ4_XS, etc.) via Strata dequant_f16
            if (w_scratch_f16 == nullptr) {
                if (hipMalloc(&w_scratch_f16, static_cast<size_t>(kNpuN) * kNpuK * sizeof(uint16_t)) != hipSuccess) {
                    return -1;
                }
            }
            strata::kernels::dequant_f16(src_type, src_ptr, row0, valid_cols, kNpuK, w_scratch_f16, stream_use);
            talus_npu_encode_weights_f16(
                w_scratch_f16, kNpuK, 0, kNpuN, valid_cols, kNpuK, 0, kNpuK, dst_dev, stream_use);
            (void)hipStreamSynchronize(static_cast<hipStream_t>(stream_use));
        }

        talus_npu_finalize_weights(ctx, buf_id);
        weight_cache[key] = CachedWeight{buf_id, src_ptr, src_type};
        return buf_id;
    }

    void UpdateStatsDelta(bool is_gdn) {
        talus_npu_stats_t st{};
        talus_npu_get_stats(ctx, &st);
        const double d_wait = std::max(0.0, st.host_wait_us_total - last_wait_us);
        const uint64_t d_cnt = st.waits >= last_waits ? (st.waits - last_waits) : 0;
        last_wait_us = st.host_wait_us_total;
        last_waits = st.waits;
        if (d_cnt == 0) return;
        const double per_wait = d_wait / static_cast<double>(d_cnt);
        if (is_gdn) {
            gdn_wait_us_accum += per_wait;
            gdn_wait_samples += 1;
        } else {
            qsa_wait_us_accum += per_wait;
            qsa_wait_samples += 1;
        }
    }
};

PrefillNpu::PrefillNpu() = default;
PrefillNpu::~PrefillNpu() = default;

bool PrefillNpu::EnabledInEnv() {
    if (!ResolveDir().empty()) return true;
    const char* x = std::getenv("TALUS_NPU_XCLBIN");
    const char* i = std::getenv("TALUS_NPU_INSTS");
    return x && x[0] && i && i[0];
}

PrefillNpu* PrefillNpu::GetOrCreate(std::string* err) {
    if (!EnabledInEnv()) return nullptr;
    static std::mutex init_mutex;
    static std::unique_ptr<PrefillNpu> instance;
    static bool attempted = false;
    static std::string cached_err;

    const std::lock_guard lock(init_mutex);
    if (attempted) {
        if (!instance && err) *err = cached_err;
        return instance.get();
    }
    attempted = true;

    const std::string dir = ResolveDir();
    const char* x_env = std::getenv("TALUS_NPU_XCLBIN");
    const char* i_env = std::getenv("TALUS_NPU_INSTS");
    std::string xclbin = (x_env && x_env[0]) ? std::string(x_env)
                                            : (dir + "/final_4096x2560x1024_128x64x128.xclbin");
    std::string insts = (i_env && i_env[0]) ? std::string(i_env)
                                            : (dir + "/insts_4096x2560x1024_128x64x128.txt");

    talus_npu_config_t cfg{};
    talus_npu_config_default(&cfg);
    cfg.xclbin_path = xclbin.c_str();
    cfg.insts_path = insts.c_str();
    cfg.tokens = kNpuTokens;
    cfg.k = kNpuK;
    cfg.n = kNpuN;

    char buf[512] = {};
    talus_npu_ctx* ctx = nullptr;
    if (talus_npu_create(&cfg, &ctx, buf, sizeof(buf)) != 0 || ctx == nullptr) {
        cached_err = buf[0] ? std::string(buf) : "talus_npu_create failed";
        std::fprintf(stderr, "talus npu: disabled (%s)\n", cached_err.c_str());
        if (err) *err = cached_err;
        return nullptr;
    }

    auto p = std::unique_ptr<PrefillNpu>(new PrefillNpu());
    p->impl_ = std::make_unique<Impl>();
    Impl& m = *p->impl_;
    m.ctx = ctx;
    if (hipStreamCreateWithFlags(&m.weight_stream, hipStreamNonBlocking) != hipSuccess) {
        cached_err = "hipStreamCreateWithFlags failed for PrefillNpu weight_stream";
        std::fprintf(stderr, "talus npu: disabled (%s)\n", cached_err.c_str());
        if (err) *err = cached_err;
        return nullptr;
    }
    m.ready_evs.resize(kReadyEventPoolSize, nullptr);
    for (auto& ev : m.ready_evs) {
        if (hipEventCreateWithFlags(&ev, hipEventDisableTiming) != hipSuccess) {
            cached_err = "hipEventCreateWithFlags failed for PrefillNpu";
            std::fprintf(stderr, "talus npu: disabled (%s)\n", cached_err.c_str());
            if (err) *err = cached_err;
            return nullptr;
        }
    }

    m.min_t = std::max(128, EnvInt("TALUS_NPU_MIN_T", 2048));
    m.max_groups = std::clamp(EnvInt("TALUS_NPU_MAX_GROUPS", static_cast<int>(kMaxGroups)), 1, static_cast<int>(kMaxGroups));
    m.z_slices_cfg = EnvInt("TALUS_NPU_Z_SLICES", -1);
    m.q_slices_cfg = EnvInt("TALUS_NPU_Q_SLICES", -1);
    m.shexp_on = EnvInt("TALUS_NPU_SHEXP", 1) != 0;
    m.auto_tune = EnvInt("TALUS_NPU_AUTO_TUNE", 1) != 0;

    if (m.z_slices_cfg >= 0) {
        m.z_slices = static_cast<uint32_t>(std::clamp(m.z_slices_cfg, 0, static_cast<int>(kZSlicesMax)));
    }
    if (m.q_slices_cfg >= 0) {
        m.q_slices = static_cast<uint32_t>(std::clamp(m.q_slices_cfg, 0, static_cast<int>(kQSlicesMax)));
    }

    char safety[256] = {};
    (void)talus_npu_check_env_safety(safety, sizeof(safety));
    std::fprintf(
        stderr,
        "talus npu: enabled (4096x2560x1024, min_T=%lld, z_slices=%u%s, q_slices=%u%s, shexp=%s, %s)\n",
        static_cast<long long>(m.min_t),
        m.z_slices, m.z_slices_cfg < 0 && m.auto_tune ? " auto" : "",
        m.q_slices, m.q_slices_cfg < 0 && m.auto_tune ? " auto" : "",
        m.shexp_on ? "on" : "off",
        safety
    );

    instance = std::move(p);
    return instance.get();
}

bool PrefillNpu::Available() const noexcept {
    return impl_ && impl_->ctx && talus_npu_healthy(impl_->ctx) != 0;
}

bool PrefillNpu::Eligible(int64_t t_n) const noexcept {
    if (!Available()) return false;
    if (t_n < impl_->min_t) return false;
    const int64_t groups = (t_n + kNpuTokens - 1) / kNpuTokens;
    return groups >= 1 && groups <= impl_->max_groups;
}

void PrefillNpu::SetContext(const void* wt_ptr) {
    if (!impl_) return;
    const std::lock_guard lock(impl_->mutex);
    if (impl_->current_wt != wt_ptr) {
        // Weight pointers may have been reallocated for a new model; entries also check src_ptr.
        impl_->current_wt = wt_ptr;
    }
}

void* PrefillNpu::RawCtx() const noexcept {
    return impl_ ? impl_->ctx : nullptr;
}

void PrefillNpu::WaitCallback(void* arg) {
    auto* self = static_cast<PrefillNpu*>(arg);
    if (self && self->impl_ && self->impl_->ctx) {
        // Note: stream is injected via gdn_recurrence_npu wrapper; see WaitGdnZ
    }
}

bool PrefillNpu::StartGdnZ(
    void* cs,
    strata::prefill::Gemm* gemm,
    int64_t layer,
    const WeightDesc& wg,
    const uint16_t* mixed_h,
    int64_t t_n,
    int64_t* z_gpu_cols_out
) {
    if (!Eligible(t_n) || wg.ne0 != kNpuK || wg.ne1 != kZCols || mixed_h == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    if (m.op_inflight.exchange(false)) {
        (void)talus_npu_wait(m.ctx, cs);
    }
    if (m.z_slices == 0) return false;

    const int64_t groups = (t_n + kNpuTokens - 1) / kNpuTokens;
    if (!m.EnsureCapacity(groups)) return false;

    const uint32_t slices = m.z_slices;
    const uint32_t gpu_cols = kZCols - slices * kNpuN;
    const uint32_t slice0 = kZSlicesMax - slices;

    std::vector<int32_t> b_ids(slices);
    for (uint32_t s = 0; s < slices; ++s) {
        const uint32_t row0 = gpu_cols + s * kNpuN;
        const int32_t bid = m.EnsureWeightSlice(cs, gemm, layer, OpKind::GdnZ, slice0 + s, row0, kNpuN, wg);
        if (bid < 0) return false;
        b_ids[s] = bid;
    }

    void* stage = talus_npu_stage_stream(m.ctx);
    hipEvent_t ready_ev = m.NextReadyEvent();
    if (hipEventRecord(ready_ev, static_cast<hipStream_t>(cs)) != hipSuccess) return false;
    if (hipStreamWaitEvent(static_cast<hipStream_t>(stage), ready_ev, 0) != hipSuccess) return false;

    for (int64_t g = 0; g < groups; ++g) {
        const uint32_t valid_t = static_cast<uint32_t>(std::min<int64_t>(kNpuTokens, t_n - g * kNpuTokens));
        const int32_t a_id = m.a_bufs[static_cast<size_t>(g)];
        const uint16_t* x_g = mixed_h + static_cast<size_t>(g) * kNpuTokens * kNpuK;
        talus_npu_encode_activations_f16(
            x_g, kNpuK, kNpuTokens, valid_t, kNpuK, 0, kNpuK, talus_npu_device_ptr(m.ctx, a_id), stage);

        std::vector<int32_t> c_ids(slices);
        for (uint32_t s = 0; s < slices; ++s) {
            c_ids[s] = m.c_pool[static_cast<size_t>(g * kQSlicesMax + s)];
        }
        if (talus_npu_submit(m.ctx, stage, a_id, b_ids.data(), c_ids.data(), static_cast<int32_t>(slices)) == 0) {
            if (g > 0) (void)talus_npu_wait(m.ctx, cs);
            return false;
        }
    }

    m.op_inflight.store(true);
    m.active_groups = groups;
    m.active_z_gpu_cols = gpu_cols;
    m.gdn_calls_in_chunk += 1;
    if (z_gpu_cols_out) *z_gpu_cols_out = static_cast<int64_t>(gpu_cols);
    return true;
}

void PrefillNpu::GdnRecurrence(
    float* state, const float* h, const float* gate, const float* beta, const float* z,
    const float* gamma, float eps, float* y, uint16_t* y16, int64_t t, void* stream,
    int64_t ld16
) {
    struct WaitArg {
        PrefillNpu* npu;
        void* stream;
    } arg{this, stream};
    auto cb = [](void* p) {
        auto* a = static_cast<WaitArg*>(p);
        a->npu->WaitGdnZ(a->stream);
    };
    const uint8_t* const* z_streams = nullptr;
    uint32_t z_gpu_cols = kZCols;
    {
        const std::lock_guard lock(impl_->mutex);
        z_streams = impl_->z_ptrs_dev;
        z_gpu_cols = impl_->active_z_gpu_cols;
    }
    strata::prefill::gdn_recurrence_npu(
        state, h, gate, beta, z, gamma, eps, y, y16, t, stream, ld16,
        cb, &arg, z_streams, kZSlicesMax, kNpuTokens, z_gpu_cols);
}

void PrefillNpu::WaitGdnZ(void* cs) {
    if (!impl_ || !impl_->ctx) return;
    talus_npu_wait(impl_->ctx, cs);
    const std::lock_guard lock(impl_->mutex);
    impl_->op_inflight.store(false);
    impl_->UpdateStatsDelta(true);
}

bool PrefillNpu::StartQsaQ(
    void* cs,
    strata::prefill::Gemm* gemm,
    int64_t layer,
    const WeightDesc& wq,
    const uint16_t* mixed_h,
    int64_t t_n,
    int64_t* q_gpu_cols_out
) {
    if (!Eligible(t_n) || wq.ne0 != kNpuK || wq.ne1 != kQCols || mixed_h == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    if (m.op_inflight.exchange(false)) {
        (void)talus_npu_wait(m.ctx, cs);
    }
    if (m.q_slices == 0) return false;

    const int64_t groups = (t_n + kNpuTokens - 1) / kNpuTokens;
    if (!m.EnsureCapacity(groups)) return false;

    const uint32_t slices = m.q_slices;
    const uint32_t gpu_cols = kQCols - slices * kNpuN;
    const uint32_t slice0 = kQSlicesMax - slices;

    std::vector<int32_t> b_ids(slices);
    for (uint32_t s = 0; s < slices; ++s) {
        const uint32_t row0 = gpu_cols + s * kNpuN;
        const int32_t bid = m.EnsureWeightSlice(cs, gemm, layer, OpKind::QsaQ, slice0 + s, row0, kNpuN, wq);
        if (bid < 0) return false;
        b_ids[s] = bid;
    }

    void* stage = talus_npu_stage_stream(m.ctx);
    hipEvent_t ready_ev = m.NextReadyEvent();
    if (hipEventRecord(ready_ev, static_cast<hipStream_t>(cs)) != hipSuccess) return false;
    if (hipStreamWaitEvent(static_cast<hipStream_t>(stage), ready_ev, 0) != hipSuccess) return false;

    for (int64_t g = 0; g < groups; ++g) {
        const uint32_t valid_t = static_cast<uint32_t>(std::min<int64_t>(kNpuTokens, t_n - g * kNpuTokens));
        const int32_t a_id = m.a_bufs[static_cast<size_t>(g)];
        const uint16_t* x_g = mixed_h + static_cast<size_t>(g) * kNpuTokens * kNpuK;
        talus_npu_encode_activations_f16(
            x_g, kNpuK, kNpuTokens, valid_t, kNpuK, 0, kNpuK, talus_npu_device_ptr(m.ctx, a_id), stage);

        std::vector<int32_t> c_ids(slices);
        for (uint32_t s = 0; s < slices; ++s) {
            c_ids[s] = m.c_pool[static_cast<size_t>(g * kQSlicesMax + s)];
        }
        if (talus_npu_submit(m.ctx, stage, a_id, b_ids.data(), c_ids.data(), static_cast<int32_t>(slices)) == 0) {
            if (g > 0) (void)talus_npu_wait(m.ctx, cs);
            return false;
        }
    }

    m.op_inflight.store(true);
    m.active_groups = groups;
    m.active_q_gpu_cols = gpu_cols;
    m.active_q_slices = slices;
    m.qsa_calls_in_chunk += 1;
    if (q_gpu_cols_out) *q_gpu_cols_out = static_cast<int64_t>(gpu_cols);
    return true;
}

bool PrefillNpu::FinishQsaQ(void* cs, float* qf_dev, int64_t t_n) {
    if (!impl_ || !impl_->ctx || qf_dev == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    m.op_inflight.store(false);
    if (talus_npu_wait(m.ctx, cs) == 0) return false;
    m.UpdateStatsDelta(false);

    for (int64_t g = 0; g < m.active_groups; ++g) {
        const uint32_t valid_t = static_cast<uint32_t>(std::min<int64_t>(kNpuTokens, t_n - g * kNpuTokens));
        float* dst_g = qf_dev + static_cast<size_t>(g) * kNpuTokens * kQCols;
        for (uint32_t s = 0; s < m.active_q_slices; ++s) {
            const int32_t c_id = m.c_pool[static_cast<size_t>(g * kQSlicesMax + s)];
            const uint32_t col0 = m.active_q_gpu_cols + s * kNpuN;
            talus_npu_decode_output(
                talus_npu_device_ptr(m.ctx, c_id), dst_g, kNpuTokens, valid_t, kNpuN,
                kQCols, col0, 0, kNpuN, cs);
        }
    }
    return true;
}

bool PrefillNpu::StartMoeShexp(
    void* cs,
    strata::prefill::Gemm* gemm,
    int64_t layer,
    const WeightDesc& wsg,
    const WeightDesc& wsu,
    const uint16_t* mixed_h,
    int64_t t_n
) {
    if (!Eligible(t_n) || !impl_->shexp_on || mixed_h == nullptr) return false;
    if (wsg.ne0 != kNpuK || wsu.ne0 != kNpuK || wsg.ne1 != kShexpCols || wsu.ne1 != kShexpCols) {
        return false;
    }
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    if (m.op_inflight.exchange(false)) {
        (void)talus_npu_wait(m.ctx, cs);
    }
    const int64_t groups = (t_n + kNpuTokens - 1) / kNpuTokens;
    if (!m.EnsureCapacity(groups)) return false;

    const int32_t bg = m.EnsureWeightSlice(cs, gemm, layer, OpKind::ShexpGate, 0, 0, kShexpCols, wsg);
    const int32_t bu = m.EnsureWeightSlice(cs, gemm, layer, OpKind::ShexpUp, 0, 0, kShexpCols, wsu);
    if (bg < 0 || bu < 0) return false;

    void* stage = talus_npu_stage_stream(m.ctx);
    hipEvent_t ready_ev = m.NextReadyEvent();
    if (hipEventRecord(ready_ev, static_cast<hipStream_t>(cs)) != hipSuccess) return false;
    if (hipStreamWaitEvent(static_cast<hipStream_t>(stage), ready_ev, 0) != hipSuccess) return false;

    const int32_t b_ids[2] = {bg, bu};
    for (int64_t g = 0; g < groups; ++g) {
        const uint32_t valid_t = static_cast<uint32_t>(std::min<int64_t>(kNpuTokens, t_n - g * kNpuTokens));
        const int32_t a_id = m.a_bufs[static_cast<size_t>(g)];
        const uint16_t* x_g = mixed_h + static_cast<size_t>(g) * kNpuTokens * kNpuK;
        talus_npu_encode_activations_f16(
            x_g, kNpuK, kNpuTokens, valid_t, kNpuK, 0, kNpuK, talus_npu_device_ptr(m.ctx, a_id), stage);

        const int32_t c_ids[2] = {
            m.c_pool[static_cast<size_t>(g * kQSlicesMax + 0)],
            m.c_pool[static_cast<size_t>(g * kQSlicesMax + 1)],
        };
        if (talus_npu_submit(m.ctx, stage, a_id, b_ids, c_ids, 2) == 0) {
            if (g > 0) (void)talus_npu_wait(m.ctx, cs);
            return false;
        }
    }

    m.op_inflight.store(true);
    m.active_groups = groups;
    return true;
}

bool PrefillNpu::FinishMoeShexp(void* cs, uint16_t* sh_h_dev, int64_t t_n) {
    if (!impl_ || !impl_->ctx || sh_h_dev == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    m.op_inflight.store(false);
    if (talus_npu_wait(m.ctx, cs) == 0) return false;
    m.UpdateStatsDelta(false);

    for (int64_t g = 0; g < m.active_groups; ++g) {
        const uint32_t valid_t = static_cast<uint32_t>(std::min<int64_t>(kNpuTokens, t_n - g * kNpuTokens));
        const int32_t cg = m.c_pool[static_cast<size_t>(g * kQSlicesMax + 0)];
        const int32_t cu = m.c_pool[static_cast<size_t>(g * kQSlicesMax + 1)];
        uint16_t* out_g = sh_h_dev + static_cast<size_t>(g) * kNpuTokens * kShexpCols;
        talus_npu_swiglu(
            talus_npu_device_ptr(m.ctx, cg),
            talus_npu_device_ptr(m.ctx, cu),
            out_g,
            kNpuTokens,
            valid_t,
            kNpuN,
            kShexpCols,
            cs
        );
    }
    return true;
}

void PrefillNpu::EndChunk() {
    if (!impl_) return;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    if (m.op_inflight.exchange(false) && m.ctx != nullptr) {
        (void)talus_npu_wait(m.ctx, nullptr);
    }
    if (!m.auto_tune) return;

    if (m.z_slices_cfg < 0 && m.gdn_wait_samples >= 4) {
        const double avg_wait = m.gdn_wait_us_accum / m.gdn_wait_samples;
        if (avg_wait > 350.0 && m.z_slices > 1) {
            m.z_slices -= 1;
        } else if (avg_wait < 40.0 && m.z_slices < kZSlicesMax) {
            m.z_slices += 1;
        }
        m.gdn_wait_us_accum = 0.0;
        m.gdn_wait_samples = 0;
    }
    if (m.q_slices_cfg < 0 && m.qsa_wait_samples >= 2) {
        const double avg_wait = m.qsa_wait_us_accum / m.qsa_wait_samples;
        if (avg_wait > 400.0 && m.q_slices > 1) {
            m.q_slices -= 1;
        } else if (avg_wait < 40.0 && m.q_slices < kQSlicesMax) {
            m.q_slices += 1;
        }
        m.qsa_wait_us_accum = 0.0;
        m.qsa_wait_samples = 0;
    }
    m.gdn_calls_in_chunk = 0;
    m.qsa_calls_in_chunk = 0;
}

}  // namespace talus::npu
