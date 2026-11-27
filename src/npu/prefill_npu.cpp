#include "prefill_npu.hpp"

#include <hip/hip_runtime.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
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

constexpr uint32_t kNpuMaxTokens = 4096;
constexpr uint32_t kNpuK = 2560;
constexpr uint32_t kNpuOutK = 6144;
constexpr uint32_t kNpuN = 1024;
constexpr int64_t kMaxGroups = 4;
constexpr int64_t kGpuTailThreshold = 384;
constexpr uint32_t kZCols = 6144;
constexpr uint32_t kZSlicesMax = 6;
constexpr uint32_t kQkvCols = 10240;
constexpr uint32_t kQkvSlicesMax = 10;
constexpr uint32_t kQkvAutoMax = 6;
constexpr uint32_t kQkvPoolOffset = kZSlicesMax;  // C slots [6..15] kept disjoint from GDN Z [0..5]
constexpr uint32_t kQCols = 12288;
constexpr uint32_t kQSlicesMax = 12;
constexpr uint32_t kOutCols = 2560;
constexpr uint32_t kOutSlicesMax = 2;
constexpr uint32_t kPoolSlices = 16;  // >= max(kQSlicesMax, kZSlicesMax + kQkvSlicesMax)
constexpr uint32_t kShexpCols = 640;
constexpr size_t kReadyEventPoolSize = 256;
constexpr size_t kBytesPer8 = 9;

enum class OpKind : uint8_t {
    GdnZ = 0,
    QsaQ = 1,
    ShexpGate = 2,
    ShexpUp = 3,
    GdnQkv = 4,
    OutProj = 5,
};

enum class StatOp : uint8_t {
    GdnQkv = 0,
    GdnZ = 1,
    QsaQ = 2,
    Shexp = 3,
    OutProj = 4,
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

struct GroupPlan {
    int64_t row_offset{0};
    uint32_t valid_rows{0};
    uint32_t kernel_rows{0};
    int32_t kernel_id{0};
};

struct KernelEntry {
    uint32_t m{0};
    uint32_t k{0};
    int32_t id{-1};
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

bool FileExists(const std::string& path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

int BucketForTokens(int64_t t_n) {
    if (t_n < 1536) return 0;
    if (t_n < 3584) return 1;
    return 2;
}

}  // namespace

struct PrefillNpu::Impl {
    talus_npu_ctx* ctx{nullptr};
    std::vector<hipEvent_t> ready_evs;
    size_t ready_ev_cursor{0};
    hipStream_t weight_stream{nullptr};
    int64_t min_t{512};
    int64_t max_groups{kMaxGroups};

    bool multi_kernel{false};
    bool has_k6144{false};
    uint32_t max_k{kNpuK};
    std::vector<KernelEntry> k2560_kernels;  // sorted descending by m
    std::vector<KernelEntry> k6144_kernels;  // sorted descending by m

    int qkv_slices_cfg{-1};  // -1 = auto
    int z_slices_cfg{-1};    // -1 = auto
    int q_slices_cfg{-1};    // -1 = auto
    int out_slices_cfg{-1};  // -1 = auto
    bool shexp_on{true};
    bool auto_tune{true};

    struct TuneBucket {
        uint32_t qkv_slices{2};
        uint32_t z_slices{2};
        uint32_t q_slices{4};
        uint32_t out_slices{1};
        double qkv_wait_us{0.0};
        int qkv_samples{0};
        double z_wait_us{0.0};
        int z_samples{0};
        double q_wait_us{0.0};
        int q_samples{0};
        double out_wait_us{0.0};
        int out_samples{0};
    };
    std::array<TuneBucket, 3> buckets{};

    // Pooled A and C NPU buffers sized for up to `alloc_groups` groups of 4096 tokens
    int64_t alloc_groups{0};
    std::vector<int32_t> a_bufs;   // [alloc_groups]
    std::vector<int32_t> c_pool;   // [alloc_groups * kPoolSlices]
    const uint8_t** z_ptrs_host{nullptr};
    const uint8_t** z_ptrs_dev{nullptr};
    uint16_t* w_scratch_f16{nullptr};

    const void* current_wt{nullptr};
    std::unordered_map<SliceKey, CachedWeight, SliceKeyHash> weight_cache;

    // Active in-flight tracking for current half-layer
    std::atomic<bool> op_inflight{false};
    std::vector<GroupPlan> active_plan;
    int64_t active_npu_rows{0};
    int active_bucket{2};
    uint32_t active_qkv_gpu_cols{kQkvCols};
    uint32_t active_qkv_slices{0};
    uint32_t active_z_gpu_cols{kZCols};
    uint32_t active_z_group_tokens{kNpuMaxTokens};
    uint32_t active_q_gpu_cols{kQCols};
    uint32_t active_q_slices{0};
    uint32_t active_out_gpu_cols{kOutCols};
    uint32_t active_out_slices{0};
    int64_t cached_a_layer{-1};
    int64_t cached_a_tn{0};
    int64_t cached_a_npu_rows{0};

    // Auto-tune telemetry
    double last_wait_us{0.0};
    uint64_t last_waits{0};

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

    const std::vector<KernelEntry>& KernelsForK(uint32_t k_dim) const noexcept {
        return (k_dim == kNpuOutK) ? k6144_kernels : k2560_kernels;
    }

    int32_t FindSmallestGe(uint32_t k_dim, int64_t rows, uint32_t* out_m) const noexcept {
        const auto& ks = KernelsForK(k_dim);
        int32_t best_id = -1;
        uint32_t best_m = 0;
        for (const auto& e : ks) {
            if (static_cast<int64_t>(e.m) >= rows && (best_id < 0 || e.m < best_m)) {
                best_id = e.id;
                best_m = e.m;
            }
        }
        if (best_id >= 0 && out_m) *out_m = best_m;
        return best_id;
    }

    int32_t FindLargestLe(uint32_t k_dim, int64_t rows, uint32_t* out_m) const noexcept {
        const auto& ks = KernelsForK(k_dim);
        for (const auto& e : ks) {
            if (static_cast<int64_t>(e.m) <= rows) {
                if (out_m) *out_m = e.m;
                return e.id;
            }
        }
        return -1;
    }

    bool PlanGroups(int64_t t_n, uint32_t k_dim, bool allow_gpu_tail, std::vector<GroupPlan>* out, int64_t* total_rows_out) const {
        out->clear();
        const auto& ks = KernelsForK(k_dim);
        if (ks.empty() || t_n < min_t) return false;

        int64_t rem = t_n;
        int64_t row0 = 0;
        for (int64_t g = 0; g < max_groups && rem > 0; ++g) {
            if (allow_gpu_tail && g > 0 && rem < kGpuTailThreshold) {
                break;
            }
            if (allow_gpu_tail && g == 0 && ks.size() > 1 && rem < kGpuTailThreshold) {
                break;
            }

            int32_t kid = -1;
            uint32_t km = 0;

            if (ks.size() == 1) {
                // Single-kernel legacy mode (4096 only)
                km = ks[0].m;
                kid = ks[0].id;
            } else {
                // Multi-M kernel selection: pick exact/near-fit tile to minimize wasted padding
                if (rem >= 3584) {
                    kid = FindLargestLe(k_dim, 4096, &km);
                } else if (rem >= 1536) {
                    kid = FindSmallestGe(k_dim, std::min<int64_t>(rem, 2048), &km);
                } else if (rem >= 768) {
                    kid = FindSmallestGe(k_dim, std::min<int64_t>(rem, 1024), &km);
                } else if (rem >= kGpuTailThreshold || !allow_gpu_tail) {
                    kid = FindSmallestGe(k_dim, std::min<int64_t>(rem, 512), &km);
                }
                if (kid < 0) {
                    kid = FindLargestLe(k_dim, rem, &km);
                    if (kid < 0) kid = FindSmallestGe(k_dim, rem, &km);
                }
            }
            if (kid < 0 || km == 0) break;

            // Keep every group start offset an exact multiple of group[0].kernel_rows for GDN z_streams.
            const bool last_by_stride = (g > 0 && km < (*out)[0].kernel_rows);
            const uint32_t valid = static_cast<uint32_t>(std::min<int64_t>(rem, km));
            out->push_back(GroupPlan{row0, valid, km, kid});
            row0 += valid;
            rem -= valid;

            if (last_by_stride) break;
            if (allow_gpu_tail && rem > 0 && rem < kGpuTailThreshold) break;
        }
        if (out->empty()) return false;
        if (total_rows_out) *total_rows_out = row0;
        return true;
    }

    bool EnsureCapacity(int64_t groups) {
        if (groups <= alloc_groups) return true;
        char err[256] = {};
        const size_t max_a_bytes = static_cast<size_t>(kNpuMaxTokens) * max_k / 8 * kBytesPer8;
        const size_t max_c_bytes = static_cast<size_t>(kNpuMaxTokens) * kNpuN / 8 * kBytesPer8;
        while (alloc_groups < groups) {
            int32_t a_id = talus_npu_alloc_buffer(ctx, max_a_bytes, err, sizeof(err));
            if (a_id < 0) {
                std::fprintf(stderr, "talus npu: alloc A failed: %s\n", err);
                return false;
            }
            a_bufs.push_back(a_id);
            for (uint32_t s = 0; s < kPoolSlices; ++s) {
                int32_t c_id = talus_npu_alloc_buffer(ctx, max_c_bytes, err, sizeof(err));
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
                const int32_t c_id = c_pool[static_cast<size_t>(g * kPoolSlices + s)];
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
        uint32_t k_dim,
        const WeightDesc& w
    ) {
        const void* src_ptr = w.native_data ? w.native_data : w.bf16_data;
        const int src_type = w.native_data ? w.native_type : (w.is_bf16 ? 30 : -1);
        if (!src_ptr || src_type < 0 || w.ne0 != static_cast<int64_t>(k_dim)) return -1;

        SliceKey key{layer, static_cast<uint8_t>(op), slice};
        auto it = weight_cache.find(key);
        if (it != weight_cache.end() && it->second.src_ptr == src_ptr && it->second.src_type == src_type) {
            return it->second.buf_id;
        }

        int32_t buf_id = (it != weight_cache.end()) ? it->second.buf_id : -1;
        if (buf_id < 0) {
            char err[256] = {};
            const size_t b_bytes = static_cast<size_t>(k_dim) * kNpuN / 8 * kBytesPer8;
            buf_id = talus_npu_alloc_buffer(ctx, b_bytes, err, sizeof(err));
            if (buf_id < 0) {
                std::fprintf(stderr, "talus npu: alloc weight buffer failed: %s\n", err);
                return -1;
            }
        }

        void* dst_dev = talus_npu_staged_ptr(ctx, buf_id);
        void* stream_use = weight_stream;
        if (src_type == 8 && w.native_data != nullptr) {
            talus_npu_encode_weights_q8(
                w.native_data, k_dim, row0, kNpuN, valid_cols, k_dim, 0, k_dim, dst_dev, stream_use);
            (void)hipStreamSynchronize(static_cast<hipStream_t>(stream_use));
        } else if (src_type == 1) {
            talus_npu_encode_weights_f16(
                static_cast<const uint16_t*>(src_ptr), k_dim, row0, kNpuN, valid_cols,
                k_dim, 0, k_dim, dst_dev, stream_use);
            (void)hipStreamSynchronize(static_cast<hipStream_t>(stream_use));
        } else if (src_type == 30) {
            talus_npu_encode_weights_bf16(
                static_cast<const uint16_t*>(src_ptr), k_dim, row0, kNpuN, valid_cols,
                k_dim, 0, k_dim, dst_dev, stream_use);
            (void)hipStreamSynchronize(static_cast<hipStream_t>(stream_use));
        } else {
            // Generic GGML quant fallback (Q4_K, Q6_K, IQ4_XS, etc.) via Strata dequant_f16
            if (w_scratch_f16 == nullptr) {
                if (hipMalloc(&w_scratch_f16, static_cast<size_t>(kNpuN) * max_k * sizeof(uint16_t)) != hipSuccess) {
                    return -1;
                }
            }
            strata::kernels::dequant_f16(src_type, src_ptr, row0, valid_cols, k_dim, w_scratch_f16, stream_use);
            talus_npu_encode_weights_f16(
                w_scratch_f16, k_dim, 0, kNpuN, valid_cols, k_dim, 0, k_dim, dst_dev, stream_use);
            (void)hipStreamSynchronize(static_cast<hipStream_t>(stream_use));
        }

        talus_npu_finalize_weights(ctx, buf_id);
        weight_cache[key] = CachedWeight{buf_id, src_ptr, src_type};
        return buf_id;
    }

    void UpdateStatsDelta(StatOp op) {
        talus_npu_stats_t st{};
        talus_npu_get_stats(ctx, &st);
        const double d_wait = std::max(0.0, st.host_wait_us_total - last_wait_us);
        const uint64_t d_cnt = st.waits >= last_waits ? (st.waits - last_waits) : 0;
        last_wait_us = st.host_wait_us_total;
        last_waits = st.waits;
        if (d_cnt == 0) return;
        const double per_wait = d_wait / static_cast<double>(d_cnt);
        TuneBucket& b = buckets[static_cast<size_t>(active_bucket)];
        switch (op) {
            case StatOp::GdnQkv:
                b.qkv_wait_us += per_wait;
                b.qkv_samples += 1;
                break;
            case StatOp::GdnZ:
                b.z_wait_us += per_wait;
                b.z_samples += 1;
                break;
            case StatOp::QsaQ:
                b.q_wait_us += per_wait;
                b.q_samples += 1;
                break;
            case StatOp::OutProj:
                b.out_wait_us += per_wait;
                b.out_samples += 1;
                break;
            case StatOp::Shexp:
                break;
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
    const bool explicit_xclbin = (x_env && x_env[0]) || (i_env && i_env[0]);

    const std::string multi_xclbin = dir.empty() ? std::string() : (dir + "/talus_npu_aie2p.xclbin");
    const std::string multi_inst_4096 = dir.empty() ? std::string() : (dir + "/insts_4096x2560x1024.bin");
    const bool use_multi = !explicit_xclbin && !dir.empty() && FileExists(multi_xclbin) && FileExists(multi_inst_4096);

    std::string xclbin;
    std::string insts;
    std::string kname = "MLIR_AIE";
    if (use_multi) {
        xclbin = multi_xclbin;
        insts = multi_inst_4096;
        kname = "GEMM_4096_2560_1024";
    } else {
        xclbin = (x_env && x_env[0]) ? std::string(x_env) : (dir + "/final_4096x2560x1024_128x64x128.xclbin");
        insts = (i_env && i_env[0]) ? std::string(i_env) : (dir + "/insts_4096x2560x1024_128x64x128.txt");
    }

    talus_npu_config_t cfg{};
    talus_npu_config_default(&cfg);
    cfg.xclbin_path = xclbin.c_str();
    cfg.insts_path = insts.c_str();
    cfg.kernel_name = kname.c_str();
    cfg.tokens = kNpuMaxTokens;
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
    m.multi_kernel = use_multi;
    m.k2560_kernels.push_back(KernelEntry{kNpuMaxTokens, kNpuK, 0});

    if (use_multi) {
        constexpr std::array<uint32_t, 3> kExtraM2560 = {2048, 1024, 512};
        for (uint32_t mm : kExtraM2560) {
            char kbuf[64], ibuf[512], ebuf[256] = {};
            std::snprintf(kbuf, sizeof(kbuf), "GEMM_%u_2560_1024", mm);
            std::snprintf(ibuf, sizeof(ibuf), "%s/insts_%ux2560x1024.bin", dir.c_str(), mm);
            if (FileExists(ibuf)) {
                int32_t kid = talus_npu_register_kernel(ctx, kbuf, ibuf, mm, kNpuK, kNpuN, ebuf, sizeof(ebuf));
                if (kid >= 0) {
                    m.k2560_kernels.push_back(KernelEntry{mm, kNpuK, kid});
                } else {
                    std::fprintf(stderr, "talus npu: skipping %s (%s)\n", kbuf, ebuf);
                }
            }
        }
        constexpr std::array<uint32_t, 4> kM6144 = {4096, 2048, 1024, 512};
        for (uint32_t mm : kM6144) {
            char kbuf[64], ibuf[512], ebuf[256] = {};
            std::snprintf(kbuf, sizeof(kbuf), "GEMM_%u_6144_1024", mm);
            std::snprintf(ibuf, sizeof(ibuf), "%s/insts_%ux6144x1024.bin", dir.c_str(), mm);
            if (FileExists(ibuf)) {
                int32_t kid = talus_npu_register_kernel(ctx, kbuf, ibuf, mm, kNpuOutK, kNpuN, ebuf, sizeof(ebuf));
                if (kid >= 0) {
                    m.k6144_kernels.push_back(KernelEntry{mm, kNpuOutK, kid});
                    m.has_k6144 = true;
                    m.max_k = kNpuOutK;
                } else {
                    std::fprintf(stderr, "talus npu: skipping %s (%s)\n", kbuf, ebuf);
                }
            }
        }
    }

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

    const int default_min_t = (m.k2560_kernels.size() > 1) ? 1536 : 2048;
    m.min_t = std::max(128, EnvInt("TALUS_NPU_MIN_T", default_min_t));
    m.max_groups = std::clamp(EnvInt("TALUS_NPU_MAX_GROUPS", static_cast<int>(kMaxGroups)), 1, static_cast<int>(kMaxGroups));
    m.qkv_slices_cfg = EnvInt("TALUS_NPU_QKV_SLICES", -1);
    m.z_slices_cfg = EnvInt("TALUS_NPU_Z_SLICES", -1);
    m.q_slices_cfg = EnvInt("TALUS_NPU_Q_SLICES", -1);
    m.out_slices_cfg = EnvInt("TALUS_NPU_OUT_SLICES", 0);
    m.shexp_on = EnvInt("TALUS_NPU_SHEXP", 1) != 0;
    m.auto_tune = EnvInt("TALUS_NPU_AUTO_TUNE", 1) != 0;

    for (auto& b : m.buckets) {
        if (m.qkv_slices_cfg >= 0) {
            b.qkv_slices = static_cast<uint32_t>(std::clamp(m.qkv_slices_cfg, 0, static_cast<int>(kQkvSlicesMax)));
        }
        if (m.z_slices_cfg >= 0) {
            b.z_slices = static_cast<uint32_t>(std::clamp(m.z_slices_cfg, 0, static_cast<int>(kZSlicesMax)));
        }
        if (m.q_slices_cfg >= 0) {
            b.q_slices = static_cast<uint32_t>(std::clamp(m.q_slices_cfg, 0, static_cast<int>(kQSlicesMax)));
        }
        if (!m.has_k6144 || m.out_slices_cfg <= 0) {
            b.out_slices = 0;
        } else {
            b.out_slices = static_cast<uint32_t>(std::clamp(m.out_slices_cfg, 0, static_cast<int>(kOutSlicesMax)));
        }
    }

    char safety[256] = {};
    (void)talus_npu_check_env_safety(safety, sizeof(safety));
    std::fprintf(
        stderr,
        "talus npu: enabled (%s, kernels=%zu+%zu, min_T=%lld, qkv_slices=%u%s, z_slices=%u%s, q_slices=%u%s, out_slices=%u%s, shexp=%s, %s)\n",
        use_multi ? "multi-M 512..4096x{2560,6144}x1024" : "4096x2560x1024",
        m.k2560_kernels.size(), m.k6144_kernels.size(),
        static_cast<long long>(m.min_t),
        m.buckets[2].qkv_slices, m.qkv_slices_cfg < 0 && m.auto_tune ? " auto" : "",
        m.buckets[2].z_slices, m.z_slices_cfg < 0 && m.auto_tune ? " auto" : "",
        m.buckets[2].q_slices, m.q_slices_cfg < 0 && m.auto_tune ? " auto" : "",
        m.buckets[2].out_slices, m.has_k6144 && m.out_slices_cfg < 0 && m.auto_tune ? " auto" : "",
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
    return t_n <= impl_->max_groups * static_cast<int64_t>(kNpuMaxTokens);
}

void PrefillNpu::SetContext(const void* wt_ptr) {
    if (!impl_) return;
    const std::lock_guard lock(impl_->mutex);
    if (impl_->current_wt != wt_ptr) {
        impl_->current_wt = wt_ptr;
    }
    impl_->cached_a_layer = -1;
    impl_->cached_a_tn = 0;
    impl_->cached_a_npu_rows = 0;
}

void* PrefillNpu::RawCtx() const noexcept {
    return impl_ ? impl_->ctx : nullptr;
}

void PrefillNpu::WaitCallback(void* arg) {
    auto* self = static_cast<PrefillNpu*>(arg);
    (void)self;
}

bool PrefillNpu::StartGdnQkv(
    void* cs,
    strata::prefill::Gemm* gemm,
    int64_t layer,
    const WeightDesc& wqkv,
    const uint16_t* mixed_h,
    int64_t t_n,
    int64_t* qkv_gpu_cols_out,
    int64_t* npu_rows_out
) {
    if (qkv_gpu_cols_out) *qkv_gpu_cols_out = kQkvCols;
    if (npu_rows_out) *npu_rows_out = 0;
    if (!Eligible(t_n) || wqkv.ne0 != kNpuK || wqkv.ne1 != kQkvCols || mixed_h == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    if (m.op_inflight.exchange(false)) {
        (void)talus_npu_wait(m.ctx, cs);
    }
    m.cached_a_layer = -1;
    m.cached_a_tn = 0;
    m.cached_a_npu_rows = 0;

    const int bucket = BucketForTokens(t_n);
    const uint32_t slices = m.buckets[static_cast<size_t>(bucket)].qkv_slices;
    if (slices == 0) return false;

    std::vector<GroupPlan> plan;
    int64_t npu_rows = 0;
    if (!m.PlanGroups(t_n, kNpuK, npu_rows_out != nullptr, &plan, &npu_rows)) return false;
    if (!m.EnsureCapacity(static_cast<int64_t>(plan.size()))) return false;

    const uint32_t gpu_cols = kQkvCols - slices * kNpuN;
    const uint32_t slice0 = kQkvSlicesMax - slices;

    std::vector<int32_t> b_ids(slices);
    for (uint32_t s = 0; s < slices; ++s) {
        const uint32_t row0 = gpu_cols + s * kNpuN;
        const int32_t bid = m.EnsureWeightSlice(cs, gemm, layer, OpKind::GdnQkv, slice0 + s, row0, kNpuN, kNpuK, wqkv);
        if (bid < 0) return false;
        b_ids[s] = bid;
    }

    void* stage = talus_npu_stage_stream(m.ctx);
    hipEvent_t ready_ev = m.NextReadyEvent();
    if (hipEventRecord(ready_ev, static_cast<hipStream_t>(cs)) != hipSuccess) return false;
    if (hipStreamWaitEvent(static_cast<hipStream_t>(stage), ready_ev, 0) != hipSuccess) return false;

    for (size_t g = 0; g < plan.size(); ++g) {
        const GroupPlan& gp = plan[g];
        const int32_t a_id = m.a_bufs[g];
        const uint16_t* x_g = mixed_h + static_cast<size_t>(gp.row_offset) * kNpuK;
        talus_npu_encode_activations_f16(
            x_g, kNpuK, gp.kernel_rows, gp.valid_rows, kNpuK, 0, kNpuK, talus_npu_device_ptr(m.ctx, a_id), stage);

        std::vector<int32_t> c_ids(slices);
        for (uint32_t s = 0; s < slices; ++s) {
            c_ids[s] = m.c_pool[g * kPoolSlices + kQkvPoolOffset + s];
        }
        if (talus_npu_submit_kernel(m.ctx, gp.kernel_id, stage, a_id, b_ids.data(), c_ids.data(), static_cast<int32_t>(slices)) == 0) {
            if (g > 0) (void)talus_npu_wait(m.ctx, cs);
            return false;
        }
    }

    m.op_inflight.store(true);
    m.active_plan = std::move(plan);
    m.active_npu_rows = npu_rows;
    m.active_bucket = bucket;
    m.active_qkv_gpu_cols = gpu_cols;
    m.active_qkv_slices = slices;
    m.cached_a_layer = layer;
    m.cached_a_tn = t_n;
    m.cached_a_npu_rows = npu_rows;
    if (qkv_gpu_cols_out) *qkv_gpu_cols_out = static_cast<int64_t>(gpu_cols);
    if (npu_rows_out) *npu_rows_out = npu_rows;
    return true;
}

bool PrefillNpu::FinishGdnQkv(void* cs, float* qkv_dev, [[maybe_unused]] int64_t t_n) {
    if (!impl_ || !impl_->ctx || qkv_dev == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    m.op_inflight.store(false);
    if (talus_npu_wait(m.ctx, cs) == 0) return false;
    m.UpdateStatsDelta(StatOp::GdnQkv);

    for (size_t g = 0; g < m.active_plan.size(); ++g) {
        const GroupPlan& gp = m.active_plan[g];
        float* dst_g = qkv_dev + static_cast<size_t>(gp.row_offset) * kQkvCols;
        for (uint32_t s = 0; s < m.active_qkv_slices; ++s) {
            const int32_t c_id = m.c_pool[g * kPoolSlices + kQkvPoolOffset + s];
            const uint32_t col0 = m.active_qkv_gpu_cols + s * kNpuN;
            talus_npu_decode_output(
                talus_npu_device_ptr(m.ctx, c_id), dst_g, gp.kernel_rows, gp.valid_rows, kNpuN,
                kQkvCols, col0, 0, kNpuN, cs);
        }
    }
    return true;
}

bool PrefillNpu::StartGdnZ(
    void* cs,
    strata::prefill::Gemm* gemm,
    int64_t layer,
    const WeightDesc& wg,
    const uint16_t* mixed_h,
    int64_t t_n,
    int64_t* z_gpu_cols_out,
    int64_t* npu_rows_out
) {
    if (z_gpu_cols_out) *z_gpu_cols_out = kZCols;
    if (npu_rows_out) *npu_rows_out = 0;
    if (!Eligible(t_n) || wg.ne0 != kNpuK || wg.ne1 != kZCols || mixed_h == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    if (m.op_inflight.exchange(false)) {
        (void)talus_npu_wait(m.ctx, cs);
    }

    const int bucket = BucketForTokens(t_n);
    const uint32_t slices = m.buckets[static_cast<size_t>(bucket)].z_slices;
    if (slices == 0) {
        m.cached_a_layer = -1;
        m.cached_a_tn = 0;
        m.cached_a_npu_rows = 0;
        return false;
    }

    std::vector<GroupPlan> plan;
    int64_t npu_rows = 0;
    if (!m.PlanGroups(t_n, kNpuK, npu_rows_out != nullptr, &plan, &npu_rows)) return false;
    if (!m.EnsureCapacity(static_cast<int64_t>(plan.size()))) return false;

    const bool reuse_a = (m.cached_a_layer == layer && m.cached_a_tn == t_n && m.cached_a_npu_rows == npu_rows);
    m.cached_a_layer = -1;
    m.cached_a_tn = 0;
    m.cached_a_npu_rows = 0;

    const uint32_t gpu_cols = kZCols - slices * kNpuN;
    const uint32_t slice0 = kZSlicesMax - slices;

    std::vector<int32_t> b_ids(slices);
    for (uint32_t s = 0; s < slices; ++s) {
        const uint32_t row0 = gpu_cols + s * kNpuN;
        const int32_t bid = m.EnsureWeightSlice(cs, gemm, layer, OpKind::GdnZ, slice0 + s, row0, kNpuN, kNpuK, wg);
        if (bid < 0) return false;
        b_ids[s] = bid;
    }

    void* stage = nullptr;
    if (!reuse_a) {
        stage = talus_npu_stage_stream(m.ctx);
        hipEvent_t ready_ev = m.NextReadyEvent();
        if (hipEventRecord(ready_ev, static_cast<hipStream_t>(cs)) != hipSuccess) return false;
        if (hipStreamWaitEvent(static_cast<hipStream_t>(stage), ready_ev, 0) != hipSuccess) return false;
    }

    for (size_t g = 0; g < plan.size(); ++g) {
        const GroupPlan& gp = plan[g];
        const int32_t a_id = m.a_bufs[g];
        if (!reuse_a) {
            const uint16_t* x_g = mixed_h + static_cast<size_t>(gp.row_offset) * kNpuK;
            talus_npu_encode_activations_f16(
                x_g, kNpuK, gp.kernel_rows, gp.valid_rows, kNpuK, 0, kNpuK, talus_npu_device_ptr(m.ctx, a_id), stage);
        }

        std::vector<int32_t> c_ids(slices);
        for (uint32_t s = 0; s < slices; ++s) {
            c_ids[s] = m.c_pool[g * kPoolSlices + s];
        }
        if (talus_npu_submit_kernel(m.ctx, gp.kernel_id, stage, a_id, b_ids.data(), c_ids.data(), static_cast<int32_t>(slices)) == 0) {
            if (g > 0) (void)talus_npu_wait(m.ctx, cs);
            return false;
        }
    }

    m.op_inflight.store(true);
    m.active_z_group_tokens = plan[0].kernel_rows;
    m.active_plan = std::move(plan);
    m.active_npu_rows = npu_rows;
    m.active_bucket = bucket;
    m.active_z_gpu_cols = gpu_cols;
    if (z_gpu_cols_out) *z_gpu_cols_out = static_cast<int64_t>(gpu_cols);
    if (npu_rows_out) *npu_rows_out = npu_rows;
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
    uint32_t z_tokens = kNpuMaxTokens;
    uint32_t z_npu_valid = static_cast<uint32_t>(t);
    {
        const std::lock_guard lock(impl_->mutex);
        z_streams = impl_->z_ptrs_dev;
        z_gpu_cols = impl_->active_z_gpu_cols;
        z_tokens = impl_->active_z_group_tokens;
        z_npu_valid = static_cast<uint32_t>(std::min<int64_t>(t, impl_->active_npu_rows));
    }
    strata::prefill::gdn_recurrence_npu(
        state, h, gate, beta, z, gamma, eps, y, y16, t, stream, ld16,
        cb, &arg, z_streams, kZSlicesMax, z_tokens, z_gpu_cols, z_npu_valid);
}

void PrefillNpu::WaitGdnZ(void* cs) {
    if (!impl_ || !impl_->ctx) return;
    talus_npu_wait(impl_->ctx, cs);
    const std::lock_guard lock(impl_->mutex);
    impl_->op_inflight.store(false);
    impl_->UpdateStatsDelta(StatOp::GdnZ);
}

bool PrefillNpu::StartQsaQ(
    void* cs,
    strata::prefill::Gemm* gemm,
    int64_t layer,
    const WeightDesc& wq,
    const uint16_t* mixed_h,
    int64_t t_n,
    int64_t* q_gpu_cols_out,
    int64_t* npu_rows_out
) {
    if (q_gpu_cols_out) *q_gpu_cols_out = kQCols;
    if (npu_rows_out) *npu_rows_out = 0;
    if (!Eligible(t_n) || wq.ne0 != kNpuK || wq.ne1 != kQCols || mixed_h == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    if (m.op_inflight.exchange(false)) {
        (void)talus_npu_wait(m.ctx, cs);
    }
    m.cached_a_layer = -1;
    m.cached_a_tn = 0;
    m.cached_a_npu_rows = 0;

    const int bucket = BucketForTokens(t_n);
    const uint32_t slices = m.buckets[static_cast<size_t>(bucket)].q_slices;
    if (slices == 0) return false;

    std::vector<GroupPlan> plan;
    int64_t npu_rows = 0;
    if (!m.PlanGroups(t_n, kNpuK, npu_rows_out != nullptr, &plan, &npu_rows)) return false;
    if (!m.EnsureCapacity(static_cast<int64_t>(plan.size()))) return false;

    const uint32_t gpu_cols = kQCols - slices * kNpuN;
    const uint32_t slice0 = kQSlicesMax - slices;

    std::vector<int32_t> b_ids(slices);
    for (uint32_t s = 0; s < slices; ++s) {
        const uint32_t row0 = gpu_cols + s * kNpuN;
        const int32_t bid = m.EnsureWeightSlice(cs, gemm, layer, OpKind::QsaQ, slice0 + s, row0, kNpuN, kNpuK, wq);
        if (bid < 0) return false;
        b_ids[s] = bid;
    }

    void* stage = talus_npu_stage_stream(m.ctx);
    hipEvent_t ready_ev = m.NextReadyEvent();
    if (hipEventRecord(ready_ev, static_cast<hipStream_t>(cs)) != hipSuccess) return false;
    if (hipStreamWaitEvent(static_cast<hipStream_t>(stage), ready_ev, 0) != hipSuccess) return false;

    for (size_t g = 0; g < plan.size(); ++g) {
        const GroupPlan& gp = plan[g];
        const int32_t a_id = m.a_bufs[g];
        const uint16_t* x_g = mixed_h + static_cast<size_t>(gp.row_offset) * kNpuK;
        talus_npu_encode_activations_f16(
            x_g, kNpuK, gp.kernel_rows, gp.valid_rows, kNpuK, 0, kNpuK, talus_npu_device_ptr(m.ctx, a_id), stage);

        std::vector<int32_t> c_ids(slices);
        for (uint32_t s = 0; s < slices; ++s) {
            c_ids[s] = m.c_pool[g * kPoolSlices + s];
        }
        if (talus_npu_submit_kernel(m.ctx, gp.kernel_id, stage, a_id, b_ids.data(), c_ids.data(), static_cast<int32_t>(slices)) == 0) {
            if (g > 0) (void)talus_npu_wait(m.ctx, cs);
            return false;
        }
    }

    m.op_inflight.store(true);
    m.active_plan = std::move(plan);
    m.active_npu_rows = npu_rows;
    m.active_bucket = bucket;
    m.active_q_gpu_cols = gpu_cols;
    m.active_q_slices = slices;
    if (q_gpu_cols_out) *q_gpu_cols_out = static_cast<int64_t>(gpu_cols);
    if (npu_rows_out) *npu_rows_out = npu_rows;
    return true;
}

bool PrefillNpu::FinishQsaQ(void* cs, float* qf_dev, [[maybe_unused]] int64_t t_n) {
    if (!impl_ || !impl_->ctx || qf_dev == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    m.op_inflight.store(false);
    if (talus_npu_wait(m.ctx, cs) == 0) return false;
    m.UpdateStatsDelta(StatOp::QsaQ);

    for (size_t g = 0; g < m.active_plan.size(); ++g) {
        const GroupPlan& gp = m.active_plan[g];
        float* dst_g = qf_dev + static_cast<size_t>(gp.row_offset) * kQCols;
        for (uint32_t s = 0; s < m.active_q_slices; ++s) {
            const int32_t c_id = m.c_pool[g * kPoolSlices + s];
            const uint32_t col0 = m.active_q_gpu_cols + s * kNpuN;
            talus_npu_decode_output(
                talus_npu_device_ptr(m.ctx, c_id), dst_g, gp.kernel_rows, gp.valid_rows, kNpuN,
                kQCols, col0, 0, kNpuN, cs);
        }
    }
    return true;
}

bool PrefillNpu::StartOutProj(
    void* cs,
    strata::prefill::Gemm* gemm,
    int64_t layer,
    const WeightDesc& wo,
    const uint16_t* act_h,
    int64_t t_n,
    int64_t ldx,
    int64_t* out_gpu_cols_out,
    int64_t* npu_rows_out
) {
    if (out_gpu_cols_out) *out_gpu_cols_out = kOutCols;
    if (npu_rows_out) *npu_rows_out = 0;
    if (!impl_ || !impl_->has_k6144 || !Eligible(t_n) || wo.ne0 != kNpuOutK || wo.ne1 != kOutCols || act_h == nullptr) {
        return false;
    }
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    if (m.op_inflight.exchange(false)) {
        (void)talus_npu_wait(m.ctx, cs);
    }
    m.cached_a_layer = -1;
    m.cached_a_tn = 0;
    m.cached_a_npu_rows = 0;

    const int bucket = BucketForTokens(t_n);
    const uint32_t slices = m.buckets[static_cast<size_t>(bucket)].out_slices;
    if (slices == 0) return false;

    std::vector<GroupPlan> plan;
    int64_t npu_rows = 0;
    if (!m.PlanGroups(t_n, kNpuOutK, npu_rows_out != nullptr, &plan, &npu_rows)) return false;
    if (ldx > 0 && ldx != static_cast<int64_t>(kNpuOutK) && npu_rows < t_n && (t_n - npu_rows) < 64) {
        // Gemm::native padded-X path requires T >= 64 for the GPU tail slice; cover all rows on NPU instead.
        if (!m.PlanGroups(t_n, kNpuOutK, false, &plan, &npu_rows)) return false;
    }
    if (!m.EnsureCapacity(static_cast<int64_t>(plan.size()))) return false;

    const uint32_t gpu_cols = kOutCols - slices * kNpuN;
    const uint32_t slice0 = kOutSlicesMax - slices;
    const size_t stride_x = (ldx > 0) ? static_cast<size_t>(ldx) : static_cast<size_t>(kNpuOutK);

    std::vector<int32_t> b_ids(slices);
    for (uint32_t s = 0; s < slices; ++s) {
        const uint32_t row0 = gpu_cols + s * kNpuN;
        const int32_t bid = m.EnsureWeightSlice(cs, gemm, layer, OpKind::OutProj, slice0 + s, row0, kNpuN, kNpuOutK, wo);
        if (bid < 0) return false;
        b_ids[s] = bid;
    }

    void* stage = talus_npu_stage_stream(m.ctx);
    hipEvent_t ready_ev = m.NextReadyEvent();
    if (hipEventRecord(ready_ev, static_cast<hipStream_t>(cs)) != hipSuccess) return false;
    if (hipStreamWaitEvent(static_cast<hipStream_t>(stage), ready_ev, 0) != hipSuccess) return false;

    for (size_t g = 0; g < plan.size(); ++g) {
        const GroupPlan& gp = plan[g];
        const int32_t a_id = m.a_bufs[g];
        const uint16_t* x_g = act_h + static_cast<size_t>(gp.row_offset) * stride_x;
        talus_npu_encode_activations_f16(
            x_g, stride_x, gp.kernel_rows, gp.valid_rows, kNpuOutK, 0, kNpuOutK,
            talus_npu_device_ptr(m.ctx, a_id), stage);

        std::vector<int32_t> c_ids(slices);
        for (uint32_t s = 0; s < slices; ++s) {
            c_ids[s] = m.c_pool[g * kPoolSlices + s];
        }
        if (talus_npu_submit_kernel(m.ctx, gp.kernel_id, stage, a_id, b_ids.data(), c_ids.data(), static_cast<int32_t>(slices)) == 0) {
            if (g > 0) (void)talus_npu_wait(m.ctx, cs);
            return false;
        }
    }

    m.op_inflight.store(true);
    m.active_plan = std::move(plan);
    m.active_npu_rows = npu_rows;
    m.active_bucket = bucket;
    m.active_out_gpu_cols = gpu_cols;
    m.active_out_slices = slices;
    if (out_gpu_cols_out) *out_gpu_cols_out = static_cast<int64_t>(gpu_cols);
    if (npu_rows_out) *npu_rows_out = npu_rows;
    return true;
}

bool PrefillNpu::FinishOutProj(void* cs, float* bo_dev, [[maybe_unused]] int64_t t_n) {
    if (!impl_ || !impl_->ctx || bo_dev == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    m.op_inflight.store(false);
    if (talus_npu_wait(m.ctx, cs) == 0) return false;
    m.UpdateStatsDelta(StatOp::OutProj);

    for (size_t g = 0; g < m.active_plan.size(); ++g) {
        const GroupPlan& gp = m.active_plan[g];
        float* dst_g = bo_dev + static_cast<size_t>(gp.row_offset) * kOutCols;
        for (uint32_t s = 0; s < m.active_out_slices; ++s) {
            const int32_t c_id = m.c_pool[g * kPoolSlices + s];
            const uint32_t col0 = m.active_out_gpu_cols + s * kNpuN;
            talus_npu_decode_output(
                talus_npu_device_ptr(m.ctx, c_id), dst_g, gp.kernel_rows, gp.valid_rows, kNpuN,
                kOutCols, col0, 0, kNpuN, cs);
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
    int64_t t_n,
    int64_t* npu_rows_out
) {
    if (npu_rows_out) *npu_rows_out = 0;
    if (!Eligible(t_n) || !impl_->shexp_on || mixed_h == nullptr) return false;
    if (wsg.ne0 != kNpuK || wsu.ne0 != kNpuK || wsg.ne1 != kShexpCols || wsu.ne1 != kShexpCols) {
        return false;
    }
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    if (m.op_inflight.exchange(false)) {
        (void)talus_npu_wait(m.ctx, cs);
    }
    m.cached_a_layer = -1;
    m.cached_a_tn = 0;
    m.cached_a_npu_rows = 0;

    std::vector<GroupPlan> plan;
    int64_t npu_rows = 0;
    if (!m.PlanGroups(t_n, kNpuK, npu_rows_out != nullptr, &plan, &npu_rows)) return false;
    if (!m.EnsureCapacity(static_cast<int64_t>(plan.size()))) return false;

    const int32_t bg = m.EnsureWeightSlice(cs, gemm, layer, OpKind::ShexpGate, 0, 0, kShexpCols, kNpuK, wsg);
    const int32_t bu = m.EnsureWeightSlice(cs, gemm, layer, OpKind::ShexpUp, 0, 0, kShexpCols, kNpuK, wsu);
    if (bg < 0 || bu < 0) return false;

    void* stage = talus_npu_stage_stream(m.ctx);
    hipEvent_t ready_ev = m.NextReadyEvent();
    if (hipEventRecord(ready_ev, static_cast<hipStream_t>(cs)) != hipSuccess) return false;
    if (hipStreamWaitEvent(static_cast<hipStream_t>(stage), ready_ev, 0) != hipSuccess) return false;

    const int32_t b_ids[2] = {bg, bu};
    for (size_t g = 0; g < plan.size(); ++g) {
        const GroupPlan& gp = plan[g];
        const int32_t a_id = m.a_bufs[g];
        const uint16_t* x_g = mixed_h + static_cast<size_t>(gp.row_offset) * kNpuK;
        talus_npu_encode_activations_f16(
            x_g, kNpuK, gp.kernel_rows, gp.valid_rows, kNpuK, 0, kNpuK, talus_npu_device_ptr(m.ctx, a_id), stage);

        const int32_t c_ids[2] = {
            m.c_pool[g * kPoolSlices + 0],
            m.c_pool[g * kPoolSlices + 1],
        };
        if (talus_npu_submit_kernel(m.ctx, gp.kernel_id, stage, a_id, b_ids, c_ids, 2) == 0) {
            if (g > 0) (void)talus_npu_wait(m.ctx, cs);
            return false;
        }
    }

    m.op_inflight.store(true);
    m.active_plan = std::move(plan);
    m.active_npu_rows = npu_rows;
    m.active_bucket = BucketForTokens(t_n);
    if (npu_rows_out) *npu_rows_out = npu_rows;
    return true;
}

bool PrefillNpu::FinishMoeShexp(void* cs, uint16_t* sh_h_dev, [[maybe_unused]] int64_t t_n) {
    if (!impl_ || !impl_->ctx || sh_h_dev == nullptr) return false;
    const std::lock_guard lock(impl_->mutex);
    Impl& m = *impl_;
    m.op_inflight.store(false);
    if (talus_npu_wait(m.ctx, cs) == 0) return false;
    m.UpdateStatsDelta(StatOp::Shexp);

    for (size_t g = 0; g < m.active_plan.size(); ++g) {
        const GroupPlan& gp = m.active_plan[g];
        const int32_t cg = m.c_pool[g * kPoolSlices + 0];
        const int32_t cu = m.c_pool[g * kPoolSlices + 1];
        uint16_t* out_g = sh_h_dev + static_cast<size_t>(gp.row_offset) * kShexpCols;
        talus_npu_swiglu(
            talus_npu_device_ptr(m.ctx, cg),
            talus_npu_device_ptr(m.ctx, cu),
            out_g,
            gp.kernel_rows,
            gp.valid_rows,
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
    m.cached_a_layer = -1;
    m.cached_a_tn = 0;
    m.cached_a_npu_rows = 0;
    if (m.op_inflight.exchange(false) && m.ctx != nullptr) {
        (void)talus_npu_wait(m.ctx, nullptr);
    }
    if (!m.auto_tune) return;

    Impl::TuneBucket& b = m.buckets[static_cast<size_t>(m.active_bucket)];
    if (m.qkv_slices_cfg < 0 && b.qkv_samples >= 4) {
        const double avg_wait = b.qkv_wait_us / b.qkv_samples;
        if (avg_wait > 300.0 && b.qkv_slices > 0) {
            b.qkv_slices -= 1;
        } else if (avg_wait < 40.0 && b.qkv_slices < kQkvAutoMax) {
            b.qkv_slices += 1;
        }
        b.qkv_wait_us = 0.0;
        b.qkv_samples = 0;
    }
    if (m.z_slices_cfg < 0 && b.z_samples >= 4) {
        const double avg_wait = b.z_wait_us / b.z_samples;
        if (avg_wait > 350.0 && b.z_slices > 1) {
            b.z_slices -= 1;
        } else if (avg_wait < 40.0 && b.z_slices < kZSlicesMax) {
            b.z_slices += 1;
        }
        b.z_wait_us = 0.0;
        b.z_samples = 0;
    }
    if (m.q_slices_cfg < 0 && b.q_samples >= 2) {
        const double avg_wait = b.q_wait_us / b.q_samples;
        if (avg_wait > 400.0 && b.q_slices > 1) {
            b.q_slices -= 1;
        } else if (avg_wait < 40.0 && b.q_slices < kQSlicesMax) {
            b.q_slices += 1;
        }
        b.q_wait_us = 0.0;
        b.q_samples = 0;
    }
    if (m.has_k6144 && m.out_slices_cfg < 0 && b.out_samples >= 4) {
        const double avg_wait = b.out_wait_us / b.out_samples;
        if (avg_wait > 300.0 && b.out_slices > 0) {
            b.out_slices -= 1;
        } else if (avg_wait < 40.0 && b.out_slices < kOutSlicesMax) {
            b.out_slices += 1;
        }
        b.out_wait_us = 0.0;
        b.out_samples = 0;
    }
}

}  // namespace talus::npu
