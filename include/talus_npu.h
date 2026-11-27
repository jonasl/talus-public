/*
 * include/talus_npu.h - Talus XDNA2 NPU C ABI for hybrid GPU+NPU inference on AMD Strix Halo.
 *
 * SPDX-License-Identifier: MIT
 *
 * Zero build-time dependency on XRT headers: `libxrt_coreutil.so.2` is loaded dynamically at
 * runtime via `dlopen`. On machines without XRT or `/dev/accel/accel0`, `talus_npu_create`
 * returns a non-zero status with a diagnostic error message and GPU execution continues unchanged.
 */

#ifndef TALUS_NPU_H
#define TALUS_NPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct talus_npu_ctx talus_npu_ctx;

typedef struct talus_npu_config {
    const char* xclbin_path;
    const char* insts_path;
    uint32_t tokens;        /**< M tile rows per NPU invocation (e.g. 4096). */
    uint32_t k;             /**< K inner dimension (e.g. 2560). */
    uint32_t n;             /**< N output columns per slice (e.g. 1024). */
    uint32_t timeout_ms;    /**< Per-run wait timeout in ms (0 = default 8000). */
    int32_t isolate;        /**< 0 = zero-copy hipHostMalloc+XRT shared BO, 1 = private BO + copy. */
    int32_t keepalive;      /**< 1 = 1s heartbeat prevents D3hot wakeups, 0 = off. */
    int32_t allow_unheld_fclk; /**< 1 = allow auto pp_dpm_fclk (AMD PI 1.0.0.2c+), 0 = require held fclk. */
    const char* kernel_name; /**< Optional kernel name in xclbin (NULL = "MLIR_AIE"). */
} talus_npu_config_t;

typedef struct talus_npu_stats {
    uint64_t submits;
    uint64_t waits;
    uint64_t slices_run;
    double encode_us_total;
    double npu_wait_us_total;
    double host_wait_us_total;
    double copy_us_total;
} talus_npu_stats_t;

/** Populate `out` with default settings (`tokens=4096, k=2560, n=1024`, env overrides applied). */
void talus_npu_config_default(talus_npu_config_t* out);

/**
 * Create an NPU GEMM context, load the `.xclbin` and instruction stream, and run a startup
 * self-check heartbeat. Returns 0 on success; on failure returns non-zero and writes a
 * human-readable message to `err` (if non-null).
 */
int32_t talus_npu_create(const talus_npu_config_t* cfg, talus_npu_ctx** out_ctx, char* err, size_t err_cap);

/** Destroy an NPU context and release XRT / HIP buffers. Safe on NULL. */
void talus_npu_destroy(talus_npu_ctx* ctx);

/** Non-zero while the NPU context has not suffered a timeout or fatal error. */
int32_t talus_npu_healthy(const talus_npu_ctx* ctx);

uint32_t talus_npu_tokens(const talus_npu_ctx* ctx);
uint32_t talus_npu_k(const talus_npu_ctx* ctx);
uint32_t talus_npu_n(const talus_npu_ctx* ctx);
size_t talus_npu_a_bytes(const talus_npu_ctx* ctx);
size_t talus_npu_b_bytes(const talus_npu_ctx* ctx);
size_t talus_npu_c_bytes(const talus_npu_ctx* ctx);

/**
 * Allocate a pinned GPU-visible + NPU-mapped buffer of `bytes`.
 * Returns buffer id >= 0 on success, or -1 on error.
 */
int32_t talus_npu_alloc_buffer(talus_npu_ctx* ctx, size_t bytes, char* err, size_t err_cap);

void* talus_npu_host_ptr(const talus_npu_ctx* ctx, int32_t buf_id);
void* talus_npu_device_ptr(const talus_npu_ctx* ctx, int32_t buf_id);
void* talus_npu_staged_ptr(const talus_npu_ctx* ctx, int32_t buf_id);

/** Finalize a weight buffer after GPU encoding (syncs to NPU / copies in isolate mode). */
void talus_npu_finalize_weights(talus_npu_ctx* ctx, int32_t buf_id);

/** Dedicated non-blocking HIP side stream owned by `ctx` for overlapping activation encoding. */
void* talus_npu_stage_stream(const talus_npu_ctx* ctx);

/**
 * Submit `count` NPU GEMM slices reading A buffer `a_id` and B buffers `b_ids[0..count)` into
 * C buffers `c_ids[0..count)`.
 *
 * Ordering: waits for `ready_stream` (via an async HIP event, without blocking `ready_stream`),
 * then dispatches the NPU work on the NPU worker thread. Returns 1 on success, 0 on failure.
 */
int32_t talus_npu_submit(
    talus_npu_ctx* ctx,
    void* ready_stream,
    int32_t a_id,
    const int32_t* b_ids,
    const int32_t* c_ids,
    int32_t count
);

/**
 * Register an additional kernel + instruction stream from the already-loaded xclbin.
 * Returns kernel_id >= 0 on success, or -1 on error.
 */
int32_t talus_npu_register_kernel(
    talus_npu_ctx* ctx,
    const char* kernel_name,
    const char* insts_path,
    uint32_t tokens,
    uint32_t k,
    uint32_t n,
    char* err,
    size_t err_cap
);

/** Return kernel_id >= 0 matching `(tokens, k, n)`, or -1 if not registered. */
int32_t talus_npu_find_kernel(const talus_npu_ctx* ctx, uint32_t tokens, uint32_t k, uint32_t n);

/** Submit `count` NPU GEMM slices using a specific registered `kernel_id`. */
int32_t talus_npu_submit_kernel(
    talus_npu_ctx* ctx,
    int32_t kernel_id,
    void* ready_stream,
    int32_t a_id,
    const int32_t* b_ids,
    const int32_t* c_ids,
    int32_t count
);

/**
 * Order `consumer_stream` to wait until all previously submitted NPU slices finish and their C
 * buffers are visible to the GPU. Uses hardware stream-memory wait when available, else host func.
 * Returns 1 on success, 0 on failure.
 */
int32_t talus_npu_wait(talus_npu_ctx* ctx, void* consumer_stream);

/** Read accumulated NPU stats into `out`. */
void talus_npu_get_stats(const talus_npu_ctx* ctx, talus_npu_stats_t* out);

/**
 * Check Strix Halo operational prerequisites (GPU `pp_dpm_fclk` held and NPU `power_mode`).
 * Writes a status string to `diag` and returns 1 if safe for concurrent GPU+NPU load, 0 if not.
 */
int32_t talus_npu_check_env_safety(char* diag, size_t diag_cap);

/* ---- HIP staging / decoding kernels for v8bfp16ebs8 tile layout ---- */

/** Encode FP16 activations `[valid_tokens, k_valid]` (stride `ldx`, zero-padded to `tokens`) to A tile stream `dst`. `valid_tokens == 0` means `tokens`. */
void talus_npu_encode_activations_f16(
    const uint16_t* x_f16,
    size_t ldx,
    uint32_t tokens,
    uint32_t valid_tokens,
    uint32_t k_slice,
    uint32_t k_offset,
    uint32_t k_valid,
    void* dst_dev,
    void* hip_stream
);

/** Encode Q8_0 weight rows `[row0, row0 + valid_cols)` (`k_total` wide) into `cols`-wide B tile stream `dst_dev`. */
void talus_npu_encode_weights_q8(
    const void* w_q8_0,
    uint32_t k_total,
    uint32_t row0,
    uint32_t cols,
    uint32_t valid_cols,
    uint32_t k_slice,
    uint32_t k_offset,
    uint32_t k_valid,
    void* dst_dev,
    void* hip_stream
);

/** Pack two Q8_0 weight matrices (`a` in cols `0..valid_cols`, `b` in cols `valid_cols..2*valid_cols`) into one `cols`-wide B buffer. */
void talus_npu_encode_weights_q8_pair(
    const void* w_a_q8_0,
    const void* w_b_q8_0,
    uint32_t k_total,
    uint32_t cols,
    uint32_t valid_cols,
    uint32_t k_slice,
    uint32_t k_offset,
    uint32_t k_valid,
    void* dst_dev,
    void* hip_stream
);

/** Encode FP16 weight rows `[row0, row0 + valid_cols)` (`k_total` wide) into `cols`-wide B tile stream `dst_dev`. */
void talus_npu_encode_weights_f16(
    const uint16_t* w_f16,
    uint32_t k_total,
    uint32_t row0,
    uint32_t cols,
    uint32_t valid_cols,
    uint32_t k_slice,
    uint32_t k_offset,
    uint32_t k_valid,
    void* dst_dev,
    void* hip_stream
);

/** Encode BF16 weight rows `[row0, row0 + valid_cols)` (`k_total` wide) into `cols`-wide B tile stream `dst_dev`. */
void talus_npu_encode_weights_bf16(
    const uint16_t* w_bf16,
    uint32_t k_total,
    uint32_t row0,
    uint32_t cols,
    uint32_t valid_cols,
    uint32_t k_slice,
    uint32_t k_offset,
    uint32_t k_valid,
    void* dst_dev,
    void* hip_stream
);

/** Decode NPU C tile stream (`tokens x cols`, first `valid_tokens` rows; `0` = `tokens`) into FP32 `out` (`out_pitch`, starting at column `col0`). */
void talus_npu_decode_output(
    const void* src_dev,
    float* out_dev,
    uint32_t tokens,
    uint32_t valid_tokens,
    uint32_t cols,
    size_t out_pitch,
    uint32_t col0,
    int32_t accumulate,
    uint32_t valid_cols,
    void* hip_stream
);

/** Fused SiLU(gate) * up from two NPU C tile streams directly into FP16 `[valid_tokens, valid_cols]` (`valid_tokens == 0` means `tokens`). */
void talus_npu_swiglu(
    const void* gate_c_dev,
    const void* up_c_dev,
    uint16_t* out_f16_dev,
    uint32_t tokens,
    uint32_t valid_tokens,
    uint32_t cols,
    uint32_t valid_cols,
    void* hip_stream
);

/**
 * Fused SiLU(gate) * up from a SINGLE packed NPU C buffer holding `[gate (0..valid_cols) | up (valid_cols..2*valid_cols)]`
 * (`2 * valid_cols <= cols`). Halves NPU dispatch and memory traffic for shared experts with `n_ff <= 512`.
 */
void talus_npu_swiglu_packed(
    const void* packed_c_dev,
    uint16_t* out_f16_dev,
    uint32_t tokens,
    uint32_t valid_tokens,
    uint32_t cols,
    uint32_t valid_cols,
    void* hip_stream
);

#ifdef __cplusplus
}
#endif

#endif /* TALUS_NPU_H */
