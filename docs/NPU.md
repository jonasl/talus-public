# XDNA2 NPU Hybrid GPU+NPU Execution (`aie2p`)

AMD Strix Halo APUs combine a `gfx1151` (RDNA 3.5) GPU and an XDNA2 (`aie2p` / `/dev/accel/accel0`) NPU sharing unified LPDDR5X memory. This document describes:

1. The **implemented** NPU runtime, HIP staging kernels, and `qwen4exp` (Flash-Next) hybrid GPU+NPU prefill offload (inspired by [talus-public#9](https://github.com/nsbradley88/talus-public/issues/9) / [`ZakFong/f-gufo`](https://github.com/ZakFong/f-gufo)).
2. **Future work and optimization proposals** for extending XDNA2 NPU offload to boost both **output (decode/speculative)** and **input (prefill)** rates across additional Talus model engines.

All NPU paths are **opt-in** (gated by `TALUS_NPU_DIR` or `TALUS_NPU_XCLBIN`/`TALUS_NPU_INSTS`). When unset, Talus executes purely on GPU with identical default numerical behavior.

---

## Status Summary

| Component / Model | Status | Notes |
|---|---|---|
| Core XRT `dlopen` runtime & zero-copy BOs (`src/npu/npu_gemm.*`) | Implemented | `libxrt_coreutil.so.2` pimpl ABI; `hipHostMallocMapped` + `1u << 25` (`AMDXDNA_BO_SHARE`); multi-kernel registration (`RegisterKernel`/`SubmitKernel`); idle-gated 2s keepalive (`2s`–`5s` idle window) |
| HIP `v8bfp16ebs8` staging & SwiGLU kernels (`src/npu/npu_stage.cu`) | Implemented | Positive saturation bump fix, `Q8_0`/`F16`/`BF16` + `dequant_f16` fallback, packed pair, strided `ldx`, `valid_tokens` padding, `accumulate` decode mode |
| Multi-`M` `mlir-aie` `v1.4.5` AIE2p kernels (`tools/npu/talus_npu_aie2p.xclbin`) | Implemented | `512x2560x1024`, `1024x2560x1024`, `2048x2560x1024`, `4096x2560x1024` + reproducible build (`build_xclbin.sh`, `mm_bfp_atb.cc`, `n32_core_atb.py`) |
| `qwen4exp` (Flash-Next) hybrid prefill (`src/npu/prefill_npu.*`) | Implemented | Rust (`prefill.rs`) and C++ (`prefill.cpp`) prompt paths; multi-`M` `PlanGroups` & GPU-tail hybrid dispatch (`< 384` tail on GPU in Rust); GDN `qkv` + `z`, QSA `q`, MoE `shexp`, opt-in `OutProj` (`K=6144` hooks); per-bucket auto-tuning; compatible with `--vision` and `--batch` slots |
| Concurrent `--batch` decode + `BPREFILL` overlap (`src/program/generate.cpp`) | Implemented | Zero-copy slot-resident `BPREFILL` with SJF preemption; `ov_thr` runs GPU `batch_step()` decode on `ver.cs()` concurrently with `sp.run` GPU+NPU prefill |
| Rust kernel bindings (`rust/crates/talus-kernels/src/npu.rs`) | Implemented | Safe `NpuContext` / `NpuFns` wrapper for single-kernel C ABI and unit tests (`prefill.rs` uses `tk_qwen4exp_p2_npu_*` for multi-kernel dispatch) |
| Parity & throughput harness (`tests/hip/npu_bench.cpp`) | Implemented | Built as `talus-npu-bench` |
| DFlash2 / MTP & batched `slot_mtp` decode offload | Proposed | Requires small-`M` (`64`/`128`/`256`) `mlir-aie` kernels |
| Vision Tower (`src/vision/`) offload & overlap | Proposed | Overlaps ViT patches on NPU with LLM prefix prefill or `--batch` decode |
| Background/parked NPU prefill during GPU-only decode phases | Proposed | Decouple `BYIELD`-parked or queued NPU prefill stages when no `BPREFILL` chunk is active |
| Prebuilt `K=6144`, `K=2048`, `K=5120`, ViT, and small-`M` `.xclbin` kernels | Proposed | `OutProj` runtime hooks (`K=6144`) already exist in `prefill_npu.cpp` / `prefill.rs`; needs compiled kernel binaries |
| Dense `qwen35` (`Qwen3.8-27B`) & `glm` (`GLM-5.3-Flash`) offload | Proposed | Split-K (`K=5120`) and zero-padded `K=2048` into `K=2560` (or native `K` kernels) |
| Hot MoE expert offload (`counts_host_`) | Proposed | Requires medium-`M` (`M=128`) `mlir-aie` kernel |

---

## Prerequisites

### System Firmware (AMD PI `1.0.0.2c` or Newer)

On AMD Strix Halo platforms running earlier system firmware, dynamic fabric clock scaling (`pp_dpm_fclk` in `auto` DPM mode) during concurrent GPU (`gfx1151`) and XDNA2 NPU (`aie2p`) execution can trigger NPU firmware hangs and XRT command timeouts.

AMD resolved this fabric clock interaction at the firmware level in **AMD PI version `1.0.0.2c`**. For NPU prefill to run stably without manual fabric clock workarounds, **the host must run BIOS/system firmware incorporating AMD PI version `1.0.0.2c` (or newer)**, downloaded and installed from the system manufacturer.

*(Legacy workaround for un-updated hosts: on firmware older than AMD PI `1.0.0.2c`, `sudo ./tools/npu/fabric-clock.sh hold` can pin `pp_dpm_fclk` to its highest level while `TALUS_NPU_REQUIRE_HELD_FCLK=1` enforces the check, at the cost of higher idle power and heat until released with `release`.)*

### Hardware, Driver, and NPU Artifacts

- **Hardware**: AMD Strix Halo APU (`gfx1151` GPU + XDNA2 `aie2p` NPU) with unified memory.
- **Linux kernel & userspace**:
  - `amdxdna` kernel driver exposing `/dev/accel/accel0`.
  - XRT runtime library `libxrt_coreutil.so.2` + `libxrt_driver_xdna.so.2` installed on the host. No XRT development headers or `mlir-aie` toolchain are required at build time (`libxrt_coreutil.so.2` is loaded via `dlopen`).
  - **Locked memory limit (`RLIMIT_MEMLOCK` / `ulimit -l`)**: XRT's `amdxdna` driver maps a 64 MiB `dev_heap` buffer with `MAP_LOCKED`, in addition to the pinned `hipHostMallocMapped` BOs allocated for NPU activations, weights, and outputs. The process's locked-memory limit (`ulimit -l`) must be large enough to hold both (otherwise `xrt::hw_context` creation or BO allocation fails with `mmap(..., flags=8209) failed (err=-11): Resource temporarily unavailable`). Configure `memlock` to `unlimited` (or at least 2 GiB, depending on slice and group counts) for your user or the `render` group in `/etc/security/limits.d/` (and re-login; for systemd services, set `LimitMEMLOCK=infinity`).
- **NPU kernel binaries** (bundled in `tools/npu/`):
  - Multi-`M` `mlir-aie` `v1.4.5` ATB kernels (`512x2560x1024`, `1024x2560x1024`, `2048x2560x1024`, `4096x2560x1024`):
    - `tools/npu/talus_npu_aie2p.xclbin`
    - `tools/npu/insts_512x2560x1024.bin`, `insts_1024x2560x1024.bin`, `insts_2048x2560x1024.bin`, `insts_4096x2560x1024.bin`
    - Reproducible build sources: `tools/npu/mm_bfp_atb.cc`, `tools/npu/n32_core_atb.py`, `tools/npu/build_xclbin.sh`
  - Legacy single-shape fallback (from [`ZakFong/f-gufo`](https://github.com/ZakFong/f-gufo/tree/fork-enhancements/tools/npu), alongside `mm_bfp_mixed.cc` and `n32_core_placed.py`):
    - `tools/npu/final_4096x2560x1024_128x64x128.xclbin`
    - `tools/npu/insts_4096x2560x1024_128x64x128.txt`
  - Point `TALUS_NPU_DIR=./tools/npu` (auto-selects `talus_npu_aie2p.xclbin` when present, or set `TALUS_NPU_XCLBIN` and `TALUS_NPU_INSTS` directly).

---

## Enabling & Configuration

### Quick Start

```bash
# 1. Build Talus (includes NPU runtime + talus-npu-bench when HIP is enabled)
./build.sh
cmake --build build --target talus-npu-bench -j"$(nproc)"

# 2. Run the standalone parity/benchmark check
TALUS_NPU_DIR=./tools/npu ./build/talus-npu-bench --require-npu

# 3. Enable during qwen4exp prefill
TALUS_NPU_DIR=./tools/npu ./build/talus-cli ...
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TALUS_NPU_DIR` | unset (off) | Directory containing `talus_npu_aie2p.xclbin` + `insts_*x2560x1024.bin` (or legacy `final_4096x2560x1024_128x64x128.xclbin` + `.txt`; `GUFO_NPU_DIR` accepted as alias) |
| `TALUS_NPU_XCLBIN` | `<dir>/talus_npu_aie2p.xclbin` | Explicit path to `.xclbin` |
| `TALUS_NPU_INSTS` | `<dir>/insts_4096x2560x1024.bin` | Explicit path to instruction stream file |
| `TALUS_NPU_KERNEL_NAME` | `GEMM_4096_2560_1024` (multi-`M`) / `MLIR_AIE` (legacy) | Override primary kernel name inside the `.xclbin` |
| `TALUS_NPU_MIN_T` | `1536` (`2048` legacy) | Minimum chunk token count `t_n` eligible for NPU offload (clamped `>= 128`) |
| `TALUS_NPU_MAX_GROUPS` | `4` | Maximum NPU groups per chunk (`1..4`, up to `16384` tokens) |
| `TALUS_NPU_QKV_SLICES` | `-1` (auto, starts at `2`) | Fixed number of `1024`-col slices (`0..10`, auto-tunes `0..6`) offloaded for GDN `attn_qkv` |
| `TALUS_NPU_Z_SLICES` | `-1` (auto, starts at `2`) | Fixed number of `1024`-col slices (`0..6`) offloaded for GDN `attn_gate` (`z`) |
| `TALUS_NPU_Q_SLICES` | `-1` (auto, starts at `4`) | Fixed number of `1024`-col slices (`0..12`) offloaded for QSA `attn_q` |
| `TALUS_NPU_OUT_SLICES` | `0` | Opt-in `1024`-col slices (`0..2`) for `K=6144` `ssm_out`/`attn_output` when `K=6144` kernels are present |
| `TALUS_NPU_SHEXP` | `1` | `1` = offload MoE `ffn_gate_shexp` + `ffn_up_shexp` with fused SwiGLU, `0` = GPU |
| `TALUS_NPU_AUTO_TUNE` | `1` | Dynamically adjusts `qkv_slices`, `z_slices`, and `q_slices` per token-length bucket based on `host_wait_us_total` |
| `TALUS_NPU_ISOLATE` | `0` | `0` = zero-copy `hipHostMallocMapped` shared BO, `1` = private XRT BO + `CopyPool` |
| `TALUS_NPU_KEEPALIVE` | `1` | Idle-gated 2s heartbeat (fires for 5s after last NPU submit to avoid `D3hot` wakeups between chunks without stealing decode bandwidth; `0` = off) |
| `TALUS_NPU_HW_WAIT` | `0` | `1` = use `hipStreamWaitValue32` on mapped completion counter when supported |
| `TALUS_NPU_TIMEOUT_MS` | `8000` | Per-run XRT wait timeout in milliseconds |
| `TALUS_NPU_REQUIRE_HELD_FCLK` | `0` | `1` = require held `pp_dpm_fclk` (`manual` top level) before NPU init (legacy pre-`1.0.0.2c` safety guard) |
| `TALUS_NPU_REQUIRED` | `0` | `1` = fail prompt instead of falling back to GPU if NPU initialization fails |
| `TALUS_NPU_STATS` | unset | Print periodic NPU timing breakdown (`encode`, `launch`, `wait`, `copy`, `stream blocked`) |
| `TALUS_NPU_TRACE` | unset | File path for timestamped NPU lifecycle trace lines |

---

## Implemented Architecture (`qwen4exp` Hybrid Prefill)

### 1. XRT Runtime & Memory Sharing (`src/npu/npu_gemm.{hpp,cpp}`, `src/npu/talus_npu.cpp`, `include/talus_npu.h`)

- **XRT C++ ABI loader**: Opens `libxrt_coreutil.so.2` via `dlopen` and resolves mangled C++ pimpl symbols for `xrt::device`, `xrt::xclbin`, `xrt::hw_context`, `xrt::kernel`, `xrt::bo`, `xrt::run`, and `xrt::uuid` (sret).
- **Zero-copy GPU/NPU buffers**: `NpuGemm::NewBuffer` allocates pinned mapped host memory (`hipHostMalloc(..., hipHostMallocMapped)`) and registers it with XRT using `xrt::bo::flags = 1u << 25` (`AMDXDNA_BO_SHARE`), allowing both HIP kernels and the NPU to access the same physical pages without host copies. (`TALUS_NPU_ISOLATE=1` uses `1u << 31` private XRT BOs + multi-threaded `CopyPool` for debugging.)
- **Multi-kernel registration & per-kernel `xrt::run` pools**: `RegisterKernel` / `FindKernel` / `SubmitKernel` load multiple `M x K x N` kernels (`GEMM_{4096,2048,1024,512}_2560_1024`, plus optional `GEMM_{M}_6144_1024`) onto a single shared `xrt::hw_context`. Each `KernelSlot` pre-allocates `16` reusable `xrt::run` handles with fixed arguments (`opcode=3`, instruction BO, word count) and pipelines up to `kMaxInFlight = 4` concurrent slices on the worker thread.
- **Async stream coordination & idle-gated keepalive**: Activation encoding runs on a dedicated non-blocking `stage_stream` ordered after `cs` via `hipEventRecord` + `hipStreamWaitEvent` (without blocking `cs`). Completion on `cs` uses either `hipStreamWaitValue32` (`TALUS_NPU_HW_WAIT=1`) or `hipLaunchHostFunc`. The keepalive thread (`TALUS_NPU_KEEPALIVE=1`) fires `Heartbeat()` when idle between `2s` and `5s` after the last NPU submit, keeping the NPU warm across chunk boundaries while letting it drop into `D3hot` during decode-heavy or idle periods.

### 2. HIP `v8bfp16ebs8` Staging & Epilogue Kernels (`src/npu/npu_stage.cu`)

- **Positive saturation fix (`NpuEncodeGroup`)**: For values in `[1.984375, 2.0) * 2^(e-127)`, checks whether incrementing the shared 8-element block exponent by `+1` lowers L2 reconstruction error over the group rather than saturating to `+127/64`.
- **Flexible weight encoding**: `PrefillNpu::EnsureWeightSlice` lazily encodes and caches weight slices on first layer encounter from `Q8_0` (`talus_npu_encode_weights_q8`), `F16` (`talus_npu_encode_weights_f16`), `BF16` (`talus_npu_encode_weights_bf16`), or any other GGML quantization (`Q4_K`, `Q6_K`, `IQ4_NL`, etc.) via `strata::kernels::dequant_f16`.
- **Packed gate/up & fused SwiGLU**: `talus_npu_encode_weights_q8_pair` + `talus_npu_swiglu_packed` support packing two `<= 512`-col matrices into one `1024`-col NPU slice; `talus_npu_swiglu` fuses two NPU C tile streams directly into FP16 output without writing FP32 intermediates.
- **Bounded chunk rows (`valid_tokens`), strided activations, & accumulation**: `talus_npu_encode_activations_f16` supports arbitrary row stride `ldx` (for `OutProj` inputs), `k_offset`/`k_valid`, and zero-pads rows `>= valid_tokens` safely across `M = 512, 1024, 2048, 4096` tiles; `talus_npu_decode_output` and `swiglu` skip trailing padded rows, and `decode_output` supports `accumulate` (`0` overwrite / `1` accumulate).

### 3. `qwen4exp` (Flash-Next) Prompt Integration (`src/npu/prefill_npu.{hpp,cpp}`, `src/program/qwen4exp_p2.cpp`, `rust/crates/talus-model/src/qwen4exp/prefill.rs`, `src/prefill/prefill.cpp`)

Both the Rust (`TALUS_FN_RUST=1`) and C++ (`TALUS_FN_RUST=0`) prompt paths integrate `PrefillNpu` at the same model seams (`n_embd = 2560`):

- **Multi-`M` tile planning & GPU-tail hybrid dispatch (`PlanGroups`)**: When `talus_npu_aie2p.xclbin` is loaded, `PlanGroups` selects from `M = 4096, 2048, 1024, 512` per group (up to `TALUS_NPU_MAX_GROUPS = 4`) to minimize padding overhead. On the Rust `prefill.rs` path (`npu_rows_out`), any trailing remainder smaller than `kGpuTailThreshold = 384` tokens (`t_n - npu_rows`) is dispatched to the GPU (`gemm_native` / `native_proj`) instead of padding a `512`-row NPU tile; the C++ `prefill.cpp` path dispatches full-chunk NPU groups.
- **Per-bucket adaptive slice tuning**: `PrefillNpu` tracks `host_wait_us_total` independently across three token-length buckets (`< 1536`, `1536..3583`, `>= 3584`) and dynamically adjusts `qkv_slices`, `z_slices`, `q_slices`, and (when `K=6144` kernels are loaded) `out_slices`.

1. **GDN layers (36 of 48 layers) — `attn_qkv.weight` (width `10240`) + `attn_gate.weight` (`z`, width `6144`)**:
   - **Phase 1 (`StartGdnQkv` / `FinishGdnQkv`)**: Trailing `qkv_slices` (`1024` cols each) of `attn_qkv` are submitted to NPU (`C` pool slots `6..15`) while GPU computes the remaining `qkv_gpu_cols` columns of `attn_qkv`, `ssm_alpha`, `ssm_beta`, and `gdn_gates` (plus any `< 384`-token tail on Rust). Before `gdn_conv`, `FinishGdnQkv` waits for NPU and decodes the trailing `qkv` columns into `m.qkv`.
   - **Phase 2 (`StartGdnZ` / `gdn_recurrence_npu`)**: Trailing `z_slices` (`1024` cols each) of `attn_gate` are submitted to NPU (`C` pool slots `0..5`, disjoint from `qkv` slots) — reusing the already-encoded `a_bufs` activations from `StartGdnQkv` without re-encoding `m.mixed_h` — while GPU computes any remaining `z_gpu_cols` columns of `z`, `gdn_conv`, and the GDN recurrence kernel. (When `qkv_slices == 0`, `StartGdnZ` starts before full GPU `attn_qkv`.)
   - `gdn_recurrence_npu` (`src/prefill/kernels.cu`, `include/strata/prefill/kernels.hpp`) runs the recurrence first, waits for NPU via `before_epilogue`, and executes `gdn_out_norm_npu_kernel`, which reads NPU `z` columns directly from `z_streams` (`v8bfp16ebs8` C tiles) without decoding `z` into FP32 DRAM.
2. **QSA layers (12 of 48 layers) — `attn_q.weight` (`qf`, width `12288`)**:
   - Trailing `q_slices` (`1024` cols each) of `attn_q` are submitted to NPU while GPU computes `attn_k`, `attn_v`, GPU `attn_q` columns, indexer projections, RoPE, KV staging/append, and top-K block selection.
   - Before `split_q`, `FinishQsaQ` waits for NPU and runs `talus_npu_decode_output` into the trailing columns of `m.Qf`.
3. **Attention / GDN output projections (`StartOutProj` / `FinishOutProj`, opt-in via `TALUS_NPU_OUT_SLICES`)**:
   - Runtime hooks are wired in `PrefillNpu` (`K = 6144` -> `2560`, `ssm_out.weight` and `attn_output.weight`) and `prefill.rs` whenever `GEMM_{M}_6144_1024` kernels are present in `TALUS_NPU_DIR` (prebuilt `tools/npu/talus_npu_aie2p.xclbin` currently ships `K = 2560` shapes only).
4. **MoE layers (all 48 layers) — `ffn_gate_shexp.weight` + `ffn_up_shexp.weight` (width `640`)**:
   - Submitted to NPU (`2` slices per group) before GPU routed expert execution, overlapping shared-expert up/gate projections with all `512` routed experts (`fused_experts` or `moe_grouped`).
   - After routed experts finish, `FinishMoeShexp` waits for NPU and runs `talus_npu_swiglu` directly into FP16 `m.sh_h` before `ffn_down_shexp` and `moe_combine` / `moe_write_norm_rs`.

### 4. Native Vision Tower & `--batch` Slot Compatibility

- **Multimodal vision prompts (`--vision`, `TALUS_VISION_MM`)**:
  - On the solo Rust serve loop (`TALUS_FN_RUST_VISION=1`), `NativeEncoder` (`rust/crates/talus-model/src/qwen4exp/vision.rs`) encodes `@raw` frames via `tk_vision_encode_raw` and hands device pointers to `prefill.rs` through `DevicePictures` (`PictureRows::Device` in `picture.rs`).
  - Under `--batch` (`talus-server/src/batch.rs` + `src/program/generate.cpp`), `BGENI ... @raw` frames are decoded by `StdinReader` and encoded by `tk_vision_encode_raw` into `PictureRows::HostTable` (`sp.embd_rows`).
  - Once `embed_chunk_rows` merges image embeddings into `m.emb` and `gr_broadcast` / `gr16_broadcast` populates `R`, every chunk with `t_n >= TALUS_NPU_MIN_T` receives NPU offload transparently across both text and image tokens.
- **Multi-slot batching (`--batch`, `BGEN`/`BGENI`/`BPREFILL`/`BYIELD`/`BSTOP`)**:
  - `BPREFILL` (`src/program/generate.cpp`, `rust/crates/talus-server/src/batch.rs`) runs zero-copy slot-resident prefill directly into the target slot's session with Shortest-Job-First (SJF) preemption, while `ov_thr` concurrently runs GPU `batch_step()` decode windows on `ver.cs()` (high-priority HIP verify stream) alongside `sp.run` GPU+NPU prefill on `sp.cs`.
  - Vision (`BGENI`) and logprob/schema requests use the `read_part` path, interleaving `batch_step()` decode windows (`STRATA_BATCH_DECODE_SHARE`) and honoring `BYIELD` mid-read preemption.
  - Because `PrefillNpu` completes and synchronizes all NPU slices within each `sp.run` chunk (with an `op_inflight` drain guard in `Start*` and `EndChunk()` for cancelled/aborted segments), NPU offload composes safely across `BPREFILL` SJF preemptions, `BYIELD` yields, `BGEN` slot admissions, and `parallel_solo` handoffs.

---

## Future Work & High-Leverage Uses of the XDNA2 NPU in Talus

To keep both `gfx1151` (GPU) and `aie2p` (NPU) utilized across both **output (decode/speculative)** and **input (prefill)** workloads — including native vision and multi-slot batched serving — the following extensions remain unimplemented and target Talus's remaining bottlenecks.

### A. Boosting Output (Decode & Speculative Decoding) Rates

Interactive `T=1` decode on Strix Halo is memory-bandwidth bound, leaving GPU compute underutilized while the NPU sits idle. The NPU's ~4 MB local AIE2p SRAM and matrix throughput help decode whenever tokens are **batched/verified together** or **speculative work overlaps GPU verification**:

1. **DFlash2 / MTP Multi-Token Block Drafting & Lookahead (`qwen35`, `mimo`, `qwen4exp`)**:
   - `qwen35` (`rust/crates/talus-model/src/qwen35/engine.rs`, `src/qwen35/q36_dflash.cu`), `mimo`, and `qwen4exp` verify `K = 8..32` draft tokens in a single target-model forward pass while running smaller draft MLP/attention layers.
   - Compiling a small-`M` (`64xKxN`) `mlir-aie` kernel lets the NPU execute **pipelined lookahead draft FFN projections or LM-head shortlists** concurrently while the GPU verifies the current candidate block, hiding drafting latency behind target verification.
2. **Batched Slot Verification & Multi-Slot `slot_mtp` Drafting (`--batch`, `--batch-mtp`)**:
   - Under concurrent multi-request serving in `talus-server` (`rust/crates/talus-server/src/batch.rs`), `batch_step()` (`src/program/generate.cpp`) packs `B` active slots (`ver.run_slot_rows`) into a single verification batch (`T = B` or `B * K` with speculative tokens) and then loops sequentially over `slot_mtp[b]->draft(...)`.
   - An `M = 64` or `M = 128` `xclbin` enables offloading dense `gate/up` or `attn_q` verification slices when `B * K >= 64`, and **batching all active `slot_mtp` draft forward passes into a single NPU call** overlapped with GPU `commit_slot_prefixes`.
3. **Zero-Stall Background Auxiliary Tasks During Decode**:
   - Between interactive decode steps (or during streaming), the NPU can asynchronously process **dense embedding/reranking models** (`M = 256..4096`) or **async session-save / context summarization prefill** for a queued request without taking GPU CUs away from the active decode stream.

### B. Boosting Input (Prefill) Rates with Vision & Batching

1. **Native Vision Tower (ViT) Offload & Overlap (`src/vision/vit_gpu.cu`, `tk_vision_*`, `NativeEncoder`)**:
   - `VitGpu` processes `N_patches = 4 * N_img_tokens` (up to `16,384` patches for a `4096`-token image) through non-causal dense transformer blocks (`qkv`, `o`, `up`, `down`, and `2x2` spatial merger). Currently `tk_vision_encode_raw` runs synchronously on GPU before LLM prefill begins.
   - **GPU+NPU overlap**: Because ViT encoding has no causal dependency on preceding text tokens until `embed_chunk_rows`, running ViT dense projections on NPU **concurrently with GPU prefill of the text prompt prefix** (`[0 .. first_<|image_pad|>)`) or **during active `--batch` `batch_step()` decode windows** hides vision encoding latency.
   - **Multi-image `M = 4096` patch packing**: Packing patches across multiple images in a single `GENI`/`BGENI` request (or queued requests) into `4096`-patch NPU tiles amortizes launch overhead even for smaller images (`256..1024` patches).
   - **Zero-copy `DevicePictures` handoff**: Writing ViT outputs directly into a reusable `hipHostMallocMapped` / device arena rather than copying `VitGpu` -> `v->scratch` -> `hipMalloc` (`NativeEncoder`) -> `D2D` (`m.emb`) eliminates redundant memory copies on unified memory.
2. **Background/Parked NPU Prefill During GPU-Only Decode Phases**:
   - While `BPREFILL` + `ov_thr` already overlaps GPU `batch_step()` decode with active slot-resident GPU+NPU prefill, non-`BPREFILL` (`read_part` / `BGENI`) paths still alternate serially, and queued or `BYIELD`-parked requests do not run NPU-only prefill stages asynchronously during GPU-only decode windows.
   - Decoupling queued NPU activation staging/projections (or running background `BPREFILL` for `BGENI` once vision is integrated) during GPU decode phases would keep both accelerators saturated across all request types.
3. **Dense `Qwen3.5` / `Qwen3.8-27B` (`rust/crates/talus-model/src/qwen35/engine.rs`)**:
   - Dense 27B layers have large projections (`K = 5120`, `n_ff = 17408`, plus `attn_qkv`/`attn_gate`).
   - Either splitting `K = 5120` into two `2560` halves using `talus_npu_decode_output(..., accumulate=1)` or compiling a dedicated `K = 5120` `xclbin` allows offloading dense `ffn_gate`/`ffn_up` or GDN `z` slices concurrently with GPU attention and recurrence.
4. **`GLM-5.3-Flash` (`rust/crates/talus-model/src/glm/prefill.rs`)**:
   - `GLM-5.3-Flash` uses `GLM_PCH = 4096` and `n_embd = 2048`.
   - `K_valid = 2048` can be zero-padded directly into the existing `512..4096x2560x1024` kernels via `talus_npu_encode_activations_f16` / `talus_npu_encode_weights_*` (or served by a dedicated `K = 2048` `xclbin`) to offload `KDA` (`z`/`q`) and shared-expert MLP slices.
5. **Hot MoE Experts via `counts_host_` (`qwen4exp`)**:
   - During `T = 4096` MoE routing, the hottest `10..20` experts per layer often receive `128..512+` tokens.
   - Using `counts_host_` to partition work — running the largest experts on GPU (`WMMA`/`MMQ`) while dispatching medium-batch (`~64..128` token) packed `gate+up` hot experts onto an `M = 128` NPU kernel — can further balance MoE prefill load.

### C. Runtime & Kernel Optimizations

- **Additional `mlir-aie` shapes**: Multi-`M` `K = 2560` kernels (`512`, `1024`, `2048`, `4096`) are already built into `talus_npu_aie2p.xclbin`; remaining shapes to compile are:
  - Small-`M` (`M = 64, 128, 256`) kernels for speculative verify, `slot_mtp` batching, hot MoE experts, and smaller tails;
  - `K = 6144` (`GEMM_{M}_6144_1024`) binaries in `tools/npu/build_xclbin.sh` to activate the existing `StartOutProj` / `FinishOutProj` runtime hooks;
  - Native `K = 2048`, `K = 5120`, and ViT-width kernels to eliminate zero-padding or two-pass split-K.
- **Direct BF16 activation encoding (`EncodeActivationsBF16`)**: Add a `BF16` activation encoder for engines/paths whose native 16-bit activation buffer is `BF16` rather than `FP16`.
- **Full multi-kernel Rust bindings (`rust/crates/talus-kernels/src/npu.rs`)**: Extend `NpuConfig` / `NpuContext` in `talus-kernels` to expose `kernel_name`, `register_kernel`, `find_kernel`, and `submit_kernel` (currently used only via `qwen4exp_p2.cpp` FFI).
- **Cross-model weight arena & `MemAvailable` integration**: Add explicit eviction/LRU bounds on cached NPU weight buffers when switching models inside a long-lived `talus-server` process, and account for `PrefillNpu` pinned shared BOs alongside `NativeEncoder::floor_claim()` in `talus_mem::pressure`.
