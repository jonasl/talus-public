# XDNA2 NPU Hybrid GPU+NPU Execution (`aie2p`)

AMD Strix Halo APUs combine a `gfx1151` (RDNA 3.5) GPU and an XDNA2 (`aie2p` / `/dev/accel/accel0`) NPU sharing unified LPDDR5X memory. This document describes:

1. The **implemented** NPU runtime, HIP staging kernels, and `qwen4exp` (Flash-Next) hybrid GPU+NPU prefill offload (inspired by [talus-public#9](https://github.com/nsbradley88/talus-public/issues/9) / [`ZakFong/f-gufo`](https://github.com/ZakFong/f-gufo)).
2. **Future work and optimization proposals** for extending XDNA2 NPU offload to boost both **output (decode/speculative)** and **input (prefill)** rates across additional Talus model engines.

All NPU paths are **opt-in** (gated by `TALUS_NPU_DIR` or `TALUS_NPU_XCLBIN`/`TALUS_NPU_INSTS`). When unset, Talus executes purely on GPU with identical default numerical behavior.

---

## Status Summary

| Component / Model | Status | Notes |
|---|---|---|
| Core XRT `dlopen` runtime & zero-copy BOs (`src/npu/npu_gemm.*`) | Implemented | `libxrt_coreutil.so.2` pimpl ABI; `hipHostMallocMapped` + `1u << 25` (`AMDXDNA_BO_SHARE`) |
| HIP `v8bfp16ebs8` staging & SwiGLU kernels (`src/npu/npu_stage.cu`) | Implemented | Positive saturation bump fix, `Q8_0`/`F16`/`BF16`, packed pair, `valid_tokens` padding |
| `qwen4exp` (Flash-Next) hybrid prefill (`src/npu/prefill_npu.*`) | Implemented | Rust (`prefill.rs`) and C++ (`prefill.cpp`) prompt paths; GDN `z`, QSA `q`, MoE `shexp`; compatible with `--vision` and `--batch` slots |
| Rust kernel bindings (`rust/crates/talus-kernels/src/npu.rs`) | Implemented | Safe `NpuContext` / `NpuFns` wrapper and unit tests |
| Parity & throughput harness (`tests/hip/npu_bench.cpp`) | Implemented | Built as `talus-npu-bench` |
| DFlash2 / MTP & batched `slot_mtp` decode offload | Proposed | Requires small-M (`64`/`128`/`256`) `mlir-aie` kernels |
| Vision Tower (`src/vision/`) offload & overlap | Proposed | Overlaps ViT patches on NPU with LLM prefix prefill or `--batch` decode |
| Interleaved `--batch` prefill/decode overlap | Proposed | Pipelines next-chunk NPU prefill with `batch_step()` GPU decode |
| Dense `qwen35` (`Qwen3.8-27B`) & `glm` (`GLM-5.3-Flash`) offload | Proposed | Split-K (`K=5120`) and zero-padded `K=2048` into `K=2560` |
| Hot MoE expert offload (`counts_host_`) | Proposed | Requires medium-M (`M=128`) `mlir-aie` kernel |

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
  - **Locked memory limit (`RLIMIT_MEMLOCK` / `ulimit -l`)**: XRT's `amdxdna` driver maps a 64 MiB `dev_heap` buffer with `MAP_LOCKED` (in addition to `hipHostMallocMapped` NPU staging BOs). If `ulimit -l` is left at the default 8 MiB (`8192` KB), `xrt::hw_context` creation fails with `mmap(..., len=67108864, ..., flags=8209) failed (err=-11): Resource temporarily unavailable`. Configure `memlock` to `unlimited` (or at least 2 GiB) for the `render` group / user in `/etc/security/limits.d/` (and re-login).
- **NPU kernel binaries** (bundled in `tools/npu/` from [`ZakFong/f-gufo`](https://github.com/ZakFong/f-gufo/tree/fork-enhancements/tools/npu), alongside AIE sources `mm_bfp_mixed.cc` and `n32_core_placed.py`):
  - `tools/npu/final_4096x2560x1024_128x64x128.xclbin`
  - `tools/npu/insts_4096x2560x1024_128x64x128.txt`
  - Point `TALUS_NPU_DIR=./tools/npu` (or set `TALUS_NPU_XCLBIN` and `TALUS_NPU_INSTS` directly).

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
| `TALUS_NPU_DIR` | unset (off) | Directory containing `final_4096x2560x1024_128x64x128.xclbin` and `insts_4096x2560x1024_128x64x128.txt` (`GUFO_NPU_DIR` accepted as alias) |
| `TALUS_NPU_XCLBIN` | `<dir>/final_4096x2560x1024_128x64x128.xclbin` | Explicit path to `.xclbin` |
| `TALUS_NPU_INSTS` | `<dir>/insts_4096x2560x1024_128x64x128.txt` | Explicit path to instruction stream file |
| `TALUS_NPU_MIN_T` | `2048` | Minimum chunk token count `t_n` eligible for NPU offload |
| `TALUS_NPU_MAX_GROUPS` | `4` | Maximum `4096`-token groups per chunk (`1..4`, up to `16384` tokens) |
| `TALUS_NPU_Z_SLICES` | `-1` (auto, starts at `2`) | Fixed number of `1024`-col slices (`0..6`) offloaded for GDN `attn_gate` (`z`) |
| `TALUS_NPU_Q_SLICES` | `-1` (auto, starts at `4`) | Fixed number of `1024`-col slices (`0..12`) offloaded for QSA `attn_q` |
| `TALUS_NPU_SHEXP` | `1` | `1` = offload MoE `ffn_gate_shexp` + `ffn_up_shexp` with fused SwiGLU, `0` = GPU |
| `TALUS_NPU_AUTO_TUNE` | `1` | Dynamically adjusts `z_slices` and `q_slices` based on `host_wait_us_total` |
| `TALUS_NPU_ISOLATE` | `0` | `0` = zero-copy `hipHostMallocMapped` shared BO, `1` = private XRT BO + `CopyPool` |
| `TALUS_NPU_KEEPALIVE` | `1` | Background 1s heartbeat to prevent `D3hot` wakeups between chunks (`0` = off) |
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
- **Pipelined `xrt::run` pool**: Pre-allocates a pool of `16` reusable `xrt::run` handles with fixed kernel arguments (`opcode=3`, instruction BO, word count) and maintains `kMaxInFlight = 4` concurrent slices on the worker thread, avoiding per-slice ERT command allocation overhead.
- **Async stream coordination**: Activation encoding runs on a dedicated non-blocking `stage_stream` ordered after the compute stream `cs` via `hipEventRecord` + `hipStreamWaitEvent` (without blocking `cs`). Completion on `cs` uses either `hipStreamWaitValue32` (`TALUS_NPU_HW_WAIT=1`) or `hipLaunchHostFunc`.

### 2. HIP `v8bfp16ebs8` Staging & Epilogue Kernels (`src/npu/npu_stage.cu`)

- **Positive saturation fix (`NpuEncodeGroup`)**: For values in `[1.984375, 2.0) * 2^(e-127)`, checks whether incrementing the shared 8-element block exponent by `+1` lowers L2 reconstruction error over the group rather than saturating to `+127/64`.
- **Flexible weight encoding**: `PrefillNpu::EnsureWeightSlice` lazily encodes and caches weight slices on first layer encounter from `Q8_0` (`talus_npu_encode_weights_q8`), `F16` (`talus_npu_encode_weights_f16`), `BF16` (`talus_npu_encode_weights_bf16`), or any other GGML quantization (`Q4_K`, `Q6_K`, `IQ4_NL`, etc.) via `strata::kernels::dequant_f16`.
- **Packed gate/up & fused SwiGLU**: `talus_npu_encode_weights_q8_pair` + `talus_npu_swiglu_packed` support packing two `<= 512`-col matrices into one `1024`-col NPU slice; `talus_npu_swiglu` fuses two NPU C tile streams directly into FP16 output without writing FP32 intermediates.
- **Bounded chunk rows (`valid_tokens`)**: When `t_n` is not an exact multiple of `4096` (for example `t_n >= 2048` remainder chunks), `talus_npu_encode_activations_f16` zero-pads rows `>= valid_tokens` safely and `decode_output` / `swiglu` skip trailing padded rows.

### 3. `qwen4exp` (Flash-Next) Prompt Integration (`src/npu/prefill_npu.{hpp,cpp}`, `src/program/qwen4exp_p2.cpp`, `rust/crates/talus-model/src/qwen4exp/prefill.rs`, `src/prefill/prefill.cpp`)

Both the Rust (`TALUS_FN_RUST=1`) and C++ (`TALUS_FN_RUST=0`) prompt paths integrate `PrefillNpu` at the same three model seams (`n_embd = 2560`):

1. **GDN layers (36 of 48 layers) — `attn_gate.weight` (`z`, width `6144`)**:
   - Trailing `z_slices` (`1024` cols each) of `z` are submitted to NPU while GPU computes `attn_qkv`, any remaining `z_gpu_cols` columns of `z`, `ssm_alpha`, `ssm_beta`, `gdn_gates`, `gdn_conv`, and the GDN recurrence kernel.
   - `gdn_recurrence_npu` (`src/prefill/kernels.cu`, `include/strata/prefill/kernels.hpp`) runs the recurrence first, waits for NPU via `before_epilogue`, and executes `gdn_out_norm_npu_kernel`, which reads NPU `z` columns directly from `z_streams` (`v8bfp16ebs8` C tiles) without decoding `z` into FP32 DRAM.
2. **QSA layers (12 of 48 layers) — `attn_q.weight` (`qf`, width `12288`)**:
   - Trailing `q_slices` (`1024` cols each) of `attn_q` are submitted to NPU while GPU computes `attn_k`, `attn_v`, GPU `attn_q` columns, indexer projections, RoPE, KV staging/append, and top-K block selection.
   - Before `split_q`, `FinishQsaQ` waits for NPU and runs `talus_npu_decode_output` into the trailing columns of `m.Qf`.
3. **MoE layers (all 48 layers) — `ffn_gate_shexp.weight` + `ffn_up_shexp.weight` (width `640`)**:
   - Submitted to NPU (`2` slices per `4096`-token group) before GPU routed expert execution, overlapping shared-expert up/gate projections with all `512` routed experts (`fused_experts` or `moe_grouped`).
   - After routed experts finish, `FinishMoeShexp` waits for NPU and runs `talus_npu_swiglu` directly into FP16 `m.sh_h` before `ffn_down_shexp` and `moe_combine` / `moe_write_norm_rs`.

### 4. Native Vision Tower & `--batch` Slot Compatibility

- **Multimodal vision prompts (`--vision`, `TALUS_VISION_MM`)**:
  - On the solo Rust serve loop (`TALUS_FN_RUST_VISION=1`), `NativeEncoder` (`rust/crates/talus-model/src/qwen4exp/vision.rs`) encodes `@raw` frames via `tk_vision_encode_raw` and hands device pointers to `prefill.rs` through `DevicePictures` (`PictureRows::Device` in `picture.rs`).
  - Under `--batch` (`talus-server/src/batch.rs` + `src/program/generate.cpp`), `BGENI ... @raw` frames are decoded by `StdinReader` and encoded by `tk_vision_encode_raw` into `PictureRows::HostTable` (`sp.embd_rows`).
  - Once `embed_chunk_rows` merges image embeddings into `m.emb` and `gr_broadcast` / `gr16_broadcast` populates `R`, every chunk with `t_n >= TALUS_NPU_MIN_T` receives NPU offload transparently across both text and image tokens.
- **Multi-slot batching (`--batch`, `BGEN`/`BGENI`/`BYIELD`/`BSTOP`)**:
  - `read_part` (`src/program/generate.cpp`) steps through long prompts one `batch_read_piece` (`C` tokens, `include/strata/program/batch_read.hpp`) at a time, interleaving `batch_step()` decode windows (`STRATA_BATCH_DECODE_SHARE`) and honoring `BYIELD` mid-read preemption.
  - Because `PrefillNpu` completes and synchronizes all NPU slices within each `sp.run` chunk (with an `op_inflight` drain guard in `Start*` and `EndChunk()` for cancelled/aborted segments), NPU offload composes safely across `BYIELD` preemptions, `BGEN` slot admissions, and solo-to-slot handoffs (`parallel_solo`).

---

## Future Work & High-Leverage Uses of the XDNA2 NPU in Talus

To keep both `gfx1151` (GPU) and `aie2p` (NPU) utilized across both **output (decode/speculative)** and **input (prefill)** workloads — including native vision and multi-slot batched serving — the following extensions target Talus's remaining bottlenecks.

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
2. **Overlapping NPU Prefill with Interleaved `--batch` Decode (`read_part` + `batch_step()`)**:
   - Today, `read_part` (`src/program/generate.cpp`) alternates serially between running one `sp.run` prompt chunk (GPU+NPU) and running `batch_step()` decode windows (`STRATA_BATCH_DECODE_SHARE`) on GPU while the NPU waits idle.
   - Launching next-chunk NPU staging/projections (or background prefill for a `BYIELD`-parked / queued request) concurrently while GPU executes `batch_step()` keeps both accelerators busy during mixed prefill+decode traffic.
3. **Dense `Qwen3.5` / `Qwen3.8-27B` (`rust/crates/talus-model/src/qwen35/engine.rs`)**:
   - Dense 27B layers have large projections (`K = 5120`, `n_ff = 17408`, plus `attn_qkv`/`attn_gate`).
   - Either splitting `K = 5120` into two `2560` halves using `talus_npu_decode_output(..., accumulate=1)` or compiling a dedicated `K = 5120` `xclbin` allows offloading dense `ffn_gate`/`ffn_up` or GDN `z` slices concurrently with GPU attention and recurrence.
4. **`GLM-5.3-Flash` (`rust/crates/talus-model/src/glm/prefill.rs`)**:
   - `GLM-5.3-Flash` uses `GLM_PCH = 4096` and `n_embd = 2048`.
   - `K_valid = 2048` can be zero-padded directly into the existing `4096x2560x1024` kernel via `talus_npu_encode_activations_f16` / `talus_npu_encode_weights_*` (or served by a dedicated `K = 2048` `xclbin`) to offload `KDA` (`z`/`q`) and shared-expert MLP slices.
5. **Hot MoE Experts via `counts_host_` (`qwen4exp`)**:
   - During `T = 4096` MoE routing, the hottest `10..20` experts per layer often receive `128..512+` tokens.
   - Using `counts_host_` to partition work — running the largest experts on GPU (`WMMA`/`MMQ`) while dispatching medium-batch (`~64..128` token) packed `gate+up` hot experts onto an `M = 128` NPU kernel — can further balance MoE prefill load.

### C. Runtime & Kernel Optimizations

- **Additional `mlir-aie` shapes**: Compile small-`M` (`M = 64, 128, 256`) kernels for speculative verify, `slot_mtp` batching, hot MoE experts, and smaller chunks, plus native `K = 2048`, `K = 5120`, and ViT-width kernels to eliminate zero-padding or two-pass split-K.
- **Idle-gated `keepalive` for decode-heavy `--batch` serving**: Pause the 1s `Heartbeat()` after a short idle grace period when no prefill chunks arrive (allowing `D3hot` power savings during long multi-slot decode phases or server idle, while remaining hot across interleaved `BYIELD` / `batch_read_piece` chunks).
- **Direct BF16 activation encoding (`EncodeActivationsBF16`)**: Add a `BF16` activation encoder for engines/paths whose native 16-bit activation buffer is `BF16` rather than `FP16`.
- **Cross-model weight arena & `MemAvailable` integration**: Add explicit eviction/LRU bounds on cached NPU weight buffers when switching models inside a long-lived `talus-server` process, and account for `PrefillNpu` pinned shared BOs alongside `NativeEncoder::floor_claim()` in `talus_mem::pressure`.
