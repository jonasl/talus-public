# Native Vision Support (`tk_vision` & `talus-server`)

Talus Turbo supports multimodal image inputs on OpenAI- and Anthropic-compatible API endpoints through an in-engine GPU vision tower (`src/vision/`, exported as `tk_vision_*` in `libtalus_kernels.so`), with an optional fallback to an external `strata-vision` encoder subprocess.

Both modes share the same server intake, image-token splicing, and `img_keys` conversation-cache identity. In **native mode** (`"vision": {"native": true, ...}`), images are sent to the engine as raw encoded bytes (`@raw` frames) and decoded, resized, and projected directly on the GPU inside `talus-cli`—eliminating temporary `.img`/`.sve` files, IPC round-trips, and subprocess management.

---

## 1. Usage Manual

### 1.1 Run Config (`talus-server`)

Add a `"vision"` object to your `talus-server` JSON config (`rust/crates/talus-server/src/config.rs`, `rust/crates/talus-server/src/vision.rs`). When `"vision"` is present, `talus-server` automatically passes `--vision` to the engine if it is not already in `args`.

#### Native mode (recommended)

```json
{
  "exe": "rust/target/release/talus-cli",
  "args": [
    "--native", "/models/Qwen3.8-Flash-Next/model.gguf",
    "--batch", "4",
    "--batch-mtp"
  ],
  "tokenizer": "/models/Qwen3.8-Flash-Next/tokenizer",
  "model_name": "qwen3.8-flash-next",
  "vision": {
    "native": true,
    "mmproj": "/models/Qwen3.8-Flash-Next/mmproj-F16.gguf"
  }
}
```

When `"native": true`:
- `exe` is not required (`strata-vision` is never spawned).
- `talus-server` injects `TALUS_VISION_MM=<mmproj>` and `TALUS_FN_RUST_VISION=1` into the engine environment (unless overridden in `"env"`).
- `talus-server` preserves `--batch`, `--slots`, and `--batch-mtp` flags in native mode: `talus-server`'s batch-slot scheduler (`rust/crates/talus-server/src/batch.rs`, `engine.rs`) has full support for native vision (`BGENI ... @raw`), and both single-session (`--batch 0`) and batched (`--batch N`) engine paths encode `@raw` frames via `tk_vision`. (Only the legacy Python `serve/server.py` strips `--batch`/`--slots`/`--batch-mtp` in native mode.)

#### Subprocess mode (`strata-vision`)

If `"native"` is omitted or `false`, `talus-server` spawns an external `strata-vision` / `strata-vision-hip` child process and exchanges SVE1 embedding files on disk:

```json
{
  "vision": {
    "native": false,
    "exe": "/path/to/strata-vision-hip",
    "mmproj": "/models/Qwen3.8-Flash-Next/mmproj-F16.gguf",
    "gpu": true,
    "cache_entries": 64
  }
}
```

#### `"vision"` configuration reference

| Field | Default | Applies to | Description |
|---|---|---|---|
| `native` | `false` | Both | `true` = in-engine `tk_vision` tower (`@raw`); `false` = `strata-vision` subprocess |
| `mmproj` | *required* | Both | Path to the GGUF vision projector (`out_dim` must equal the model's `n_embd`) |
| `exe` | *required if `!native`* | Subprocess | Path to `strata-vision` / `strata-vision-hip` binary |
| `min_tokens` | tower / child default | Both | Override minimum visual tokens per image for `smart_resize` |
| `max_tokens` | tower / child default | Both | Override maximum visual tokens per image for `smart_resize` |
| `cache_entries` | `64` | Both | LRU entry cap: grid geometry cache in native mode; `.sve` embedding cache in subprocess mode |
| `enc_timeout_secs` | `120` | Both | Timeout for `curl` image URL fetches (and per-`ENC` timeout in subprocess mode) |
| `gpu` | `false` | Subprocess | Pass `--gpu` to `strata-vision` child process |
| `threads` | unset | Subprocess | Thread count for CPU `strata-vision` |
| `ready_timeout_secs` | `300` | Subprocess | Startup `READY` timeout for the child process |
| `lib_dirs` | `[]` | Subprocess | Extra `LD_LIBRARY_PATH` entries for the child process |
| `cache_dir` | `<temp>/talus-vision-<pid>` | Subprocess | Directory for `.img` and `.sve` files |
| `model` | unset | Compat | Accepted and ignored (kept for `serve/server.py` config compatibility) |

### 1.2 API Intake & Supported Formats

When `"vision"` is configured, `/v1/models` advertises `capabilities.images: true` and the server accepts image parts on:

- **OpenAI Chat Completions (`/v1/chat/completions`)**: `{"type": "image_url", "image_url": {"url": "..."}}` or `{"type": "image_url", "image_url": "..."}`.
- **OpenAI Responses (`/v1/responses`)**: `{"type": "input_image", "image_url": ...}`.
- **Anthropic Messages (`/v1/messages`)**: `{"type": "image", "source": {"type": "base64", "data": "..."}}` or `{"type": "image", "source": {"type": "url", "url": "..."}}`.

Supported image sources and formats:
- **Sources**: `data:` URLs (standard or URL-safe base64, padding optional), `http://` / `https://` URLs (fetched via `curl`), `file://` URIs, and local file paths.
- **Image containers**: **JPEG**, **PNG**, **BMP**, and **GIF** (first frame), decoded via vendored `stb_image`. **WebP** is explicitly rejected at intake with an HTTP 400 error explaining the supported formats.
- **When vision is not configured**: Any image, audio, or file content part is rejected with HTTP 400 (`code: "unsupported_content"` on OpenAI routes; `invalid_request_error` on `/v1/messages`).
- **Frame size limit**: `talus-cli`'s stdin reader accepts `@raw` binary frames up to `MAX_FRAME_BYTES` (`12 MiB`).

### 1.3 Engine Environment Variables

#### Core engine & diagnostics

| Variable | Default | Description |
|---|---|---|
| `TALUS_VISION_MM` | unset | Path to `mmproj` GGUF for the in-engine native tower (set automatically by `talus-server` when `vision.native: true`) |
| `TALUS_FN_RUST_VISION` | unset | `1` = allow the single-session Rust serve loop (`--batch 0`) to handle `--vision` `GENI` requests |
| `TALUS_VISION_MAX_ROWS` | `16384` | Maximum total image token rows accepted per request by `NativeEncoder` |
| `TALUS_VISION_DUMP` | unset | Directory path; writes each native encode's output as `<16-hex-key>.sve` (SVE1 format) for parity debugging |
| `TALUS_VISION_MOCK` | unset | Directory of `<16-hex-key>.sve` files for `MockEncoder` (test seam for `@px` requests without running the GPU tower) |
| `TALUS_VISION_MOCK_MAX_ROWS` | `16384` | Per-request row cap for `MockEncoder` |
| `TALUS_FN_VISION_SAY` | unset | Logs `GENI` parse verdicts (`vision/geni/pixels/emb/frame`) on the Rust serve loop |

#### GPU tower tuning (`src/vision/vit_gpu.cu`)

| Variable | Default | Description |
|---|---|---|
| `STRATA_VIT_FA` | `1` on `gfx1151` | `1` = fused WMMA flash attention (`k_attn_fa_gfx11`), `0` = batched score-matrix attention |
| `STRATA_VIT_BF16_AS_F16` | auto (`1` on `gfx1151`) | Hold BF16 projector weights as F16 for `hipBLAS` GEMMs (`0` keeps BF16) |
| `STRATA_VIT_HPA16` | `1` on `gfx1151` | Use F16 GEMM outputs with F32 accumulation (rocBLAS `gfx1151` fast path) |
| `STRATA_VIT_HPA16_ATT` | `0` | Enable HPA16 on non-FA batched attention GEMMs |
| `STRATA_VIT_LT_SOL` | unset | Override hipBLASLt solution selection index when non-default |
| `STRATA_VIT_PROF` | unset | Print per-encode tower timing and retry diagnostics |

### 1.4 Building, Updating, and Testing

`strata_vision` (`src/vision/*.cpp`, `src/vision/vit_gpu.cu`) is compiled automatically when HIP or CUDA is enabled in `CMakeLists.txt` and whole-archived into `libtalus_kernels.so` so `tk_vision_*` symbols are always exported.

- **Full build**:
  ```bash
  ./build.sh
  ```
- **Updating `libtalus_kernels.so` next to `talus-cli`**: Never overwrite a live shared library in place with `cp` (truncating a mapped `.so` inode crashes running engines). Always replace atomically with `install`:
  ```bash
  install -m 755 build/libtalus_kernels.so rust/target/release/
  ```
- **Standalone tower selftest & CPU parity check**:
  ```bash
  ./build/talus-vision-selftest --mmproj /path/to/mmproj.gguf --cpu-oracle image.png out.sve
  ```
  `--cpu-oracle` compares GPU output against the F32 reference tower (`VitRef`) and reports the maximum elementwise delta; `--no-warm` skips the `2048x2048` startup warm-up; `--emit-pixels <raw>` writes the `smart_resize`'d RGB8 buffer.
- **Server load & interference harness**:
  ```bash
  python3 tools/vision_bench.py smoke
  python3 tools/vision_bench.py suite --reps 40 --json-out artifacts/vb.json
  ```

---

## 2. Architecture & Design

### 2.1 End-to-End Request Flow

```
Client (OpenAI / Anthropic JSON)
  │
  ▼
talus-server (frontend.rs → server.rs → vision.rs → batch.rs / engine.rs)
  ├─ Decode data URL / fetch URL (curl) / read file; sniff JPEG/PNG/BMP/GIF (reject WebP)
  ├─ Hash image bytes → 16-hex SHA-256 prefix (`img_keys`)
  ├─ Query `VGEOM` once per engine process → `(align, min_tokens, max_tokens)`
  ├─ Predict resized grid via `smart_resize` → expand `<|vision_start|><|image_pad|>...<|vision_end|>`
  └─ Schedule solo (`GENI`) or batch slot (`BGENI`) with `img_keys=... img_px=... img_bytes=... @raw` + raw bytes
  │
  ▼
talus-cli stdin reader (rust/crates/talus-model/src/qwen4exp/vision.rs)
  ├─ `frame_len()` / `read_frame()` reads exact `sum(img_bytes)` after `GENI` or `BGENI`
  └─ Encodes `@raw` via `tk_vision`:
       ├─ Single-session (`--batch 0`, `engine.rs`): `NativeEncoder::encode_raw` → `DevicePictures` (`PictureRows::Device`)
       └─ Batched (`--batch N`, `generate.cpp` slot scheduler + Rust `prefill.rs`): `take_frame` → `tk_vision_encode_raw` → `sp.embd_rows`
  │
  ▼
LLM Prefill (`prefill.rs` / `prefill.cpp`)
  └─ Merges image rows into `m.emb` at `<|image_pad|>` positions, applies M-RoPE, runs hybrid GPU/NPU prefill
```

### 2.2 Single-Sourced Geometry (`VGEOM` & `smart_resize`)

To place the exact number of `<|image_pad|>` tokens and compute conversation-cache prefix lengths *before* submitting to the engine, `talus-server` needs the projector's grid alignment (`align = patch * merge`) and token bounds (`min_tokens`, `max_tokens`):

1. Once per engine process, `StdioSubprocessEngine::vgeom()` sends `VGEOM\n` between requests.
2. Whichever engine loop is active (single-session `engine.rs` or `--batch` `generate.cpp`) responds with `VGEOM <align> <min_tokens> <max_tokens>` read from `tk_vision_info` (or `ERR VGEOM: ...` if no native tower is configured).
3. `talus-server/src/vision.rs` parses the image header dimensions (`image_size`) and runs `smart_resize`, an exact port of `src/vision/image_prep.cpp`'s arithmetic.
4. `talus-server` sends both the predicted pixel dimensions (`img_px=w1xh1,...`) and the raw compressed image bytes (`img_bytes=n1,... @raw`). Inside the engine, `NativeEncoder::encode_raw` verifies that the tower's prepared grid `(nx * align, ny * align)` matches `img_px` and fails explicitly on any geometry drift.

Because geometry is queried from the engine and validated per request, changing `mmproj` never requires hardcoding projector bounds in `talus-server`.

### 2.3 Wire Protocol (`@raw`, `@px`, and `SVE1`)

The engine stdin protocol (`rust/crates/talus-model/src/qwen4exp/cli.rs`, `vision.rs`, `src/program/generate.cpp`) supports three image carrier forms on `GENI` and `BGENI`:

- **`@raw` (native production path)**:
  ```
  GENI <max_new> [opts...] img_keys=<hex,...> img_px=<WxH,...> img_bytes=<len,...> @raw <token_ids>
  <raw encoded bytes of image 1><raw encoded bytes of image 2>...
  ```
  `BGENI <slot> <max_new> ... @raw <token_ids>` uses the same trailing binary frame. `frame_len()` recognizes both `GENI ` and `BGENI ` so the Rust stdin reader consumes the binary payload before the engine's batch scheduler rewrites `BGENI` into `GENI` for admission.
- **`@px` (in-process resized RGB8 path)**:
  Carries `img_px=<WxH,...> @px` followed by back-to-back row-major `RGB8` bytes (`sum(W * H * 3)`). Supported by `tk_vision_encode` and `MockEncoder` (`TALUS_VISION_MOCK`).
- **SVE1 file path (subprocess path)**:
  Points to a combined SVE1 file written by `strata-vision`. Each record starts with `int32_t[5]{0x31455653 ("SVE1"), n, nx, ny, out_dim}` followed by `n * out_dim` row-major `float32` embeddings. `TALUS_VISION_DUMP` writes the exact same format so native output can be diffed bit-for-bit against `strata-vision` / `talus-vision-selftest`.

### 2.4 Batching & Engine Execution Paths

The Rust server (`talus-server`) provides full continuous batching (`--batch` / `--slots`, `--batch-mtp`) for native vision via `rust/crates/talus-server/src/batch.rs` and `engine.rs`:
- `talus-server` routes `BT`/`BDONE` per slot on a dedicated reader thread, selects slots by held conversation prefix and group economics (`pick_slot`), admits vision requests with `BGENI ... @raw` + binary frame, and supports `parallel_solo` hand-overs and `BYIELD` mid-read preemption alongside text requests.
- Because `BPREFILL` (zero-copy slot-resident prefill) is currently restricted to requests without image embeddings (`req.embeddings.is_none()`), `talus-server` admits vision requests on batch slots via `BGENI`, while concurrent text slots continue decoding between/alongside admissions.

Inside the engine (`talus-cli`), `@raw` encoding works in both execution configurations:

- **Batched engine (`--batch N`, `src/program/generate.cpp` + Rust `stdin_reader` & `prefill.rs`)**:
  - When `o.batch > 0`, the engine runs the `--batch` slot scheduler in `generate.cpp`, while the Rust `stdin_reader` owns stdin framing (`read_frame()` / `take_frame()` for both `GENI` and `BGENI`) and Rust `prefill.rs` runs prompt prefill.
  - `vision_native_ensure()` lazily calls `tk_vision_load` on the first `VGEOM` or `@raw` request and checks `out_dim == n_embd`. `generate.cpp` slices each picture by `img_bytes`, calls `tk_vision_encode_raw` directly, copies rows out of `tk_vision`'s per-handle scratch before the next picture overwrites it, and feeds `PictureRows::HostTable` (`sp.embd_rows`) into `prefill.rs`.
- **Single-session engine (`--batch 0`, `rust/crates/talus-model/src/qwen4exp/engine.rs`)**:
  - When `o.batch <= 0` and `TALUS_FN_RUST_VISION=1`, `generate.cpp` hands the serve loop to `cb_run_serve_loop` in `talus-model/src/qwen4exp/engine.rs`.
  - `NativeEncoder` implements the `ImageEncoder` trait, loading `tk_vision_*` via `dlsym(RTLD_DEFAULT, ...)` at startup. `encode_raw()` invokes `tk_vision_encode_raw`, uploads the resulting `float32` rows to `hipMalloc` buffers, records the completion `hipEvent_t` on the request stream, and passes device pointers (`PictureRows::Device` in `picture.rs`) directly to `prefill.rs` (`release()` synchronizes and frees the buffers after the prompt pass).
  - `NativeEncoder::floor_claim()` adds `round_up_256MiB(device_bytes) + 512 MiB` to `talus_mem::pressure`'s `MemAvailable` admission floor so unified memory reservations account for the resident tower and host working set.

### 2.5 C ABI (`tk_vision`) & Relocated Vision Tower (`src/vision/`)

`src/vision/tk_vision.{h,cpp}` exposes five functions from `libtalus_kernels.so`:

- `tk_vision_load(mmproj, max_patches, err, errlen)`: Opens the GGUF (`gguf_lite.cpp`), validates projector geometry (`vit_geometry.cpp`), uploads weights to GPU (`VitGpu::load`), and executes a synthetic `2048x2048` (`4096`-token) warm-up encode so the growable GPU work arena is allocated at load rather than on the first request.
- `tk_vision_info(v, out8)`: Returns `(out_dim, layers, width, patch, merge, align, min_tokens, max_tokens)` and total device bytes held (weights + arena).
- `tk_vision_encode(v, rgb, w, h, rows, nx, ny, err, errlen)`: Normalizes and encodes pre-resized `RGB8` pixels.
- `tk_vision_encode_raw(v, data, n, rows_out, nfloats, nx, ny, err, errlen)`: Decodes JPEG/PNG/BMP/GIF via `stb_image`, applies `smart_resize` + bicubic resize + normalization (`image_prep.cpp`), runs `VitGpu::encode`, and returns a pointer into handle-owned scratch.
- `tk_vision_destroy(v)`: Releases the tower, weights, and device arena.

#### Supported tower family & execution pipeline

`vit_geometry_supported()` targets the **`qwen3vl_merger`** projector family (`LayerNorm`, `GELU-tanh`, fused `attn_qkv`, non-gated FFN, `merge = 2`, `window_pattern = 0`, no deepstack outputs, `head_dim % 4 == 0` with `kVitGpuMaxHeadDim = 96` in `vit_gpu.h`).

For each picture, `src/vision/` executes:
1. **Host image preparation (`image_prep.cpp`)**: `stb_image` RGB8 decode → HF `smart_resize` with Pillow-compatible 22-bit fixed-point bicubic resampling (`a = -0.5`) → `PAD_CEIL` canvas padding → planar CHW `float32` normalization.
2. **Patch embedding (`vit_gpu.cu`)**: Rearranges patches into `2x2` spatial merge order (`dy`-outer, `dx`-inner; `2` temporal patch taps summed at load time) and runs a `[N, 768] x [768, width]` GEMM + bias + bilinear-resized learned position table (`v.position_embd.weight`).
3. **Transformer blocks**: `layers` pre-norm blocks (`LayerNorm` → fused QKV GEMM + bias → 2D half-split RoPE with `1 / sqrt(head_dim)` scaling → tiled WMMA flash attention `k_attn_fa_gfx11` on `gfx1151` or batched attention → `attn_out` + bias + residual → `LayerNorm` → `ffn_up` → `GELU-tanh` → `ffn_down` + residual). Weights are stored as `BF16` or `F16` (`Q8_0` unpacked to `F16`, `F32` converted to `BF16`; on `gfx1151`, `HPA16` computes `hipBLAS` products with 16-bit outputs and `F32` accumulation).
4. **Post-norm & spatial merger**: `v.post_ln`, then concatenates each `2x2` block (`4 * width` features) through `mm.0` → `GELU-tanh` → `mm.2`, emitting `n = (w / align) * (h / align)` row-major `float32` embeddings of width `out_dim` for M-RoPE placement and LLM prefill.

### 2.6 Caching Strategy

Native mode intentionally omits a server- or engine-side embedding-row LRU:
- `talus-server/src/vision.rs` caches only the predicted `(n, nx, ny)` grid per `img_keys` (capped by `cache_entries`), avoiding repeated `smart_resize` work across turns.
- Conversation and KV-prefix caching in the LLM engine already keys on `img_keys`, so repeated turns or cached prefixes skip re-running LLM prefill on those image positions without retaining large `float32` row buffers across requests.
