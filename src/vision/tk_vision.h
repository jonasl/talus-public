// src/vision/tk_vision.h - C ABI for the vision tower embedded in libtalus_kernels.so.
//
// The Strata vision tower (vit_gpu.h, relocated from its tools/ tree) behind a stable
// `tk_vision_*` group, exported from libtalus_kernels.so so the Rust host (talus-model's
// NativeEncoder) can resolve it with dlsym (RTLD_DEFAULT) exactly as the other tk_* groups work.
// The Rust side owns the ImageEncoder contract: these calls are synchronous from its point of
// view (the tower reads its results back to the host itself), and the device rows / stream /
// event discipline lives above this header.
//
// Two ways to feed a picture: tk_vision_encode takes pixels the caller already resized to the
// tower's smart_resize'd size (sides multiples of `patch * merge` from tk_vision_info, which the
// encode checks and refuses with an explicit error when they are not), and tk_vision_encode_raw
// takes the picture's encoded bytes and applies the tower's own decode and resize to them.
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tk_vision tk_vision_t;

// `err` is a char[TK_VISION_ERRLEN] buffer owned by the caller, filled on failure.
#define TK_VISION_ERRLEN 256

// Loads the mmproj, checks the geometry, uploads the weights, and warms up the work arena with a
// synthetic 4096-token picture (as the reference reader does, so a real first request never pays
// for buffer growth).  `max_patches` sizes the arena (0 = 4096 tokens' worth).  NULL + `err` on
// failure.  Needs a HIP/CUDA build of the tower (this library always has one).
tk_vision_t* tk_vision_load(const char* mmproj, long max_patches, char* err, size_t errlen);

// Fills out8[8] with: out_dim (n_embd), layers, width (patch embed width), patch, merge, align
// (= patch * merge), min_tokens, max_tokens; returns the device bytes held (weights + arena).
long tk_vision_info(const tk_vision_t* v, long out8[8]);

// Encodes one already-resized RGB8 picture (host memory, w*h*3 bytes).  `rows` must hold
// (w/align) * (h/align) * out_dim floats; on success *nx = w/align, *ny = h/align, and the tower's
// rows are written to it in merge order (the SVE1 record's layout, minus the header).
// Returns the row count n = nx * ny, or -1 with `err` filled (a side not a multiple of align, the
// tower's own failure, or a size the tower's head-dim limit rejects).
long tk_vision_encode(tk_vision_t* v, const uint8_t* rgb, long w, long h, float* rows, long* nx,
                      long* ny, char* err, size_t errlen);

// Encode one RAW picture (the `@raw` wire form): the tower's own stb_image decodes `data`
// (JPEG, PNG, BMP, GIF first frame), and the smart_resize / pad / normalise is the tower's, so no
// size agreement is needed anywhere upstream; the caller may not know the size before the call.
// The rows land in a handle-owned scratch (valid until the next encode_raw on this handle - the
// Rust host copies or uploads it immediately); *rows_out points into it and *nfloats =
// nx * ny * out_dim.  Returns n = nx * ny, or -1 with `err` filled (undecodable bytes or the
// tower's own failure).
long tk_vision_encode_raw(tk_vision_t* v, const uint8_t* data, long n, const float** rows_out,
                          long* nfloats, long* nx, long* ny, char* err, size_t errlen);

// Frees the tower (device weights and arena included).  NULL is a no-op.
void tk_vision_destroy(tk_vision_t* v);

#ifdef __cplusplus
}  // extern "C"
#endif
