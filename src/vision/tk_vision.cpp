// src/vision/tk_vision.cpp - see tk_vision.h.  The thinnest C wall over the relocated own/ tower:
// it only owns the Gguf (which the tower may keep referencing after load), the VitGpu, and one
// reused PreparedImage.  All numbers are the tower's own; nothing here rescales or rederives them.
#include "tk_vision.h"

#include "image_prep.h"
#include "vit_geometry.h"
#include "vit_gpu.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

struct tk_vision {
    strata_vit::Gguf gg;
    strata_vit::VitGeometry geo;
    strata_vit::VitGpu gpu;
    // encode_raw's output: the rows land here so a caller that cannot size
    // its buffer before the decode still gets them; replaced by the next encode_raw on this handle.
    std::vector<float> scratch;
};

namespace {

void say(char* err, size_t errlen, const std::string& msg) {
    if (err && errlen) std::snprintf(err, errlen, "%s", msg.c_str());
}

}  // namespace

tk_vision_t* tk_vision_load(const char* mmproj, long max_patches, char* err, size_t errlen) {
    if (!mmproj || !*mmproj) {
        say(err, errlen, "vision: no mmproj path");
        return nullptr;
    }
    std::string why;
    if (!strata_vit::VitGpu::available(why)) {
        say(err, errlen, "vision: no GPU backend (" + why + ")");
        return nullptr;
    }
    auto v = std::make_unique<tk_vision>();
    std::string e;
    if (!v->gg.open(mmproj, e) || !vit_geometry_from_gguf(v->gg, v->geo, e) ||
        !vit_geometry_supported(v->geo, e)) {
        say(err, errlen, "vision: mmproj rejected: " + e);
        return nullptr;
    }
    const long patches = max_patches > 0 ? max_patches : 4096L * 4;
    if (!v->gpu.load(v->gg, v->geo, 0, (int) patches, e)) {
        say(err, errlen, "vision: tower load failed: " + e);
        return nullptr;
    }
    // The reference reader's warm-up (own_reader.cpp): a synthetic largest picture claims the
    // growable work arena at load, so the first real request never pays for the growth.
    strata_vit::ImageU8 gray;
    gray.w = gray.h = 2048;  // 4096 tokens, a multiple of every align()
    gray.rgb.assign((size_t) 2048 * 2048 * 3, 128);
    strata_vit::PreparedImage wp;
    std::vector<float> rows;
    int nx = 0, ny = 0;
    if (!prepare_image(gray, v->geo, v->geo.min_tokens, v->geo.max_tokens, wp, e) ||
        !v->gpu.encode(wp, rows, nx, ny, e)) {
        say(err, errlen, "vision: warm-up failed: " + e);
        return nullptr;
    }
    return (tk_vision_t*) v.release();
}

long tk_vision_info(const tk_vision_t* v, long out8[8]) {
    if (!v || !out8) return 0;
    const strata_vit::VitGeometry& g = v->geo;
    out8[0] = g.out_dim;
    out8[1] = g.layers;
    out8[2] = g.width;
    out8[3] = g.patch;
    out8[4] = g.merge;
    out8[5] = g.align();
    out8[6] = g.min_tokens;
    out8[7] = g.max_tokens;
    return (long) v->gpu.device_bytes();
}

long tk_vision_encode(tk_vision_t* v, const uint8_t* rgb, long w, long h, float* rows, long* nx,
                      long* ny, char* err, size_t errlen) {
    if (!v || !rgb || !rows || w <= 0 || h <= 0) {
        say(err, errlen, "vision: bad encode arguments");
        return -1;
    }
    const int align = v->geo.align();
    if (w % align || h % align) {
        say(err, errlen, "vision: the picture is not a multiple of the align grid (server and "
                        "tower must agree on smart_resize)");
        return -1;
    }
    strata_vit::ImageU8 src;
    src.w = (int) w;
    src.h = (int) h;
    src.rgb.assign(rgb, rgb + (size_t) w * h * 3);
    // The server sent the smart_resize'd pixels; prepare_image's resize is the identity for an
    // aligned, in-bounds picture, so this is the upstream normalisation with no new geometry code.
    strata_vit::PreparedImage prep;
    std::string e;
    if (!prepare_image(src, v->geo, v->geo.min_tokens, v->geo.max_tokens, prep, e)) {
        say(err, errlen, "vision: prepare failed: " + e);
        return -1;
    }
    std::vector<float> out;
    int gx = 0, gy = 0;
    if (!v->gpu.encode(prep, out, gx, gy, e)) {
        say(err, errlen, "vision: encode failed: " + e);
        return -1;
    }
    const size_t n = (size_t) gx * (size_t) gy;
    if (out.size() != n * (size_t) v->geo.out_dim) {
        say(err, errlen, "vision: the tower changed the grid under us");
        return -1;
    }
    std::copy(out.begin(), out.end(), rows);
    if (nx) *nx = gx;
    if (ny) *ny = gy;
    return (long) n;
}

long tk_vision_encode_raw(tk_vision_t* v, const uint8_t* data, long n, const float** rows_out,
                          long* nfloats, long* nx, long* ny, char* err, size_t errlen) {
    if (!v || !data || n <= 0 || !rows_out || !nfloats) {
        say(err, errlen, "vision: bad encode_raw arguments");
        return -1;
    }
    strata_vit::ImageU8 src;
    std::string e;
    if (!strata_vit::image_decode(data, (size_t) n, src, e)) {
        say(err, errlen, "vision: decode failed: " + e);
        return -1;
    }
    strata_vit::PreparedImage prep;
    if (!prepare_image(src, v->geo, v->geo.min_tokens, v->geo.max_tokens, prep, e)) {
        say(err, errlen, "vision: prepare failed: " + e);
        return -1;
    }
    int gx = 0, gy = 0;
    // the tower fills the handle's scratch directly: it is what *rows_out will point at
    if (!v->gpu.encode(prep, v->scratch, gx, gy, e)) {
        say(err, errlen, "vision: encode failed: " + e);
        return -1;
    }
    const size_t rows_n = (size_t) gx * (size_t) gy;
    if (v->scratch.size() != rows_n * (size_t) v->geo.out_dim) {
        say(err, errlen, "vision: the tower changed the grid under us");
        return -1;
    }
    *rows_out = v->scratch.data();
    *nfloats = (long) v->scratch.size();
    if (nx) *nx = gx;
    if (ny) *ny = gy;
    return (long) rows_n;
}

void tk_vision_destroy(tk_vision_t* v) { delete v; }
