// src/vision/selftest.cpp - standalone host for the relocated vision tower.
//
// The GPU path of own_reader.cpp (Strata terminal/tools/vision/own) without the protocol: load the mmproj,
// warm up (claims the work arena at load, as the reference reader does), encode one image, write the SVE1
// record.  The point of this host is to prove compile, hipBLAS/hipBLASLt coexistence with the LLM loaded,
// and record parity against `strata-vision-hip` on the same input - nothing else.
//
//   talus-vision-selftest --mmproj <mmproj.gguf> [--warm|--no-warm] [--cpu-oracle] <image> <out.sve>
//
// --cpu-oracle additionally runs the float32 reference tower (VitRef) on the same picture and reports the
// largest per-element difference - the upstream accuracy floors (vit_gpu.cu header) in one number.
#include "vit_geometry.h"
#include "vit_gpu.h"
#include "vit_ref.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Opts {
    std::string mmproj, image, out;
    bool warm = true;
    bool cpu_oracle = false;
    std::string emit_pixels;  // --emit-pixels <raw>: write the smart_resize'd RGB8 and stop (the bytes
                              // a C1 server sends in the @px frame; the engine-side parity harness feeds them)
};

bool parse(int argc, char** argv, Opts& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mmproj" && i + 1 < argc) o.mmproj = argv[++i];
        else if (a == "--emit-pixels" && i + 1 < argc) o.emit_pixels = argv[++i];
        else if (a == "--warm") o.warm = true;
        else if (a == "--no-warm") o.warm = false;
        else if (a == "--cpu-oracle") o.cpu_oracle = true;
        else if (o.image.empty()) o.image = a;
        else if (o.out.empty()) o.out = a;
        else return false;
    }
    return !o.mmproj.empty() && !o.image.empty() && (!o.out.empty() || !o.emit_pixels.empty());
}

void usage() {
    std::fprintf(stderr, "usage: talus-vision-selftest --mmproj <mmproj.gguf> [--no-warm] [--cpu-oracle] <image> <out.sve>\n");
}

// the SVE1 record of own_reader.cpp:150-151: int32[5]{magic SVE1, n, nx, ny, out_dim} then F32 rows
bool write_sve(const std::string& path, const std::vector<float>& rows, int n, int nx, int ny, int out_dim) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const int32_t hdr[5] = {0x31455653, n, nx, ny, out_dim};
    const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                    std::fwrite(rows.data(), sizeof(float) * (size_t) out_dim, (size_t) n, f) == (size_t) n;
    std::fclose(f);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    Opts o;
    if (!parse(argc, argv, o)) {
        usage();
        return 2;
    }

    strata_vit::Gguf gg;
    strata_vit::VitGeometry geo;
    std::string err;
    if (!gg.open(o.mmproj, err) || !vit_geometry_from_gguf(gg, geo, err) || !vit_geometry_supported(geo, err)) {
        std::fprintf(stderr, "selftest: mmproj rejected: %s\n", err.c_str());
        return 1;
    }

    std::string why;
    if (!o.emit_pixels.empty()) {
        strata_vit::ImageU8 src, resized;
        strata_vit::PreparedImage prep;
        if (!image_load_file(o.image, src, err) ||
            !prepare_image(src, geo, geo.min_tokens, geo.max_tokens, prep, err, &resized)) {
            std::fprintf(stderr, "selftest: emit-pixels: %s\n", err.c_str());
            return 1;
        }
        FILE* f = std::fopen(o.emit_pixels.c_str(), "wb");
        const bool ok = f && std::fwrite(resized.rgb.data(), 1, resized.rgb.size(), f) == resized.rgb.size();
        if (f) std::fclose(f);
        if (!ok) {
            std::fprintf(stderr, "selftest: cannot write %s\n", o.emit_pixels.c_str());
            return 1;
        }
        std::printf("selftest: emitted %ldx%ld = %zu bytes -> %s\n", (long) resized.w, (long) resized.h,
                    resized.rgb.size(), o.emit_pixels.c_str());
        return 0;
    }
    if (!strata_vit::VitGpu::available(why)) {
        std::fprintf(stderr, "selftest: no GPU backend (%s)\n", why.c_str());
        return 3;
    }
    const auto load_t0 = clock::now();
    strata_vit::VitGpu gpu;
    const int tokens = std::min(geo.max_tokens, 4096);
    if (!gpu.load(gg, geo, 0, tokens * 4, err)) {
        std::fprintf(stderr, "selftest: tower load failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("selftest: tower loaded (%s, %d layers, width %d, out %d) in %.1f s, %.0f MiB device\n",
                geo.projector.c_str(), geo.layers, geo.width, geo.out_dim,
                std::chrono::duration<double>(clock::now() - load_t0).count(), (double) gpu.device_bytes() / 1048576.0);

    if (o.warm) {  // the reference reader's move: claim the arena at load, at the largest expected picture
        const auto w0 = clock::now();
        strata_vit::ImageU8 gray;
        gray.w = gray.h = 2048;
        gray.rgb.assign((size_t) 2048 * 2048 * 3, 128);
        strata_vit::PreparedImage wp;
        std::vector<float> rows;
        int nx = 0, ny = 0;
        if (!prepare_image(gray, geo, geo.min_tokens, geo.max_tokens, wp, err) ||
            !gpu.encode(wp, rows, nx, ny, err)) {
            std::fprintf(stderr, "selftest: warm-up failed: %s\n", err.c_str());
            return 1;
        }
        std::printf("selftest: warmed up at %d tokens in %.1f s, %.0f MiB device\n", nx * ny,
                    std::chrono::duration<double>(clock::now() - w0).count(), (double) gpu.device_bytes() / 1048576.0);
    }

    const auto t0 = clock::now();
    strata_vit::ImageU8 src;
    strata_vit::PreparedImage prep;
    if (!image_load_file(o.image, src, err)) {
        std::fprintf(stderr, "selftest: cannot read %s: %s\n", o.image.c_str(), err.c_str());
        return 1;
    }
    if (!prepare_image(src, geo, geo.min_tokens, geo.max_tokens, prep, err)) {
        std::fprintf(stderr, "selftest: prepare failed: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> rows;
    int nx = 0, ny = 0;
    if (!gpu.encode(prep, rows, nx, ny, err)) {
        std::fprintf(stderr, "selftest: encode failed: %s\n", err.c_str());
        return 1;
    }
    const int n = nx * ny;
    const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    if (!write_sve(o.out, rows, n, nx, ny, geo.out_dim)) {
        std::fprintf(stderr, "selftest: cannot write %s\n", o.out.c_str());
        return 1;
    }
    std::printf("selftest: OK %d %d %d %.0f ms -> %s\n", n, nx, ny, ms, o.out.c_str());

    if (o.cpu_oracle) {
        strata_vit::VitRef ref;
        if (!ref.load(gg, geo, err)) {
            std::fprintf(stderr, "selftest: oracle load failed: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> gold;
        int gnx = 0, gny = 0;
        if (!ref.encode(prep, 16, gold, gnx, gny, err)) {
            std::fprintf(stderr, "selftest: oracle encode failed: %s\n", err.c_str());
            return 1;
        }
        if (gnx != nx || gny != ny || gold.size() != rows.size()) {
            std::fprintf(stderr, "selftest: oracle grid mismatch (%dx%d vs %dx%d)\n", gnx, gny, nx, ny);
            return 1;
        }
        double max_abs = 0.0, max_rel_denom = 0.0;
        size_t worst = 0;
        for (size_t i = 0; i < gold.size(); ++i) {
            const double d = std::abs((double) gold[i] - (double) rows[i]);
            if (d > max_abs) { max_abs = d; worst = i; max_rel_denom = std::abs((double) gold[i]); }
        }
        std::printf("selftest: vs float32 oracle: max |diff| %.4g at row %zu (|gold| %.4g)\n", max_abs,
                    worst / (size_t) geo.out_dim, max_rel_denom);
    }
    return 0;
}
