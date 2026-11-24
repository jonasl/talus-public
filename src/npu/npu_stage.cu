#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace talus::npu {
namespace {

constexpr int kThreads = 256;
constexpr int kNpuTileM = 128;
constexpr int kNpuTileK = 64;
constexpr int kNpuTileN = 128;
constexpr int kNpuTileRowsC = 4 * kNpuTileM;  // 512

struct Q8_0Block {
    __half d;
    std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34);

__device__ __forceinline__ unsigned Blocks(std::size_t items) {
    return static_cast<unsigned>((items + kThreads - 1) / kThreads);
}

__device__ __forceinline__ float SiluF(float x) {
    return x / (1.0f + __expf(-x));
}

// Encodes 8 floats into 9-byte v8bfp16ebs8: [exp_byte, int8 q0..q7].
// Decoded value is q * 2^(exp - 127) / 64 = q * 2^(exp - 133).
// Checks whether bumping exp by +1 avoids positive saturation clipping in [1.984375, 2.0)*2^(e-127)
// and improves L2 reconstruction error over the 8-element group.
template <bool kCheckBump = true>
__device__ __forceinline__ void NpuEncodeGroup(const float* v, std::uint8_t* out) {
    unsigned max_exp = 0;
    float max_pos = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        max_exp = max(max_exp, (__float_as_uint(v[i]) >> 23) & 0xFFu);
        max_pos = fmaxf(max_pos, v[i]);
    }
    if constexpr (kCheckBump) {
        if (max_exp >= 7 && max_exp < 255) {
            const float step0 = __uint_as_float((max_exp - 6u) << 23);  // 2^(max_exp - 133)
            if (max_pos >= 127.5f * step0) {
                const float inv0 = __uint_as_float((260u - max_exp) << 23);
                const float inv1 = inv0 * 0.5f;
                const float step1 = step0 * 2.0f;
                float err0 = 0.0f, err1 = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float q0 = fminf(127.0f, fmaxf(-128.0f, rintf(v[i] * inv0)));
                    const float q1 = fminf(127.0f, fmaxf(-128.0f, rintf(v[i] * inv1)));
                    const float d0 = v[i] - q0 * step0;
                    const float d1 = v[i] - q1 * step1;
                    err0 += d0 * d0;
                    err1 += d1 * d1;
                }
                if (err1 < err0) {
                    max_exp += 1u;
                }
            }
        }
    }
    const float inv = __uint_as_float(max_exp >= 7 ? (260u - max_exp) << 23 : 0u);
    out[0] = static_cast<std::uint8_t>(max_exp);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float q = fminf(127.0f, fmaxf(-128.0f, rintf(v[i] * inv)));
        out[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
    }
}

__global__ void __launch_bounds__(256)
NpuEncodeActivationsKernel(const __half* __restrict__ x, std::uint8_t* __restrict__ out,
                           std::uint32_t valid_tokens, std::uint32_t k_slice, std::uint32_t k_offset,
                           std::uint32_t k_valid, std::size_t ldx) {
    __shared__ __align__(16) __half tile[kNpuTileM * kNpuTileK];
    __shared__ __align__(16) std::uint8_t staged[kNpuTileM * kNpuTileK / 8 * 9];
    const std::uint32_t tiles_k = k_slice / kNpuTileK;
    const std::uint32_t tile_row = blockIdx.x / tiles_k;
    const std::uint32_t tile_col = blockIdx.x % tiles_k;
    const std::uint32_t t = threadIdx.x;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::uint32_t row = (t >> 3) + 32 * j;
        const std::uint32_t vec = t & 7;
        const std::uint32_t col = tile_col * kNpuTileK + vec * 8;
        const std::uint32_t global_row = tile_row * kNpuTileM + row;
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (global_row < valid_tokens && k_offset + col + 8 <= k_valid) {
            raw = *reinterpret_cast<const uint4*>(
                &x[static_cast<std::size_t>(global_row) * ldx + k_offset + col]);
        }
        *reinterpret_cast<uint4*>(&tile[row * kNpuTileK + vec * 8]) = raw;
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::uint32_t gi = t + 256 * j;
        const std::uint32_t rib = gi & 7;
        const std::uint32_t block = (gi >> 3) & 1;
        const std::uint32_t sc = (gi >> 4) & 7;
        const std::uint32_t sr = gi >> 7;
        const std::uint32_t row = sr * 16 + block * 8 + rib;
        const uint4 raw = *reinterpret_cast<const uint4*>(&tile[row * kNpuTileK + sc * 8]);
        const __half* h = reinterpret_cast<const __half*>(&raw);
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = __half2float(h[i]);
        NpuEncodeGroup<false>(v, staged + gi * 9);
    }
    __syncthreads();

    const uint4* src = reinterpret_cast<const uint4*>(staged);
    uint4* dst = reinterpret_cast<uint4*>(
        out + static_cast<std::size_t>(blockIdx.x) * (kNpuTileM * kNpuTileK / 8 * 9));
    for (std::uint32_t i = t; i < kNpuTileM * kNpuTileK / 8 * 9 / 16; i += 256) {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ void DecodeBCoords(std::size_t g, std::uint32_t k_slice, std::uint32_t k_offset,
                                              std::uint32_t& kk, std::uint32_t& nn) {
    constexpr std::uint32_t kGroupsPerTile = kNpuTileK * kNpuTileN / 8;
    const std::size_t tile = g / kGroupsPerTile;
    const std::uint32_t gi = static_cast<std::uint32_t>(g % kGroupsPerTile);
    const std::uint32_t tiles_k = k_slice / kNpuTileK;
    const std::uint32_t tile_col = static_cast<std::uint32_t>(tile / tiles_k);
    const std::uint32_t tile_row = static_cast<std::uint32_t>(tile % tiles_k);
    const std::uint32_t cib = gi & 7;
    const std::uint32_t block = (gi >> 3) & 1;
    const std::uint32_t sr = (gi >> 4) & 7;
    const std::uint32_t sc2 = gi >> 7;
    kk = k_offset + tile_row * kNpuTileK + sr * 8;
    nn = tile_col * kNpuTileN + (sc2 * 2 + block) * 8 + cib;
}

__global__ void NpuEncodeWeightsQ8Kernel(const Q8_0Block* __restrict__ w, std::uint8_t* __restrict__ out,
                                         std::uint32_t row0, std::uint32_t cols, std::uint32_t valid_cols,
                                         std::uint32_t k_slice, std::uint32_t k_offset,
                                         std::uint32_t k_valid, std::uint32_t k_total) {
    const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t total = std::size_t{k_slice} * cols / 8;
    if (g >= total) return;
    std::uint32_t kk = 0, nn = 0;
    DecodeBCoords(g, k_slice, k_offset, kk, nn);
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (nn < valid_cols && kk + 8 <= k_valid) {
        const Q8_0Block& b = w[static_cast<std::size_t>(row0 + nn) * (k_total / 32) + kk / 32];
        const float scale = __half2float(b.d);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = scale * static_cast<float>(b.qs[(kk % 32) + i]);
    }
    NpuEncodeGroup<true>(v, out + g * 9);
}

__global__ void NpuEncodeWeightsQ8PairKernel(const Q8_0Block* __restrict__ wa,
                                             const Q8_0Block* __restrict__ wb,
                                             std::uint8_t* __restrict__ out,
                                             std::uint32_t cols, std::uint32_t valid_cols,
                                             std::uint32_t k_slice, std::uint32_t k_offset,
                                             std::uint32_t k_valid, std::uint32_t k_total) {
    const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t total = std::size_t{k_slice} * cols / 8;
    if (g >= total) return;
    std::uint32_t kk = 0, nn = 0;
    DecodeBCoords(g, k_slice, k_offset, kk, nn);
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (kk + 8 <= k_valid) {
        const Q8_0Block* src = nullptr;
        std::uint32_t r = 0;
        if (nn < valid_cols) {
            src = wa;
            r = nn;
        } else if (nn < 2 * valid_cols) {
            src = wb;
            r = nn - valid_cols;
        }
        if (src != nullptr) {
            const Q8_0Block& b = src[static_cast<std::size_t>(r) * (k_total / 32) + kk / 32];
            const float scale = __half2float(b.d);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = scale * static_cast<float>(b.qs[(kk % 32) + i]);
        }
    }
    NpuEncodeGroup<true>(v, out + g * 9);
}

__global__ void NpuEncodeWeightsF16Kernel(const __half* __restrict__ w, std::uint8_t* __restrict__ out,
                                          std::uint32_t row0, std::uint32_t cols, std::uint32_t valid_cols,
                                          std::uint32_t k_slice, std::uint32_t k_offset,
                                          std::uint32_t k_valid, std::uint32_t k_total) {
    const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t total = std::size_t{k_slice} * cols / 8;
    if (g >= total) return;
    std::uint32_t kk = 0, nn = 0;
    DecodeBCoords(g, k_slice, k_offset, kk, nn);
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (nn < valid_cols && kk + 8 <= k_valid) {
        const __half* row = w + static_cast<std::size_t>(row0 + nn) * k_total + kk;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = __half2float(row[i]);
    }
    NpuEncodeGroup<true>(v, out + g * 9);
}

__global__ void NpuEncodeWeightsBF16Kernel(const std::uint16_t* __restrict__ w, std::uint8_t* __restrict__ out,
                                           std::uint32_t row0, std::uint32_t cols, std::uint32_t valid_cols,
                                           std::uint32_t k_slice, std::uint32_t k_offset,
                                           std::uint32_t k_valid, std::uint32_t k_total) {
    const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t total = std::size_t{k_slice} * cols / 8;
    if (g >= total) return;
    std::uint32_t kk = 0, nn = 0;
    DecodeBCoords(g, k_slice, k_offset, kk, nn);
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (nn < valid_cols && kk + 8 <= k_valid) {
        const std::uint16_t* row = w + static_cast<std::size_t>(row0 + nn) * k_total + kk;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = __uint_as_float(static_cast<std::uint32_t>(row[i]) << 16);
        }
    }
    NpuEncodeGroup<true>(v, out + g * 9);
}

__device__ __forceinline__ std::size_t CGroupIndex(std::uint32_t row, std::uint32_t col, std::uint32_t cols) {
    constexpr std::uint32_t kGroupsPerTile = kNpuTileRowsC * kNpuTileN / 8;
    const std::uint32_t tiles_n = cols / kNpuTileN;
    const std::uint32_t tile_row = row / kNpuTileRowsC;
    const std::uint32_t r = row % kNpuTileRowsC;
    const std::uint32_t tile_col = col / kNpuTileN;
    const std::uint32_t c = col % kNpuTileN;
    const std::uint32_t rib = r & 7;
    const std::uint32_t br = (r >> 3) & 1;
    const std::uint32_t sr = r >> 4;
    const std::uint32_t bc = (c >> 3) & 1;
    const std::uint32_t sc = c >> 4;
    const std::uint32_t gi = rib | (bc << 3) | (br << 4) | (sc << 5) | (sr << 8);
    return std::size_t{tile_row * tiles_n + tile_col} * kGroupsPerTile + gi;
}

__global__ void NpuDecodeOutputKernel(const std::uint8_t* __restrict__ in, float* __restrict__ out,
                                      std::uint32_t tokens, std::uint32_t valid_tokens, std::uint32_t cols,
                                      std::size_t out_pitch, std::uint32_t col0,
                                      bool accumulate, std::uint32_t valid_cols) {
    const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g >= std::size_t{tokens} * cols / 8) return;
    constexpr std::uint32_t kGroupsPerTile = kNpuTileRowsC * kNpuTileN / 8;
    const std::size_t tile = g / kGroupsPerTile;
    const std::uint32_t gi = static_cast<std::uint32_t>(g % kGroupsPerTile);
    const std::uint32_t tiles_n = cols / kNpuTileN;
    const std::uint32_t tile_row = static_cast<std::uint32_t>(tile / tiles_n);
    const std::uint32_t tile_col = static_cast<std::uint32_t>(tile % tiles_n);
    const std::uint32_t rib = gi & 7;
    const std::uint32_t bc = (gi >> 3) & 1;
    const std::uint32_t br = (gi >> 4) & 1;
    const std::uint32_t sc = (gi >> 5) & 7;
    const std::uint32_t sr = gi >> 8;
    const std::size_t row = std::size_t{tile_row} * kNpuTileRowsC + (sr * 2 + br) * 8 + rib;
    const std::uint32_t col = tile_col * kNpuTileN + (sc * 2 + bc) * 8;
    if (row >= valid_tokens || (valid_cols != 0 && col >= valid_cols)) return;

    const std::uint8_t* p = in + g * 9;
    const float mult = ldexpf(1.0f, static_cast<int>(p[0]) - 127) / 64.0f;
    float* dst = out + row * out_pitch + col0 + col;
    if (((out_pitch | col0) & 3) != 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = static_cast<float>(static_cast<std::int8_t>(p[1 + i])) * mult;
            dst[i] = accumulate ? dst[i] + v : v;
        }
        return;
    }
    float4 r[2];
    if (accumulate) {
        r[0] = *reinterpret_cast<const float4*>(dst);
        r[1] = *reinterpret_cast<const float4*>(dst + 4);
    } else {
        r[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        r[1] = r[0];
    }
    float* rf = reinterpret_cast<float*>(r);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float v = static_cast<float>(static_cast<std::int8_t>(p[1 + i])) * mult;
        rf[i] = accumulate ? rf[i] + v : v;
    }
    *reinterpret_cast<float4*>(dst) = r[0];
    *reinterpret_cast<float4*>(dst + 4) = r[1];
}

__global__ void NpuSwigluKernel(const std::uint8_t* __restrict__ gate_in,
                                const std::uint8_t* __restrict__ up_in,
                                __half* __restrict__ out, std::uint32_t tokens,
                                std::uint32_t valid_tokens, std::uint32_t cols,
                                std::uint32_t valid_cols) {
    const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g >= std::size_t{tokens} * cols / 8) return;
    constexpr std::uint32_t kGroupsPerTile = kNpuTileRowsC * kNpuTileN / 8;
    const std::size_t tile = g / kGroupsPerTile;
    const std::uint32_t gi = static_cast<std::uint32_t>(g % kGroupsPerTile);
    const std::uint32_t tiles_n = cols / kNpuTileN;
    const std::uint32_t tile_row = static_cast<std::uint32_t>(tile / tiles_n);
    const std::uint32_t tile_col = static_cast<std::uint32_t>(tile % tiles_n);
    const std::uint32_t rib = gi & 7;
    const std::uint32_t bc = (gi >> 3) & 1;
    const std::uint32_t br = (gi >> 4) & 1;
    const std::uint32_t sc = (gi >> 5) & 7;
    const std::uint32_t sr = gi >> 8;
    const std::size_t row = std::size_t{tile_row} * kNpuTileRowsC + (sr * 2 + br) * 8 + rib;
    const std::uint32_t col = tile_col * kNpuTileN + (sc * 2 + bc) * 8;
    if (row >= valid_tokens || col >= valid_cols) return;

    const std::uint8_t* pg = gate_in + g * 9;
    const std::uint8_t* pu = up_in + g * 9;
    const float mg = ldexpf(1.0f, static_cast<int>(pg[0]) - 127) / 64.0f;
    const float mu = ldexpf(1.0f, static_cast<int>(pu[0]) - 127) / 64.0f;
    __half2 packed[4];
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const float g0 = static_cast<float>(static_cast<std::int8_t>(pg[1 + i])) * mg;
        const float g1 = static_cast<float>(static_cast<std::int8_t>(pg[2 + i])) * mg;
        const float u0 = static_cast<float>(static_cast<std::int8_t>(pu[1 + i])) * mu;
        const float u1 = static_cast<float>(static_cast<std::int8_t>(pu[2 + i])) * mu;
        packed[i / 2] = __floats2half2_rn(SiluF(g0) * u0, SiluF(g1) * u1);
    }
    *reinterpret_cast<uint4*>(out + row * valid_cols + col) = *reinterpret_cast<const uint4*>(packed);
}

__global__ void NpuSwigluPackedKernel(const std::uint8_t* __restrict__ packed_in,
                                      __half* __restrict__ out, std::uint32_t tokens,
                                      std::uint32_t cols, std::uint32_t valid_cols) {
    const std::size_t idx = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t total = std::size_t{tokens} * (valid_cols / 8);
    if (idx >= total) return;
    const std::uint32_t row = static_cast<std::uint32_t>(idx / (valid_cols / 8));
    const std::uint32_t col = static_cast<std::uint32_t>((idx % (valid_cols / 8)) * 8);
    const std::size_t gg = CGroupIndex(row, col, cols);
    const std::size_t gu = CGroupIndex(row, col + valid_cols, cols);
    const std::uint8_t* pg = packed_in + gg * 9;
    const std::uint8_t* pu = packed_in + gu * 9;
    const float mg = ldexpf(1.0f, static_cast<int>(pg[0]) - 127) / 64.0f;
    const float mu = ldexpf(1.0f, static_cast<int>(pu[0]) - 127) / 64.0f;
    __half2 packed[4];
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const float g0 = static_cast<float>(static_cast<std::int8_t>(pg[1 + i])) * mg;
        const float g1 = static_cast<float>(static_cast<std::int8_t>(pg[2 + i])) * mg;
        const float u0 = static_cast<float>(static_cast<std::int8_t>(pu[1 + i])) * mu;
        const float u1 = static_cast<float>(static_cast<std::int8_t>(pu[2 + i])) * mu;
        packed[i / 2] = __floats2half2_rn(SiluF(g0) * u0, SiluF(g1) * u1);
    }
    *reinterpret_cast<uint4*>(out + static_cast<std::size_t>(row) * valid_cols + col) =
        *reinterpret_cast<const uint4*>(packed);
}

}  // namespace

void LaunchEncodeActivationsF16(const std::uint16_t* x_f16, std::size_t ldx,
                                std::uint32_t tokens, std::uint32_t valid_tokens,
                                std::uint32_t k_slice, std::uint32_t k_offset,
                                std::uint32_t k_valid, void* dst, hipStream_t stream) {
    const std::uint32_t vt = valid_tokens == 0 ? tokens : valid_tokens;
    hipLaunchKernelGGL(
        NpuEncodeActivationsKernel,
        dim3((tokens / kNpuTileM) * (k_slice / kNpuTileK)),
        dim3(kThreads), 0, stream,
        reinterpret_cast<const __half*>(x_f16), static_cast<std::uint8_t*>(dst),
        vt, k_slice, k_offset, k_valid, ldx);
}

void LaunchEncodeWeightsQ8(const void* w_q8_0, std::uint32_t k_total, std::uint32_t row0,
                           std::uint32_t cols, std::uint32_t valid_cols, std::uint32_t k_slice,
                           std::uint32_t k_offset, std::uint32_t k_valid, void* dst,
                           hipStream_t stream) {
    const std::size_t groups = std::size_t{k_slice} * cols / 8;
    const unsigned blocks = static_cast<unsigned>((groups + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(
        NpuEncodeWeightsQ8Kernel, dim3(blocks), dim3(kThreads), 0, stream,
        static_cast<const Q8_0Block*>(w_q8_0), static_cast<std::uint8_t*>(dst),
        row0, cols, valid_cols, k_slice, k_offset, k_valid, k_total);
}

void LaunchEncodeWeightsQ8Pair(const void* wa_q8_0, const void* wb_q8_0, std::uint32_t k_total,
                               std::uint32_t cols, std::uint32_t valid_cols, std::uint32_t k_slice,
                               std::uint32_t k_offset, std::uint32_t k_valid, void* dst,
                               hipStream_t stream) {
    const std::size_t groups = std::size_t{k_slice} * cols / 8;
    const unsigned blocks = static_cast<unsigned>((groups + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(
        NpuEncodeWeightsQ8PairKernel, dim3(blocks), dim3(kThreads), 0, stream,
        static_cast<const Q8_0Block*>(wa_q8_0), static_cast<const Q8_0Block*>(wb_q8_0),
        static_cast<std::uint8_t*>(dst),
        cols, valid_cols, k_slice, k_offset, k_valid, k_total);
}

void LaunchEncodeWeightsF16(const std::uint16_t* w_f16, std::uint32_t k_total, std::uint32_t row0,
                            std::uint32_t cols, std::uint32_t valid_cols, std::uint32_t k_slice,
                            std::uint32_t k_offset, std::uint32_t k_valid, void* dst,
                            hipStream_t stream) {
    const std::size_t groups = std::size_t{k_slice} * cols / 8;
    const unsigned blocks = static_cast<unsigned>((groups + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(
        NpuEncodeWeightsF16Kernel, dim3(blocks), dim3(kThreads), 0, stream,
        reinterpret_cast<const __half*>(w_f16), static_cast<std::uint8_t*>(dst),
        row0, cols, valid_cols, k_slice, k_offset, k_valid, k_total);
}

void LaunchEncodeWeightsBF16(const std::uint16_t* w_bf16, std::uint32_t k_total, std::uint32_t row0,
                             std::uint32_t cols, std::uint32_t valid_cols, std::uint32_t k_slice,
                             std::uint32_t k_offset, std::uint32_t k_valid, void* dst,
                             hipStream_t stream) {
    const std::size_t groups = std::size_t{k_slice} * cols / 8;
    const unsigned blocks = static_cast<unsigned>((groups + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(
        NpuEncodeWeightsBF16Kernel, dim3(blocks), dim3(kThreads), 0, stream,
        w_bf16, static_cast<std::uint8_t*>(dst),
        row0, cols, valid_cols, k_slice, k_offset, k_valid, k_total);
}

void LaunchDecodeOutput(const void* src, float* out, std::uint32_t tokens,
                        std::uint32_t valid_tokens, std::uint32_t cols, std::size_t out_pitch,
                        std::uint32_t col0, bool accumulate, std::uint32_t valid_cols,
                        hipStream_t stream) {
    const std::uint32_t vt = valid_tokens == 0 ? tokens : std::min(valid_tokens, tokens);
    const std::uint32_t grid_tokens = std::min(tokens, ((vt + kNpuTileRowsC - 1) / kNpuTileRowsC) * kNpuTileRowsC);
    const std::size_t groups = std::size_t{grid_tokens} * cols / 8;
    const unsigned blocks = static_cast<unsigned>((groups + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(
        NpuDecodeOutputKernel, dim3(blocks), dim3(kThreads), 0, stream,
        static_cast<const std::uint8_t*>(src), out, grid_tokens, vt, cols,
        out_pitch, col0, accumulate, valid_cols);
}

void LaunchSwiglu(const void* gate_c, const void* up_c, std::uint16_t* out_f16,
                  std::uint32_t tokens, std::uint32_t valid_tokens, std::uint32_t cols,
                  std::uint32_t valid_cols, hipStream_t stream) {
    const std::uint32_t vt = valid_tokens == 0 ? tokens : std::min(valid_tokens, tokens);
    const std::uint32_t grid_tokens = std::min(tokens, ((vt + kNpuTileRowsC - 1) / kNpuTileRowsC) * kNpuTileRowsC);
    const std::size_t groups = std::size_t{grid_tokens} * cols / 8;
    const unsigned blocks = static_cast<unsigned>((groups + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(
        NpuSwigluKernel, dim3(blocks), dim3(kThreads), 0, stream,
        static_cast<const std::uint8_t*>(gate_c), static_cast<const std::uint8_t*>(up_c),
        reinterpret_cast<__half*>(out_f16), grid_tokens, vt, cols, valid_cols);
}

void LaunchSwigluPacked(const void* packed_c, std::uint16_t* out_f16,
                        std::uint32_t tokens, std::uint32_t valid_tokens, std::uint32_t cols,
                        std::uint32_t valid_cols, hipStream_t stream) {
    const std::uint32_t vt = valid_tokens == 0 ? tokens : std::min(valid_tokens, tokens);
    const std::size_t groups = std::size_t{vt} * (valid_cols / 8);
    const unsigned blocks = static_cast<unsigned>((groups + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(
        NpuSwigluPackedKernel, dim3(blocks), dim3(kThreads), 0, stream,
        static_cast<const std::uint8_t*>(packed_c),
        reinterpret_cast<__half*>(out_f16), vt, cols, valid_cols);
}

}  // namespace talus::npu
