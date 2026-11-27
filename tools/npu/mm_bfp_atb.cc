//===- mm_bfp_atb.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Parameterized ATB bfp16ebs8 microkernel (derived from Xilinx/mlir-aie v1.4.5
// programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering/config3).
// C is accumulated in bf16 across `rho` stripes (stripes 0/1 alias the first
// 2*m*n bf16 of the bfp16 C tile, stripes 2/3 live in the TU-local staging
// buffer) and converted in place to bfp16ebs8 after (K_PROBLEMSIZE / k) * rho calls.

#include <aie_api/aie.hpp>

template <typename T, unsigned N, aie_dm_resource R>
using InBufStream = aie::block_vector_restrict_input_buffer_stream<T, N, R>;
template <typename T, unsigned N, aie_dm_resource R>
using OutBufStream = aie::block_vector_output_buffer_stream<T, N, R>;

#ifndef K_PROBLEMSIZE
#define K_PROBLEMSIZE 2560
#endif
#ifndef M_TILE
#define M_TILE 128
#endif
#ifndef N_TILE
#define N_TILE 128
#endif
#ifndef RHO
#define RHO 4
#endif

constexpr int K_Problemsize = K_PROBLEMSIZE;
constexpr int rho = RHO;
constexpr int m = M_TILE / rho;
constexpr int k = 64;
constexpr int n = N_TILE;

constexpr int r = 8;
constexpr int s = 8;
constexpr int t = 8;
constexpr int vec = 64;

static_assert(m * rho == M_TILE && k == 64 && (n % (2 * t)) == 0);
static_assert(rho == 2 || rho == 4);
static_assert((m % (2 * r)) == 0);

#if RHO == 4
alignas(aie::vector_decl_align) bfloat16 c_bf16_2nd_half[m * rho * n / 2];
#endif

extern "C" {

static int g_counter = 0;
static int k_counter = 0;

void matmul_vectorized_bfp16(bfp16ebs8 *__restrict pA_in,
                             bfp16ebs8 *__restrict pB_in,
                             bfp16ebs8 *__restrict pC_bfp16) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  using AStream = InBufStream<bfp16ebs8, r * s, aie_dm_resource::a>;
  using BStream = InBufStream<bfp16ebs8, s * t, aie_dm_resource::b>;
  using CPtr = bfloat16 __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<r, s, t, bfp16ebs8>;

  const bfp16ebs8 *pA_base = (const bfp16ebs8 *)pA_in;

#if RHO == 4
  bfloat16 *const pC_sel = (g_counter < 2)
                               ? (bfloat16 *)pC_bfp16 + g_counter * (m * n)
                               : c_bf16_2nd_half + (g_counter - 2) * (m * n);
#else
  bfloat16 *const pC_sel = (bfloat16 *)pC_bfp16 + g_counter * (m * n);
#endif
  g_counter = (g_counter == rho - 1) ? 0 : g_counter + 1;
  CPtr const pC_base = (CPtr)pC_sel;

  for (int br = 0; br < m / (2 * r); br++) {
    BStream b_stream(pB_in);
    {
      CPtr pC = pC_base + br * (n / (2 * t)) * (4 * r * t);

      aie::vector<bfloat16, r * t> c0 = aie::load_v<r * t>(pC);
      aie::vector<bfloat16, r * t> c1 = aie::load_v<r * t>(pC + r * t);
      for (int blk = 0; blk < n / (2 * t); blk++) {
        CPtr pCb = pC + blk * (4 * r * t);
        CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 4 * r * t : pCb;

        MMUL acc0((aie::accum<accfloat, r * t>(c0)));
        MMUL acc1((aie::accum<accfloat, r * t>(c1)));
        MMUL acc2(
            (aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + 2 * r * t))));
        MMUL acc3(
            (aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + 3 * r * t))));

        c0 = aie::load_v<r * t>(pNext);
        c1 = aie::load_v<r * t>(pNext + r * t);

        AStream a_stream(pA_base);
        a_stream.seek(br * (2 * k / s));

        auto kstep = [&]<bool BFILL>() __attribute__((always_inline)) {
          aie::block_vector<bfp16ebs8, r * s> A0, A1;
          aie::block_vector<bfp16ebs8, s * t> B0, B1;
          if constexpr (BFILL) {
            B0 = b_stream.pop();
            B1 = b_stream.pop();
            A0 = a_stream.pop();
            A1 = a_stream.pop();
          } else {
            A0 = a_stream.pop();
            A1 = a_stream.pop();
            B0 = b_stream.pop();
            B1 = b_stream.pop();
          }
          acc0.mac(A0, aie::op_transpose(B0));
          acc1.mac(A0, aie::op_transpose(B1));
          acc2.mac(A1, aie::op_transpose(B0));
          acc3.mac(A1, aie::op_transpose(B1));
        };
        kstep.operator()<true>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<true>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<false>();

        aie::store_v(pCb, acc0.template to_vector<bfloat16>());
        aie::store_v(pCb + r * t, acc1.template to_vector<bfloat16>());
        aie::store_v(pCb + 2 * r * t, acc2.template to_vector<bfloat16>());
        aie::store_v(pCb + 3 * r * t, acc3.template to_vector<bfloat16>());
      }
    }
  }

  if (k_counter == (K_Problemsize / k) * rho - 1) {
    k_counter = 0;
    OutBufStream<bfp16ebs8, r * t, aie_dm_resource::c> out_stream(pC_bfp16);

#if RHO == 4
    CPtr pC_bf16 = (CPtr)pC_bfp16;
    for (int i = 0; i < m * rho * n / 2 / vec; i++) {
      aie::accum<accfloat, vec> acc_data(aie::load_v<vec>(pC_bf16 + i * vec));
      out_stream.push(acc_data.template to_vector<bfp16ebs8>());
    }
    CPtr pC_2nd = (CPtr)c_bf16_2nd_half;
    for (int i = 0; i < m * rho * n / 2 / vec; i++) {
      aie::accum<accfloat, vec> acc_data(aie::load_v<vec>(pC_2nd + i * vec));
      out_stream.push(acc_data.template to_vector<bfp16ebs8>());
    }
#else
    CPtr pC_bf16 = (CPtr)pC_bfp16;
    for (int i = 0; i < m * rho * n / vec; i++) {
      aie::accum<accfloat, vec> acc_data(aie::load_v<vec>(pC_bf16 + i * vec));
      out_stream.push(acc_data.template to_vector<bfp16ebs8>());
    }
#endif
  } else {
    k_counter += 1;
  }
  event1();
}

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  const aie::vector<bfloat16, vec> zeros = aie::zeros<bfloat16, vec>();
  bfloat16 __aie_dm_resource_c *__restrict p0 =
      (bfloat16 __aie_dm_resource_c *)cOut;
#if RHO == 4
  for (int i = 0; i < m * rho * n / vec / 2; i++)
    aie::store_v(p0 + i * vec, zeros);
  bfloat16 *__restrict p1 = c_bf16_2nd_half;
  for (int i = 0; i < m * rho * n / vec / 2; i++)
    aie::store_v(p1 + i * vec, zeros);
#else
  for (int i = 0; i < m * rho * n / vec; i++)
    aie::store_v(p0 + i * vec, zeros);
#endif
}
}
