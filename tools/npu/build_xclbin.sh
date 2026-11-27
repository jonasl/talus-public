#!/usr/bin/env bash
set -euo pipefail

# Build tools/npu/talus_npu_aie2p.xclbin and per-shape insts_*.bin using mlir-aie + Peano.
# Usage: MLIR_AIE_VENV=/home/jlo/.venv/mlir-aie ./tools/npu/build_xclbin.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${MLIR_AIE_VENV:-/home/jlo/.venv/mlir-aie}"
if [[ ! -x "${VENV}/bin/aiecc" ]]; then
  echo "error: aiecc not found at ${VENV}/bin/aiecc (set MLIR_AIE_VENV)" >&2
  exit 1
fi

PEANO_DIR="$("${VENV}/bin/python" -c 'import llvm_aie, pathlib; print(pathlib.Path(llvm_aie.__file__).resolve().parent)' 2>/dev/null || echo "${VENV}/lib/python3.12/site-packages/llvm-aie")"
MLIR_AIE_INC="$("${VENV}/bin/python" -c 'import aie, pathlib; print(pathlib.Path(aie.__file__).resolve().parents[2] / "include")')"
export PATH="${VENV}/bin:${PATH}"

WORKDIR="/tmp/opencode/npu_xclbin_build"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

SHAPES=(
  "4096 2560 1024 0x901"
  "2048 2560 1024 0x902"
  "1024 2560 1024 0x903"
  "512  2560 1024 0x904"
)

CUR_XCLBIN=""
for idx in "${!SHAPES[@]}"; do
  read -r M K N KID <<< "${SHAPES[$idx]}"
  KNAME="GEMM_${M}_${K}_${N}"
  OBJ_FILE="${WORKDIR}/mm_bfp_${M}x${K}x${N}_128x64x128_r4.o"
  INST_OUT="${SCRIPT_DIR}/insts_${M}x${K}x${N}.bin"
  TMP_XCL="${WORKDIR}/step_${idx}.xclbin"
  MLIR_FILE="${WORKDIR}/${KNAME}.mlir"
  PRJ_DIR="${WORKDIR}/${KNAME}.prj"

  echo "=== [${idx}] Building ${KNAME} (kernel_id=${KID}) ==="
  "${PEANO_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2p-none-unknown-elf \
    -D__AIE_API_AIE_ADF_HPP__ -DNDEBUG \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -ffunction-sections -fdata-sections -fstack-size-section \
    -I"${MLIR_AIE_INC}" -I"${MLIR_AIE_INC}/aie_kernels" \
    -DK_PROBLEMSIZE="${K}" -DM_TILE=128 -DN_TILE=128 -DRHO=4 \
    -c "${SCRIPT_DIR}/mm_bfp_atb.cc" -o "${OBJ_FILE}"

  "${VENV}/bin/python" "${SCRIPT_DIR}/n32_core_atb.py" \
    --dev npu2 -M "${M}" -K "${K}" -N "${N}" -m 128 -k 64 -n 128 --rho 4 \
    --emit-mlir > "${MLIR_FILE}"

  XCL_IN_ARGS=()
  if [[ -n "${CUR_XCLBIN}" ]]; then
    XCL_IN_ARGS=(--xclbin-input="${CUR_XCLBIN}")
  fi

  "${VENV}/bin/aiecc" \
    --peano="${PEANO_DIR}" \
    --dynamic-objFifos \
    --tmpdir="${PRJ_DIR}" \
    --get-xclbin --get-npu-insts \
    "${XCL_IN_ARGS[@]}" \
    --xclbin-name="${TMP_XCL}" \
    --xclbin-kernel-name="${KNAME}" \
    --xclbin-instance-name="${KNAME}" \
    --xclbin-kernel-id="${KID}" \
    --npu-insts-name="${INST_OUT}" \
    "${MLIR_FILE}"

  CUR_XCLBIN="${TMP_XCL}"
done

cp "${CUR_XCLBIN}" "${SCRIPT_DIR}/talus_npu_aie2p.xclbin"
ls -lh "${SCRIPT_DIR}/talus_npu_aie2p.xclbin" "${SCRIPT_DIR}"/insts_*x*x*.bin
