#!/usr/bin/env bash
set -euo pipefail

# Hold or release Strix Halo GPU fabric clock (pp_dpm_fclk) to prevent XDNA2 NPU
# firmware hangs during concurrent GPU + NPU execution.
#
# Usage:
#   sudo ./tools/npu/fabric-clock.sh hold
#   sudo ./tools/npu/fabric-clock.sh release
#   ./tools/npu/fabric-clock.sh status

DEV_DIR=""
for card in /sys/class/drm/card*/device; do
    if [[ -e "${card}/pp_dpm_fclk" ]]; then
        DEV_DIR="${card}"
        break
    fi
done

if [[ -z "${DEV_DIR}" ]]; then
    echo "error: no amdgpu device with pp_dpm_fclk found under /sys/class/drm" >&2
    exit 1
fi

NPU_PMODE="/sys/class/accel/accel0/device/power_mode"
CMD="${1:-status}"

case "${CMD}" in
    hold)
        if [[ "${EUID}" -ne 0 ]]; then
            echo "error: 'hold' requires root (run with sudo)" >&2
            exit 1
        fi
        mapfile -t LEVELS < "${DEV_DIR}/pp_dpm_fclk"
        if [[ "${#LEVELS[@]}" -eq 0 ]]; then
            echo "error: empty ${DEV_DIR}/pp_dpm_fclk" >&2
            exit 1
        fi
        TOP_IDX=$(( ${#LEVELS[@]} - 1 ))
        echo manual > "${DEV_DIR}/power_dpm_force_performance_level"
        echo "${TOP_IDX}" > "${DEV_DIR}/pp_dpm_fclk"
        if [[ -w "${NPU_PMODE}" ]]; then
            CUR_PMODE=$(tr -d '[:space:]' < "${NPU_PMODE}" || true)
            if [[ "${CUR_PMODE}" == "turbo" ]]; then
                echo "note: switching NPU power_mode from turbo to performance (turbo is unstable under concurrent load)"
                echo performance > "${NPU_PMODE}" || true
            fi
        fi
        echo "held fclk at level ${TOP_IDX} (${LEVELS[${TOP_IDX}]})"
        ;;
    release)
        if [[ "${EUID}" -ne 0 ]]; then
            echo "error: 'release' requires root (run with sudo)" >&2
            exit 1
        fi
        echo auto > "${DEV_DIR}/power_dpm_force_performance_level"
        echo "released fclk to auto"
        ;;
    status)
        echo "drm device: ${DEV_DIR}"
        echo "power_dpm_force_performance_level: $(cat "${DEV_DIR}/power_dpm_force_performance_level")"
        echo "pp_dpm_fclk:"
        sed 's/^/  /' "${DEV_DIR}/pp_dpm_fclk"
        if [[ -r "${NPU_PMODE}" ]]; then
            echo "npu power_mode: $(cat "${NPU_PMODE}")"
        fi
        ;;
    *)
        echo "usage: $0 {hold|release|status}" >&2
        exit 2
        ;;
esac
