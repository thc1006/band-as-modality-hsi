#!/usr/bin/env bash
# scripts/gpu_env.sh — adaptive CUDA driver-library fix (source this, don't execute).
#
# Problem this solves
# -------------------
# On some hosts/containers, LD_LIBRARY_PATH prepends a CUDA "forward-compat" libcuda
# (e.g. NGC PyTorch images ship /usr/local/cuda/compat/lib with a libcuda NEWER than the
# host's kernel driver). If that compat libcuda cannot drive the installed kernel module,
# every CUDA context creation fails with:
#     RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
# even though `nvidia-smi` shows the GPUs idle and accessible.
#
# Observed here: kernel module = 535.161.08, compat libcuda = 595.58.03 ->
# cuDevicePrimaryCtxRetain() = 999 (CUDA_ERROR_UNKNOWN). The container records this in the
# env var _CUDA_COMPAT_STATUS.
#
# Fix: make libcuda.so.1 resolve to the REAL host driver by prepending its directory and
# dropping the compat directory from LD_LIBRARY_PATH. This is CONDITIONAL — it only acts
# when the compat driver is actually broken (or when forced) — so it is a no-op on a
# healthy machine (e.g. the original RTX 3060 dev box) and safe to source unconditionally.
#
# Override knobs:
#   BANDSIM_FORCE_SYSDRIVER=1   force the fix even if _CUDA_COMPAT_STATUS looks OK
#   BANDSIM_SKIP_GPU_ENV=1      skip entirely (leave LD_LIBRARY_PATH untouched)

_bandsim_gpu_env() {
    [ -n "${BANDSIM_SKIP_GPU_ENV:-}" ] && return 0

    # Decide whether the forward-compat driver is broken and needs bypassing.
    local need_fix=0
    case "${_CUDA_COMPAT_STATUS:-}" in
        ""|"OK"|"Success"|"success"|"Enabled"|"enabled") : ;;   # healthy / absent -> leave alone
        *) need_fix=1 ;;                                         # any error text -> bypass compat
    esac
    [ -n "${BANDSIM_FORCE_SYSDRIVER:-}" ] && need_fix=1
    [ "$need_fix" -eq 0 ] && return 0

    # Locate the real host driver's libcuda (matches /proc/driver/nvidia/version).
    local sysdir=""
    local d
    for d in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do
        if ls "$d"/libcuda.so.[0-9]* >/dev/null 2>&1; then sysdir="$d"; break; fi
    done
    [ -z "$sysdir" ] && return 0   # no host libcuda found; don't touch anything

    # Drop any */cuda*/compat/* entries, then prepend the real driver dir.
    local cleaned
    cleaned="$(printf '%s' "${LD_LIBRARY_PATH:-}" \
        | tr ':' '\n' \
        | grep -vE '/cuda(-[0-9.]+)?/compat(/|$)' \
        | grep -v '^$' \
        | paste -sd: )"
    export LD_LIBRARY_PATH="${sysdir}${cleaned:+:$cleaned}"
    export BANDSIM_GPU_ENV_FIXED=1
    if [ -z "${BANDSIM_GPU_ENV_QUIET:-}" ]; then
        echo "[gpu_env] bypassing broken CUDA forward-compat driver -> using host libcuda in $sysdir" >&2
    fi
}
_bandsim_gpu_env
unset -f _bandsim_gpu_env
