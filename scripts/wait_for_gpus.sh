#!/usr/bin/env bash
# Wait for physical GPUs to be idle, then execute a command without eval.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  wait_for_gpus.sh --gpus CSV [OPTIONS] -- COMMAND [ARG ...]

Options:
  --gpus CSV                Physical GPU indices, e.g. 0 or 0,1,2,3.
  --poll-seconds N          Delay between checks. Default: 30.
  --memory-threshold-mib N  Idle memory threshold. Default: 2048.
  --util-threshold N        Idle utilization threshold. Default: 10.
  --stable-checks N         Consecutive idle checks required. Default: 2.
  --timeout-seconds N       Give up after N seconds; 0 waits forever.
  --check-once              Check once and exit; COMMAND is optional.
  -h, --help                Show this help.

Example:
  scripts/wait_for_gpus.sh --gpus 0,1,2,3 -- \
    scripts/train_voxbind.sh --model-zoo champion_100m_v2_mask075 \
      --gpus 0,1,2,3 --exp-name voxbind_champion_holo
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

gpus=""
poll_seconds=30
memory_threshold=2048
util_threshold=10
stable_checks=2
timeout_seconds=0
check_once=0
launch_command=()

while (( $# > 0 )); do
    case "$1" in
        --gpus)
            (( $# >= 2 )) || die "--gpus requires a value"
            gpus="$2"; shift 2 ;;
        --poll-seconds)
            (( $# >= 2 )) || die "--poll-seconds requires a value"
            poll_seconds="$2"; shift 2 ;;
        --memory-threshold-mib)
            (( $# >= 2 )) || die "--memory-threshold-mib requires a value"
            memory_threshold="$2"; shift 2 ;;
        --util-threshold)
            (( $# >= 2 )) || die "--util-threshold requires a value"
            util_threshold="$2"; shift 2 ;;
        --stable-checks)
            (( $# >= 2 )) || die "--stable-checks requires a value"
            stable_checks="$2"; shift 2 ;;
        --timeout-seconds)
            (( $# >= 2 )) || die "--timeout-seconds requires a value"
            timeout_seconds="$2"; shift 2 ;;
        --check-once)
            check_once=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift; launch_command=("$@"); break ;;
        *)
            die "unknown argument: $1" ;;
    esac
done

[[ -n "${gpus}" ]] || die "--gpus is required"
[[ "${gpus}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid GPU list: ${gpus}"
for value in \
    "${poll_seconds}" "${memory_threshold}" "${util_threshold}" \
    "${stable_checks}" "${timeout_seconds}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || die "expected non-negative integer, got ${value}"
done
(( poll_seconds > 0 )) || die "--poll-seconds must be positive"
(( stable_checks > 0 )) || die "--stable-checks must be positive"
if (( ! check_once && ${#launch_command[@]} == 0 )); then
    die "a command is required after --"
fi
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"

IFS=',' read -r -a gpu_indices <<< "${gpus}"

gpu_status() {
    local gpu="$1"
    local compute_pids memory_used utilization
    compute_pids="$(
        nvidia-smi -i "${gpu}" \
            --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | sed '/^[[:space:]]*$/d'
    )" || return 2
    read -r memory_used utilization < <(
        nvidia-smi -i "${gpu}" \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits \
            | tr -d ' ' | tr ',' ' '
    ) || return 2
    if [[ -z "${compute_pids}" ]] \
        && (( memory_used < memory_threshold )) \
        && (( utilization < util_threshold )); then
        printf '%s idle (%s MiB, %s%%)' "${gpu}" "${memory_used}" "${utilization}"
        return 0
    fi
    printf '%s busy (%s MiB, %s%%, pids=%s)' \
        "${gpu}" "${memory_used}" "${utilization}" \
        "${compute_pids//$'\n'/,}"
    return 1
}

started="${SECONDS}"
consecutive_idle=0
while true; do
    statuses=()
    all_idle=1
    for gpu in "${gpu_indices[@]}"; do
        if status="$(gpu_status "${gpu}")"; then
            statuses+=("${status}")
        else
            status_code=$?
            statuses+=("${status:-${gpu} query failed}")
            all_idle=0
            if (( status_code == 2 )); then
                echo "ERROR: failed to query GPU ${gpu}" >&2
            fi
        fi
    done

    echo "[$(date --iso-8601=seconds)] $(IFS='; '; echo "${statuses[*]}")"
    if (( check_once )); then
        (( all_idle )) && exit 0
        exit 1
    fi

    if (( all_idle )); then
        (( consecutive_idle += 1 ))
        if (( consecutive_idle >= stable_checks )); then
            echo "[$(date --iso-8601=seconds)] GPUs ${gpus} remained idle; launching"
            printf '  command:'
            printf ' %q' "${launch_command[@]}"
            printf '\n'
            exec "${launch_command[@]}"
        fi
    else
        consecutive_idle=0
    fi

    if (( timeout_seconds > 0 && SECONDS - started >= timeout_seconds )); then
        echo "ERROR: timed out waiting for GPUs ${gpus}" >&2
        exit 124
    fi
    sleep "${poll_seconds}"
done
