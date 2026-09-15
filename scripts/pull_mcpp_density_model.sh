#!/usr/bin/env bash
# Download the MCP density-conditioned FuncBind checkpoint from a Dropbox
# shared-file URL. The destination must live on the persistent exps mount.
set -euo pipefail

REPO="${FUNCBIND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MCP_MODEL_DIR:-${FB_PATH:-$REPO/exps/funcbind/fb_mcpp_holo_density}}"
CHECKPOINT="$MODEL_DIR/checkpoint.pth.tar"
MODEL_URL="${MCP_MODEL_URL:-}"
MIN_BYTES=${MCP_MODEL_MIN_BYTES:-1000000000}

say() { echo "[$(date -u +%H:%M:%SZ)] [mcpp-model] $*"; }
die() { say "ERROR: $*" >&2; exit 1; }

checkpoint_ready() {
    [ -s "$CHECKPOINT" ] || return 1
    local size
    size=$(stat -c %s "$CHECKPOINT")
    (( size >= MIN_BYTES )) || die "existing checkpoint is only $size bytes: $CHECKPOINT"
    if [ -n "${MCP_MODEL_SHA256:-}" ]; then
        say "verifying existing checkpoint SHA-256"
        printf '%s  %s\n' "$MCP_MODEL_SHA256" "$CHECKPOINT" | sha256sum --check -
    fi
    say "checkpoint already present: $CHECKPOINT ($size bytes)"
    return 0
}

checkpoint_ready && exit 0

[ -n "$MODEL_URL" ] || die "MCP_MODEL_URL is not set. Pass a direct Dropbox shared-file link for checkpoint.pth.tar."

if [ -n "${CURL_BIN:-}" ]; then
    curl_bin="$CURL_BIN"
elif command -v curl >/dev/null 2>&1; then
    curl_bin="$(command -v curl)"
elif [ -x /opt/conda/envs/funcbind/bin/curl ]; then
    curl_bin=/opt/conda/envs/funcbind/bin/curl
else
    die "curl is not available"
fi

# Dropbox shared links normally end in dl=0. Keep any rlkey and request the
# file body rather than the preview page. Do not print the URL: rlkey grants
# access to the shared file and should stay out of logs.
if [[ "$MODEL_URL" == *dropbox.com* ]]; then
    if [[ "$MODEL_URL" == *dl=0* ]]; then
        MODEL_URL="${MODEL_URL//dl=0/dl=1}"
    elif [[ "$MODEL_URL" != *dl=1* ]]; then
        if [[ "$MODEL_URL" == *\?* ]]; then
            MODEL_URL="${MODEL_URL}&dl=1"
        else
            MODEL_URL="${MODEL_URL}?dl=1"
        fi
    fi
fi

mkdir -p "$MODEL_DIR"
exec 9>"$MODEL_DIR/.checkpoint-download.lock"
flock 9

# A second process may have completed the download while this one waited.
checkpoint_ready && exit 0

part="$CHECKPOINT.part"
say "downloading checkpoint to $part (resumable)"
"$curl_bin" \
    --fail \
    --location \
    --continue-at - \
    --retry 5 \
    --retry-all-errors \
    --retry-delay 5 \
    --output "$part" \
    "$MODEL_URL"

size=$(stat -c %s "$part")
if (( size < MIN_BYTES )); then
    die "downloaded file is only $size bytes; expected at least $MIN_BYTES (is the link a direct checkpoint file?)"
fi

if [ -n "${MCP_MODEL_SHA256:-}" ]; then
    say "verifying SHA-256"
    printf '%s  %s\n' "$MCP_MODEL_SHA256" "$part" | sha256sum --check -
fi

mv "$part" "$CHECKPOINT"
say "checkpoint ready: $CHECKPOINT ($size bytes)"
