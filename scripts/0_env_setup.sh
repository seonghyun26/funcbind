#!/usr/bin/env bash
# Build the FuncBind image on top of the existing VoxBind image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUNCBIND_REPO="$(cd "$HERE/.." && pwd)"
VOXBIND_REPO="$(cd "$FUNCBIND_REPO/.." && pwd)"
DOCKERFILE="${DOCKERFILE:-$VOXBIND_REPO/script/dockerfile-funcbind.sbint}"
VOXBIND_BASE="${VOXBIND_BASE:-voxbind:allinone}"
IMAGE="${IMAGE:-voxbind-funcbind:sb}"
BUILD_NETWORK="${BUILD_NETWORK:-host}"

[ -f "$DOCKERFILE" ] || {
    echo "Dockerfile not found: $DOCKERFILE" >&2
    exit 2
}
[ -f "$FUNCBIND_REPO/requirements.txt" ] || {
    echo "FuncBind submodule is not initialized: $FUNCBIND_REPO" >&2
    exit 2
}
command -v docker >/dev/null || {
    echo "docker is not installed" >&2
    exit 2
}

cmd=(
    docker build
    "--network=$BUILD_NETWORK"
    -f "$DOCKERFILE"
    --build-arg "VOXBIND_BASE=$VOXBIND_BASE"
    -t "$IMAGE"
    "$VOXBIND_REPO"
)

printf 'base=%s  image=%s\n' "$VOXBIND_BASE" "$IMAGE"
if [ -n "${DRY_RUN:-}" ]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${cmd[@]}"
